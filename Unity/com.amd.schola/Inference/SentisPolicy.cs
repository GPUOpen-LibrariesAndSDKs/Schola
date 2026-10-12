// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace AMD.Schola
{
    /// <summary>
    /// Runs an ONNX policy exported by Schola through Unity Inference Engine (Sentis).
    /// </summary>
    /// <remarks>
    /// Schola's exporter names each model input after an observation key and each model output
    /// after an action key, so both spaces must be <see cref="DictSpace"/> and their keys must
    /// match the model's input and output names. Supports <see cref="BoxSpace"/> observations 
    /// or <see cref="DiscreteSpace"/> actions, one agent per call.
    /// </remarks>
    public sealed class SentisPolicy : IPolicy, IDisposable {
        readonly ModelAsset m_ModelAsset;
        readonly BackendType m_Backend;

        Worker m_Worker;
        DictSpace m_ObservationSpace;
        DictSpace m_ActionSpace;

        public SentisPolicy(ModelAsset modelAsset, BackendType backend = BackendType.CPU) {
            m_ModelAsset = modelAsset;
            m_Backend = backend;
        }

        public bool IsInitialized {
          get {
            return m_Worker != null;
          }   
        } 

        public bool Init(InteractionDefinition definition) {
            Dispose();

            if (m_ModelAsset == null) {
                Debug.LogError($"{nameof(SentisPolicy)}.Init(): No model asset assigned.");
                return false;
            }

            if (definition == null) {
                Debug.LogError($"{nameof(SentisPolicy)}.Init(): No interaction definition given.");
                return false;
            }

            if (!((definition.ObservationSpace is DictSpace observationSpace) && (definition.ActionSpace is DictSpace actionSpace))) {
                Debug.LogError($"{nameof(SentisPolicy)}.Init(): Observation and action spaces must both be DictSpace.");
                return false;
            }

            var model = ModelLoader.Load(m_ModelAsset);
            if (!(ValidateObservations(model, observationSpace) && ValidateActions(model, actionSpace))) {
                return false;
            }

            m_ObservationSpace = observationSpace;
            m_ActionSpace = actionSpace;
            m_Worker = new Worker(model, m_Backend);
            return true;
        }

        public bool Think(Point observations, out Point actions) {
            actions = null;

            if (m_Worker == null) {
                Debug.LogError($"{nameof(SentisPolicy)}.Think(): Init must succeed before Think.");
                return false;
            }

            if (!(observations is DictPoint observationDict)) {
                Debug.LogError($"{nameof(SentisPolicy)}.Think(): Observations must be a DictPoint.");
                return false;
            }
            var inputs = new List<Tensor>();

            try {
                foreach (var key in m_ObservationSpace.Spaces.Keys) {
                    if (!(observationDict.Points.TryGetValue(key, out var point) && (point is BoxPoint box))) {
                        Debug.LogError($"{nameof(SentisPolicy)}.Think(): Missing BoxPoint observation '{key}'.");
                        return false;
                    }

                    var input = new Tensor<float>(BatchedShape(box.Shape, box.Values.Count), box.Values.ToArray());
                    inputs.Add(input);
                    m_Worker.SetInput(key, input);
                }

                m_Worker.Schedule();

                var actionDict = new DictPoint();
                foreach (var entry in m_ActionSpace.Spaces) {
                    if (!TryReadAction(entry.Key, entry.Value, out var action)) {
                        return false;
                    }

                    actionDict.Points[entry.Key] = action;
                }

                actions = actionDict;
                return true;
            }
            finally {
                foreach (var input in inputs) {
                    input.Dispose();
                }
            }
        }

        public void Dispose() {
            if (m_Worker != null) {
                m_Worker.Dispose();
                m_Worker = null;
            }
        }

        bool TryReadAction(string key, Space space, out Point action) {
            action = null;

            switch (space) {
                case DiscreteSpace discrete: {
                    if (!(m_Worker.PeekOutput(key) is Tensor<int> output)) {
                        Debug.LogError($"{nameof(SentisPolicy)}.Think(): Output '{key}' is not an integer tensor.");
                        return false;
                    }

                    var value = output.DownloadToArray()[0];
                    if (value < 0 || value >= discrete.High) {
                        Debug.LogError($"{nameof(SentisPolicy)}.Think(): Output '{key}' = {value} is outside [0, {discrete.High}).");
                        return false;
                    }

                    action = new DiscretePoint { Value = value };
                    return true;
                }
                case BoxSpace box: {
                    if (!(m_Worker.PeekOutput(key) is Tensor<float> output)) {
                        Debug.LogError($"{nameof(SentisPolicy)}.Think(): Output '{key}' is not a float tensor.");
                        return false;
                    }

                    action = new BoxPoint {
                        Values = new List<float>(output.DownloadToArray()),
                        Shape = new List<int>(box.Shape),
                    };
                    return true;
                }
                default:
                    Debug.LogError($"{nameof(SentisPolicy)}.Think(): Unsupported action space '{TypeName(space)}' for '{key}'.");
                    return false;
            }
        }

        static bool ValidateObservations(Model model, DictSpace observationSpace) {
            var modelInputs = model.inputs.Select(input => input.name).ToHashSet();

            foreach (var entry in observationSpace.Spaces) {
                if (!(entry.Value is BoxSpace)) {
                    Debug.LogError($"{nameof(SentisPolicy)}.Init(): Observation '{entry.Key}' must be a BoxSpace, got {TypeName(entry.Value)}.");
                    return false;
                }

                if (!modelInputs.Contains(entry.Key)) {
                    Debug.LogError($"{nameof(SentisPolicy)}.Init(): Model has no input named '{entry.Key}'. Model inputs: {string.Join(", ", modelInputs)}.");
                    return false;
                }
            }

            foreach (var name in modelInputs) {
                if (!observationSpace.Spaces.ContainsKey(name)) {
                    Debug.LogError($"{nameof(SentisPolicy)}.Init(): Model expects input '{name}' but no observation has that key.");
                    return false;
                }
            }

            return true;
        }

        static bool ValidateActions(Model model, DictSpace actionSpace) {
            var modelOutputs = model.outputs.Select(output => output.name).ToHashSet();

            foreach (var key in actionSpace.Spaces.Keys) {
                if (!modelOutputs.Contains(key)) {
                    Debug.LogError($"{nameof(SentisPolicy)}.Init(): Model has no output named '{key}'. Model outputs: {string.Join(", ", modelOutputs)}.");
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Prepends a batch dimension of 1 to the point's shape.
        /// </summary>
        static TensorShape BatchedShape(List<int> shape, int count)
        {
            var dims = new List<int> { 1 };
            if (shape.Count == 0)
            {
                dims.Add(count);
            }
            else
            {
                dims.AddRange(shape);
            }

            return new TensorShape(dims.ToArray());
        }

        /// <summary>
        /// Class name of <paramref name="value"/> for error messages, or "null".
        /// </summary>
        static string TypeName(object value)
        {
            if (value == null)
            {
                return "null";
            }

            return value.GetType().Name;
        }
    }
}
