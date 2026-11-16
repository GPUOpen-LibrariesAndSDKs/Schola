// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Linq;

namespace AMD.Schola.Training
{
    // Flat trees avoid Unity JsonUtility's recursive serialization depth limit.
    // Node zero is the root; each subsequent node refers to an earlier parent.
    [Serializable]
    public sealed class TrainingNode
    {
        public int parent = -1;
        public string key;
        public string kind;
        public int[] shape;
        public float[] values;
        public bool[] boolValues;
        public TrainingDimension[] dimensions;
        public int high;
        public int value;
    }

    [Serializable]
    public sealed class TrainingDimension
    {
        public bool hasLow;
        public float low;
        public bool hasHigh;
        public float high;
    }

    [Serializable]
    public sealed class TrainingEntry
    {
        public string key;
        public string value;
    }

    [Serializable]
    public sealed class TrainingRequest
    {
        public int version = 1;
        public string command;
        public int autoreset;
        public bool hasSeed;
        public int seed;
        public TrainingEntry[] options;
        public TrainingNode[] action;
    }

    [Serializable]
    public sealed class TrainingResponse
    {
        public int version = 1;
        public bool ok = true;
        public string error;
        public TrainingNode[] observationSpace;
        public TrainingNode[] actionSpace;
        public TrainingNode[] observations;
        public float reward;
        public bool terminated;
        public bool truncated;
        public TrainingEntry[] info;
        public bool hasInitial;
        public TrainingNode[] initialObservations;
        public TrainingEntry[] initialInfo;
    }

    /// <summary>Conversions for the Box, Dict, Discrete and MultiBinary types supported by Unity.</summary>
    public static class TrainingWire
    {
        public const int Version = 1;
        public const int MaxMessageBytes = 4 * 1024 * 1024;

        public static TrainingNode[] EncodeSpace(Space space)
        {
            var nodes = new List<TrainingNode>();
            AddSpace(nodes, space, -1, null);
            return nodes.ToArray();
        }

        static void AddSpace(List<TrainingNode> nodes, Space space, int parent, string key)
        {
            var node = new TrainingNode { parent = parent, key = key };
            var index = nodes.Count;
            nodes.Add(node);
            if (space is DictSpace dict)
            {
                node.kind = "dict";
                foreach (var entry in dict.Spaces.OrderBy(x => x.Key, StringComparer.Ordinal))
                    AddSpace(nodes, entry.Value, index, entry.Key);
            }
            else if (space is BoxSpace box)
            {
                node.kind = "box";
                node.shape = Shape(box.Shape, box.Dimensions.Count);
                node.dimensions = box
                    .Dimensions.Select(d => new TrainingDimension
                    {
                        hasLow = d.Low.HasValue,
                        low = d.Low ?? 0,
                        hasHigh = d.High.HasValue,
                        high = d.High ?? 0,
                    })
                    .ToArray();
                foreach (var d in node.dimensions)
                {
                    if (
                        (d.hasLow && !Finite(d.low))
                        || (d.hasHigh && !Finite(d.high))
                        || (d.hasLow && d.hasHigh && d.low > d.high)
                    )
                        throw new ArgumentException("Invalid Box bounds.");
                }
            }
            else if (space is DiscreteSpace discrete && discrete.High > 0)
            {
                node.kind = "discrete";
                node.high = discrete.High;
            }
            else if (space is MultiBinarySpace multiBinary && multiBinary.Shape >= 0)
            {
                // Proto MultiBinarySpace.shape is a scalar; wire as a one-element array
                // so TrainingNode can keep Box's int[] shape field.
                node.kind = "multi_binary";
                node.shape = new[] { multiBinary.Shape };
            }
            else
                throw new NotSupportedException(
                    "Training supports Box, Dict, positive Discrete and non-negative MultiBinary spaces."
                );
        }

        public static TrainingNode[] EncodePoint(Point point)
        {
            var nodes = new List<TrainingNode>();
            AddPoint(nodes, point, -1, null);
            return nodes.ToArray();
        }

        static void AddPoint(List<TrainingNode> nodes, Point point, int parent, string key)
        {
            var node = new TrainingNode { parent = parent, key = key };
            var index = nodes.Count;
            nodes.Add(node);
            if (point is DictPoint dict)
            {
                node.kind = "dict";
                foreach (var entry in dict.Points.OrderBy(x => x.Key, StringComparer.Ordinal))
                    AddPoint(nodes, entry.Value, index, entry.Key);
            }
            else if (point is BoxPoint box)
            {
                node.kind = "box";
                node.values = box.Values.ToArray();
                node.shape = Shape(box.Shape, box.Values.Count);
                if (node.values.Any(v => !Finite(v)))
                    throw new ArgumentException("Box values must be finite.");
            }
            else if (point is DiscretePoint discrete)
            {
                node.kind = "discrete";
                node.value = discrete.Value;
            }
            else if (point is MultiBinaryPoint multiBinary)
            {
                node.kind = "multi_binary";
                node.boolValues = multiBinary.Values.ToArray();
            }
            else
                throw new NotSupportedException(
                    "Training supports Box, Dict, Discrete and MultiBinary points."
                );
        }

        public static Point DecodePoint(TrainingNode[] nodes)
        {
            if (nodes == null || nodes.Length == 0 || nodes.Length > 10000 || nodes[0].parent != -1)
                throw new ArgumentException("Invalid action tree.");
            var points = new Point[nodes.Length];
            for (var i = 0; i < nodes.Length; i++)
            {
                var node = nodes[i] ?? throw new ArgumentException("Null action node.");
                switch (node.kind)
                {
                    case "dict":
                        points[i] = new DictPoint();
                        break;
                    case "box":
                        var values =
                            node.values ?? throw new ArgumentException("Missing Box values.");
                        if (values.Any(v => !Finite(v)))
                            throw new ArgumentException("Actions must be finite.");
                        points[i] = new BoxPoint
                        {
                            Values = values.ToList(),
                            Shape = Shape(node.shape, values.Length).ToList(),
                        };
                        break;
                    case "discrete":
                        points[i] = new DiscretePoint { Value = node.value };
                        break;
                    case "multi_binary":
                        points[i] = new MultiBinaryPoint
                        {
                            Values = (node.boolValues ?? Array.Empty<bool>()).ToList(),
                        };
                        break;
                    default:
                        throw new ArgumentException("Unsupported action type.");
                }
                if (i == 0)
                    continue;
                if (
                    node.parent < 0
                    || node.parent >= i
                    || !(points[node.parent] is DictPoint dict)
                    || node.key == null
                    || dict.Points.ContainsKey(node.key)
                )
                    throw new ArgumentException("Invalid action parent or duplicate key.");
                dict.Points.Add(node.key, points[i]);
            }
            return points[0];
        }

        /// <summary>Validate the complete action before any actuator can mutate the scene.</summary>
        public static void Validate(Point point, Space space)
        {
            if (space is DictSpace dict && point is DictPoint values)
            {
                if (dict.Spaces.Count != values.Points.Count)
                    throw new ArgumentException("Action keys do not match the declared space.");
                foreach (var entry in dict.Spaces)
                {
                    if (!values.Points.TryGetValue(entry.Key, out var nested))
                        throw new ArgumentException($"Missing action key '{entry.Key}'.");
                    Validate(nested, entry.Value);
                }
                return;
            }
            if (space is BoxSpace box && point is BoxPoint vector)
            {
                if (
                    vector.Values.Count != box.Dimensions.Count
                    || !Shape(vector.Shape, vector.Values.Count)
                        .SequenceEqual(Shape(box.Shape, box.Dimensions.Count))
                )
                    throw new ArgumentException(
                        "Action shape does not match the declared Box space."
                    );
                for (var i = 0; i < vector.Values.Count; i++)
                {
                    var v = vector.Values[i];
                    var d = box.Dimensions[i];
                    if (
                        !Finite(v)
                        || (d.Low.HasValue && v < d.Low)
                        || (d.High.HasValue && v > d.High)
                    )
                        throw new ArgumentException("Action is outside the declared Box bounds.");
                }
                return;
            }
            if (
                space is DiscreteSpace discrete
                && point is DiscretePoint value
                && value.Value >= 0
                && value.Value < discrete.High
            )
                return;
            if (space is MultiBinarySpace multiBinary && point is MultiBinaryPoint flags)
            {
                if (flags.Values.Count != multiBinary.Shape)
                    throw new ArgumentException(
                        "Action length does not match the declared MultiBinary space."
                    );
                return;
            }
            throw new ArgumentException("Action does not match the declared space.");
        }

        public static TrainingEntry[] Entries(IReadOnlyDictionary<string, string> values) =>
            values.Select(x => new TrainingEntry { key = x.Key, value = x.Value }).ToArray();

        static int[] Shape(IEnumerable<int> shape, int count)
        {
            var result = shape?.ToArray() ?? Array.Empty<int>();
            if (result.Length == 0)
                result = new[] { count };
            long product = 1;
            foreach (var size in result)
            {
                if (size <= 0 || product > int.MaxValue / size)
                    throw new ArgumentException("Invalid Box shape.");
                product *= size;
            }
            if (product != count)
                throw new ArgumentException("Box shape does not match its values.");
            return result;
        }

        static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
