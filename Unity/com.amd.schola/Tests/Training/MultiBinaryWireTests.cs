// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using AMD.Schola.Training;
using NUnit.Framework;
using UnityEngine;

namespace AMD.Schola.Tests.Training
{
    public sealed class MultiBinaryWireTests
    {
        [Test]
        public void Space_DefaultAndShapedConstructorsSetShape()
        {
            Assert.That(new MultiBinarySpace().Shape, Is.EqualTo(0));
            Assert.That(new MultiBinarySpace(4).Shape, Is.EqualTo(4));
        }

        [Test]
        public void Validate_AcceptsMatchingLengthIncludingEmpty()
        {
            Assert.DoesNotThrow(() =>
                TrainingWire.Validate(
                    new MultiBinaryPoint { Values = { true, false, true } },
                    new MultiBinarySpace(3)
                )
            );
            Assert.DoesNotThrow(() =>
                TrainingWire.Validate(new MultiBinaryPoint(), new MultiBinarySpace())
            );
        }

        [Test]
        public void Validate_RejectsWrongTypeOrLength()
        {
            Assert.Throws<ArgumentException>(() =>
                TrainingWire.Validate(new DiscretePoint { Value = 0 }, new MultiBinarySpace(1))
            );
            Assert.Throws<ArgumentException>(() =>
                TrainingWire.Validate(
                    new MultiBinaryPoint { Values = { true, false } },
                    new MultiBinarySpace(3)
                )
            );
        }

        [Test]
        public void Wire_EncodesSpaceShapeAndRoundTripsPoints()
        {
            var spaceNodes = TrainingWire.EncodeSpace(new MultiBinarySpace(3));
            Assert.That(spaceNodes, Has.Length.EqualTo(1));
            Assert.That(spaceNodes[0].kind, Is.EqualTo("multi_binary"));
            Assert.That(spaceNodes[0].shape, Is.EqualTo(new[] { 3 }));

            var emptySpace = TrainingWire.EncodeSpace(new MultiBinarySpace());
            Assert.That(emptySpace[0].shape, Is.EqualTo(new[] { 0 }));

            var point = new MultiBinaryPoint { Values = { true, false, true } };
            var decoded = (MultiBinaryPoint)
                TrainingWire.DecodePoint(TrainingWire.EncodePoint(point));
            Assert.That(decoded.Values, Is.EqualTo(new[] { true, false, true }));

            var emptyDecoded = (MultiBinaryPoint)
                TrainingWire.DecodePoint(TrainingWire.EncodePoint(new MultiBinaryPoint()));
            Assert.That(emptyDecoded.Values, Is.Empty);
        }

        [Test]
        public void Wire_RoundTripsNestedDictThroughJsonUtility()
        {
            var source = new DictPoint
            {
                Points =
                {
                    ["flags"] = new MultiBinaryPoint { Values = { false, true } },
                    ["choice"] = new DiscretePoint { Value = 1 },
                },
            };
            var nodes = TrainingWire.EncodePoint(source);
            var request = JsonUtility.FromJson<TrainingRequest>(
                JsonUtility.ToJson(new TrainingRequest { action = nodes })
            );
            var decoded = (DictPoint)TrainingWire.DecodePoint(request.action);
            Assert.That(
                ((MultiBinaryPoint)decoded.Points["flags"]).Values,
                Is.EqualTo(new[] { false, true })
            );
            Assert.That(((DiscretePoint)decoded.Points["choice"]).Value, Is.EqualTo(1));
        }

        [Test]
        public void EncodeSpace_RejectsNegativeShape()
        {
            Assert.Throws<NotSupportedException>(() =>
                TrainingWire.EncodeSpace(new MultiBinarySpace(-1))
            );
        }
    }
}
