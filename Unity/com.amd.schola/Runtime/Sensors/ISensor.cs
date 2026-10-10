// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

namespace AMD.Schola
{
    /// <summary>
    /// Sensor contract mirroring Unreal IScholaSensor:
    /// InitSensor, GetObservationSpace, CollectObservations.
    /// </summary>
    public interface ISensor
    {
        void InitSensor();

        Space GetObservationSpace();

        Point CollectObservations();
    }
}
