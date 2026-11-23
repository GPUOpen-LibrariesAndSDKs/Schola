// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Misc/Optional.h"

// Everything here is in MuJoCo's units and axes; FMjcfConversion converts to Unreal's.

/** A mesh asset declared in an MJCF file. */
struct SCHOLAMJCF_API FMjcfMesh
{
	/** The name attribute, or the file name without its extension when there is none. */
	FString Name;

	/** Absolute path to the mesh file, after include and meshdir resolution. */
	FString File;

	/** Per-axis scale, from the mesh itself or its default class. */
	FVector Scale = FVector::OneVector;

	/** Where the mesh is declared within its file, for example "mujoco/asset[#0]/mesh[wedge]". */
	FString ElementPath;
};

enum class EMjcfTextureType : uint8
{
	TwoD,
	Cube,
	Skybox,
};

enum class EMjcfTextureBuiltin : uint8
{
	/** The texture is read from File. */
	None,
	Gradient,
	Checker,
	Flat,
};

enum class EMjcfTextureMark : uint8
{
	None,
	Edge,
	Cross,
	Random,
};

/** A texture asset. Builtin textures are generated from their colors instead of read from a file. */
struct SCHOLAMJCF_API FMjcfTexture
{
	/** The name attribute, or the file name without its extension when there is none. */
	FString Name;

	EMjcfTextureType Type = EMjcfTextureType::Cube;

	EMjcfTextureBuiltin Builtin = EMjcfTextureBuiltin::None;

	/** Absolute path to the image, after texturedir resolution. Empty for builtin textures. */
	FString File;

	FVector Rgb1 = FVector(0.8);
	FVector Rgb2 = FVector(0.5);
	EMjcfTextureMark Mark = EMjcfTextureMark::None;
	FVector MarkRgb = FVector::ZeroVector;
	int32 Width = 0;
	int32 Height = 0;

	FString ElementPath;
};

struct SCHOLAMJCF_API FMjcfMaterial
{
	FString Name;

	/** The name of the texture, or empty. */
	FString Texture;

	/** Red, green, blue, and alpha, as written in the file. */
	FVector4 Rgba = FVector4(1, 1, 1, 1);

	FVector2D TexRepeat = FVector2D(1, 1);

	/** Whether TexRepeat counts repetitions per meter instead of across the whole object. */
	bool bTexUniform = false;

	double Specular = 0.5;
	double Shininess = 0.5;
	double Reflectance = 0.0;
	double Emission = 0.0;

	FString ElementPath;
};

enum class EMjcfGeomType : uint8
{
	Plane,
	HField,
	Sphere,
	Capsule,
	Ellipsoid,
	Cylinder,
	Box,
	Mesh,
	Sdf,
};

struct SCHOLAMJCF_API FMjcfGeom
{
	/** Empty for unnamed geoms. */
	FString Name;

	EMjcfGeomType Type = EMjcfGeomType::Sphere;

	/**
	 * Sphere: radius. Capsule and cylinder: radius and half-length. Ellipsoid: radii. Box: half-sizes. Plane: half-sizes,
	 * where 0 means infinite, and grid spacing. Unused for meshes.
	 */
	FVector Size = FVector::ZeroVector;

	/**
	 * The name of the mesh, or empty. Primitive geoms can also name a mesh; MuJoCo then sizes and places the primitive to fit
	 * the mesh, which needs the mesh's vertices.
	 */
	FString Mesh;

	/** The name of the material, or empty. */
	FString Material;

	/** MuJoCo uses this color instead of the material's when it differs from the default gray. */
	FVector4 Rgba = DefaultRgba;

	/** MuJoCo shows groups 0 to 2 by default; models often put collision-only geoms in group 3. */
	int32 Group = 0;

	/** Placement relative to the body, including any <frame> between them. */
	FVector Position = FVector::ZeroVector;
	FQuat Rotation = FQuat::Identity;

	/** Two geoms can collide if either's ConType shares a bit with the other's ConAffinity. Both 0 means it never collides. */
	int32 ConType = 1;
	int32 ConAffinity = 1;

	/** Sliding, torsional, and rolling friction. */
	FVector Friction = FVector(1, 0.005, 0.0001);

	/** The geom's mass in kilograms when given, in which case Density is ignored. */
	TOptional<double> Mass;

	/** Kilograms per cubic meter, used for the body's mass when it has no <inertial>. */
	double Density = 1000;

	FString ElementPath;

	static const FVector4 DefaultRgba;
};

enum class EMjcfJointType : uint8
{
	Free,
	Ball,
	Slide,
	Hinge,
};

/** A joint that lets a body move relative to its parent. A body without joints is welded to its parent. */
struct SCHOLAMJCF_API FMjcfJoint
{
	/** Empty for unnamed joints. */
	FString Name;

	EMjcfJointType Type = EMjcfJointType::Hinge;

	/** Position and unit axis in the body's frame. Free joints have neither. */
	FVector Position = FVector::ZeroVector;
	FVector Axis = FVector::UpVector;

	/** Whether Range limits the joint, after applying limited="auto" and <compiler autolimits>. */
	bool bLimited = false;

	/** Hinge: radians. Slide: meters. Ball: Range.Y is the largest angle away from the rest orientation, in radians. */
	FVector2D Range = FVector2D::ZeroVector;

	/** The joint position at which the body is placed as written; radians for hinges, meters for slides. */
	double Ref = 0;

	/** The joint position at which the spring has no force. */
	double SpringRef = 0;

	double Stiffness = 0;
	double Damping = 0;

	/** Extra rotor inertia added to the joint. */
	double Armature = 0;

	/** Dry friction: the force or torque needed to start the joint moving. */
	double FrictionLoss = 0;

	FString ElementPath;
};

/** A body's mass and inertia, given with <inertial> rather than computed from its geoms. */
struct SCHOLAMJCF_API FMjcfInertial
{
	/** Center of mass, and the rotation of the principal axes, in the body's frame. */
	FVector Position = FVector::ZeroVector;
	FQuat Rotation = FQuat::Identity;

	/** Kilograms. */
	double Mass = 0;

	/** Principal moments of inertia about the rotated axes, in kg·m². */
	FVector DiagInertia = FVector::ZeroVector;
};

struct SCHOLAMJCF_API FMjcfBody
{
	/** Empty for unnamed bodies and for the world body. */
	FString Name;

	/** Placement relative to the parent body, including any <frame> between them. */
	FVector Position = FVector::ZeroVector;
	FQuat Rotation = FQuat::Identity;

	/** Unset when MuJoCo computes the mass and inertia from the geoms. */
	TOptional<FMjcfInertial> Inertial;

	/** Empty when the body is welded to its parent. */
	TArray<FMjcfJoint> Joints;

	/** Mocap bodies are moved by setting their pose, not by physics. */
	bool bMocap = false;

	/** The fraction of the body's weight that is cancelled, from 0 (none) to 1 (floats). */
	double GravComp = 0;

	TArray<FMjcfGeom> Geoms;
	TArray<FMjcfBody> Bodies;

	FString ElementPath;
};

enum class EMjcfActuatorDynamics : uint8
{
	None,
	Integrator,
	Filter,
	FilterExact,
	Muscle,
	User,
};

enum class EMjcfActuatorGain : uint8
{
	Fixed,
	Affine,
	Muscle,
	User,
};

enum class EMjcfActuatorBias : uint8
{
	None,
	Affine,
	Muscle,
	User,
};

/**
 * An actuator that drives a joint, in MuJoCo's general form. The force along the joint is Gear * (gain * control + bias),
 * where gain is GainPrm.X, plus GainPrm.Y * position + GainPrm.Z * velocity for affine gains, and bias is
 * BiasPrm.X + BiasPrm.Y * position + BiasPrm.Z * velocity for affine biases. <motor>, <position>, and <velocity> set these
 * the way MuJoCo does, so a position actuator has GainPrm.X = kp and BiasPrm = (0, -kp, -kv).
 */
struct SCHOLAMJCF_API FMjcfActuator
{
	/** Empty for unnamed actuators. */
	FString Name;

	/** The element it was written as, such as "position". */
	FString Kind;

	/** The name of the joint it drives. */
	FString Joint;

	double Gear = 1;

	EMjcfActuatorDynamics Dynamics = EMjcfActuatorDynamics::None;

	/** The time constant, in seconds, of filter dynamics. */
	double TimeConstant = 1;

	EMjcfActuatorGain GainType = EMjcfActuatorGain::Fixed;
	FVector GainPrm = FVector(1, 0, 0);

	EMjcfActuatorBias BiasType = EMjcfActuatorBias::None;
	FVector BiasPrm = FVector::ZeroVector;

	/**
	 * When positive, a <position> actuator's damping is this fraction of critical damping, which depends on the joint's
	 * inertia and so isn't known until the bodies' masses are; BiasPrm.Z is 0 until then.
	 */
	double DampRatio = 0;

	/** Whether the control is clamped to CtrlRange, after applying ctrllimited="auto" and <compiler autolimits>. */
	bool bCtrlLimited = false;
	FVector2D CtrlRange = FVector2D::ZeroVector;

	/** Whether the force is clamped to ForceRange, after applying forcelimited="auto" and <compiler autolimits>. */
	bool bForceLimited = false;
	FVector2D ForceRange = FVector2D::ZeroVector;

	FString ElementPath;
};

/** A pair of bodies whose geoms never collide with each other. */
struct SCHOLAMJCF_API FMjcfContactExclude
{
	FString Body1;
	FString Body2;
};

/** How <compiler inertiafromgeom> decides between a body's <inertial> and its geoms. */
enum class EMjcfInertiaFromGeom : uint8
{
	/** Always use <inertial>. */
	False,
	/** Always compute from the geoms, ignoring <inertial>. */
	True,
	/** Use <inertial> when the body has one, and the geoms otherwise. */
	Auto,
};

/** The parts of an MJCF file and the files it includes that the importer uses. */
struct SCHOLAMJCF_API FMjcfModel
{
	FString SourceFile;

	/** The model attribute of <mujoco>, or empty. */
	FString ModelName;

	TArray<FMjcfMesh> Meshes;
	TArray<FMjcfTexture> Textures;
	TArray<FMjcfMaterial> Materials;

	/** Every <worldbody> section, merged. It never moves; its geoms are usually the floor. */
	FMjcfBody WorldBody;

	TArray<FMjcfActuator> Actuators;
	TArray<FMjcfContactExclude> ContactExcludes;

	/** Meters per second squared, from <option gravity>. */
	FVector Gravity = FVector(0, 0, -9.81);

	/** Seconds per physics step, from <option timestep>. */
	double Timestep = 0.002;

	EMjcfInertiaFromGeom InertiaFromGeom = EMjcfInertiaFromGeom::Auto;

	const FMjcfMesh* FindMesh(const FString& Name) const;
	const FMjcfTexture* FindTexture(const FString& Name) const;
	const FMjcfMaterial* FindMaterial(const FString& Name) const;
};
