#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "CampusScene.generated.h"
class UProceduralMeshComponent;

struct FCampusNode
{
    FVector2f Position;
    TArray<int32> Links;
};

/** Georeferenced CAD footprint blockout. Geometry is rebuilt in editor and at runtime. */
UCLASS()
class ACampusScene : public AActor
{
    GENERATED_BODY()
public:
    ACampusScene();
    virtual void PostRegisterAllComponents() override;
    virtual void OnConstruction(const FTransform& Transform) override;
    virtual void BeginPlay() override;
    bool LoadCampus();
    bool LoadDestinationHighlights(const TMap<FString, int32>& BuildingKeys, FString& Error);
    void SetActiveDestinations(const TBitArray<>& Active);
    TConstArrayView<FVector2f> GetDestinationFootprint(int32 Building) const { return DestinationFootprints[Building]; }
    FVector2f MapOrigin = FVector2f::ZeroVector;
    FVector2f MapSize = FVector2f::ZeroVector;
    TArray<FCampusNode> Nodes;
    float Extent = 80000;
    int32 BuildingCount = 0;
    FString LoadError;
private:
    UPROPERTY(VisibleAnywhere, Transient) TObjectPtr<UProceduralMeshComponent> Mesh;
    UPROPERTY(Transient) TObjectPtr<UProceduralMeshComponent> DestinationMesh;
    TArray<TArray<FVector2f>> DestinationFootprints;
    TBitArray<> VisibleDestinations;
};
