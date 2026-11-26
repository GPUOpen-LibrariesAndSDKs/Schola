// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "Engine/GameViewportClient.h"
#include "GameFramework/GameModeBase.h"
#include "GameFramework/HUD.h"
#include "CrowdGameMode.generated.h"

class ACrowdComparisonHost;
class FCampusMigrationDay;
class FCampusTurnDay;
class UTexture2D;

/** Spawns the campus crowd host. */
UCLASS()
class ACrowdGameMode : public AGameModeBase
{
    GENERATED_BODY()
public:
    ACrowdGameMode();
    virtual void StartPlay() override;
    UPROPERTY() TObjectPtr<ACrowdComparisonHost> Host;
};

/** Allocates the right half of the viewport to the real 3D scene. */
UCLASS()
class UCrowdViewportClient : public UGameViewportClient
{
    GENERATED_BODY()
public:
    virtual void Draw(FViewport* Viewport, FCanvas* SceneCanvas) override;
    virtual void LayoutPlayers() override;
    virtual void PostRender(UCanvas* Canvas) override;
private:
    bool bSplit = false;
};

/** Statistics overlay shown in both modes. */
UCLASS()
class ACrowdHUD : public AHUD
{
    GENERATED_BODY()
public:
    virtual void DrawHUD() override;
    void DrawSplitDemo(UCanvas& FullCanvas, const ACrowdComparisonHost& Host);

private:
    void DrawNavigationMap(const FCampusMigrationDay& Day, int32 StudentCount);
    UPROPERTY(Transient) TObjectPtr<UTexture2D> NavigationMapTexture;
    UPROPERTY(Transient) TObjectPtr<UTexture2D> TurnMapTexture;
    UPROPERTY(Transient) TObjectPtr<UTexture2D> HighlightTexture;
    TBitArray<> MapMarkers, PerspectiveMarkers;
};
