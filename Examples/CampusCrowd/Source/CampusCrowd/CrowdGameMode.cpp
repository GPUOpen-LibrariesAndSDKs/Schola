// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "CrowdGameMode.h"
#include "CrowdComparisonHost.h"
#include "CampusScene.h"
#include "CanvasItem.h"
#include "Engine/LocalPlayer.h"
#include "ImageUtils.h"
#include "SceneView.h"
#include "TextureResource.h"
#include "Misc/Paths.h"
#include "Engine/Canvas.h"
#include "Engine/Engine.h"
#include "Engine/Font.h"
#include "Engine/Texture2D.h"
#include "Engine/World.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "UnrealClient.h"

ACrowdGameMode::ACrowdGameMode()
{
    DefaultPawnClass = nullptr;
    HUDClass = ACrowdHUD::StaticClass();
}

void ACrowdGameMode::StartPlay()
{
    Host = GetWorld()->SpawnActor<ACrowdComparisonHost>();
    Super::StartPlay();
}

void UCrowdViewportClient::Draw(FViewport* Viewport, FCanvas* SceneCanvas)
{
    const auto* Mode = GetWorld() ? GetWorld()->GetAuthGameMode<ACrowdGameMode>() : nullptr;
    const bool bNewSplit = Mode && Mode->Host && Mode->Host->IsSplitDemo();
    if (bSplit != bNewSplit)
    {
        bSplit = bNewSplit;
        LayoutPlayers();
    }
    Super::Draw(Viewport, SceneCanvas);
}

void UCrowdViewportClient::LayoutPlayers()
{
    Super::LayoutPlayers();
    if (bSplit && GetWorld())
        if (auto* Local = GetWorld()->GetFirstLocalPlayerFromController())
        {
            Local->Origin = FVector2D(.5, 0);
            Local->Size = FVector2D(.5, 1);
        }
}

void UCrowdViewportClient::PostRender(UCanvas* DrawCanvas)
{
    if (bSplit && DrawCanvas && !(FScreenshotRequest::IsScreenshotRequested() &&
        FParse::Param(FCommandLine::Get(), TEXT("CleanScreenshot"))))
    {
        const auto* Mode = GetWorld()->GetAuthGameMode<ACrowdGameMode>();
        auto* Player = GetWorld()->GetFirstPlayerController();
        auto* HUD = Player ? Cast<ACrowdHUD>(Player->GetHUD()) : nullptr;
        if (HUD && Mode && Mode->Host && Viewport)
        {
            const FIntPoint Size = Viewport->GetSizeXY();
            DrawCanvas->Init(Size.X, Size.Y, nullptr, DrawCanvas->Canvas);
            HUD->DrawSplitDemo(*DrawCanvas, *Mode->Host);
        }
    }
    Super::PostRender(DrawCanvas);
}

void ACrowdHUD::DrawSplitDemo(UCanvas& FullCanvas, const ACrowdComparisonHost& Host)
{
    TGuardValue<TObjectPtr<UCanvas>> Fullscreen(Canvas, &FullCanvas);
    const auto& Day = *Host.GetMigrationDay()->GetTurns();
    const auto& Campus = *Host.GetCampus();
    if (!TurnMapTexture)
        TurnMapTexture = FImageUtils::ImportFileAsTexture2D(FPaths::ProjectContentDir() / TEXT("Campus/turn_campus_map.png"));

    if (!HighlightTexture)
    {
        const uint8 WhitePixel[] = {255, 255, 255, 255};
        HighlightTexture = UTexture2D::CreateTransient(1, 1, PF_B8G8R8A8, NAME_None, MakeArrayView(WhitePixel));
        if (HighlightTexture) HighlightTexture->UpdateResource();
    }

    const float Width = FullCanvas.ClipX, Height = FullCanvas.ClipY, Mid = Width * .5f;
    const float UI = FMath::Clamp(FMath::Min(Width / 1600.f, Height / 900.f), .5f, 1.6f);
    const float Header = 52.f * UI, Footer = 48.f * UI;
    const FLinearColor Ink(FColor(14, 25, 32)), Panel(FColor(23, 41, 49)), Muted(FColor(177, 202, 207));
    const FLinearColor Accent(FColor(98, 226, 189)), White(FColor(235, 245, 247));
    auto* Font = GEngine->GetLargeFont();
    DrawRect(Ink, 0, 0, Mid, Height);
    DrawRect(Panel, 0, 0, Width, Header);
    DrawRect(Panel, 0, Height - Footer, Width, Footer);
    DrawRect(Ink, Mid - 1, Header, 2, Height - Header - Footer);
    DrawText(TEXT("CampusCrowd"), White, 16 * UI, 12 * UI, Font, UI * 1.2f);

    const auto Tab = [&](const TCHAR* Label, float X, bool bSelected)
    {
        DrawRect(bSelected ? FLinearColor(FColor(39, 72, 77)) : Ink, X, 8 * UI, 147 * UI, 34 * UI);
        DrawText(Label, bSelected ? Accent : Muted, X + 8 * UI, 15 * UI, Font, UI * .72f);
    };
    Tab(TEXT("F1  Four students"), Width - 316 * UI, Host.IsFourDemo());
    Tab(TEXT("F2  Scaling test"), Width - 160 * UI, !Host.IsFourDemo());
    DrawText(TEXT("Pedestrian map"), Muted, 16 * UI, Header + 7 * UI, Font, UI * .84f);
    DrawRect(Panel, Mid + 12 * UI, Header + 5 * UI, 105 * UI, 25 * UI);
    DrawText(TEXT("3D campus"), Muted, Mid + 18 * UI, Header + 7 * UI, Font, UI * .84f);
    const float MapTop = Header + 33.f * UI, MapBottom = Height - Footer - 9.f * UI;
    const float Scale = FMath::Min((Mid - 22 * UI) / Campus.MapSize.X, (MapBottom - MapTop) / Campus.MapSize.Y);
    const FVector2f Origin((Mid - Campus.MapSize.X * Scale) * .5f,
        (MapTop + MapBottom - Campus.MapSize.Y * Scale) * .5f);
    const auto Project = [&](FVector2f Meters) { return FVector2D(Origin + (Meters - Campus.MapOrigin) * Scale); };
    if (TurnMapTexture)
        DrawTexture(TurnMapTexture, Origin.X, Origin.Y, Campus.MapSize.X * Scale,
            Campus.MapSize.Y * Scale, 0, 0, 1, 1, FLinearColor::White, BLEND_Opaque);

    for (TConstSetBitIterator<> It(Host.GetActiveDestinations()); It; ++It)
    {
        const int32 Key = It.GetIndex();
        const auto Points = Campus.GetDestinationFootprint(Key);
        TArray<FCanvasUVTri> Triangles;
        const FLinearColor Tint = Day.DestinationColor(Key).CopyWithNewOpacity(.82f);
        for (int32 I = 0; I < Points.Num(); I += 3)
        {
            auto& Triangle = Triangles.AddDefaulted_GetRef();
            Triangle.V0_Pos = Project(Points[I]);
            Triangle.V1_Pos = Project(Points[I+1]);
            Triangle.V2_Pos = Project(Points[I+2]);
            Triangle.V0_Color = Triangle.V1_Color = Triangle.V2_Color = Tint;
        }
        if (!HighlightTexture || !HighlightTexture->GetResource()) continue;
        FCanvasTriangleItem Item(Triangles, HighlightTexture->GetResource());
        Item.BlendMode = SE_BLEND_Translucent;
        FullCanvas.DrawItem(Item);
    }

    if (Host.IsFourDemo())
    {
        for (int32 I = 0; I < Day.GetStudentCount(); ++I)
        {
            const auto Path = Day.GetReferencePath(I);
            const auto Color = Day.DestinationColor(Day.GetStopColorKey(Day.GetTargetStop(I)));
            for (int32 J = 1; J < Path.Num(); ++J)
            {
                const auto A = Project(Path[J-1]), B = Project(Path[J]);
                DrawLine(A.X, A.Y, B.X, B.Y, Color.CopyWithNewOpacity(.65f), 2.f * UI);
            }
        }
    }

    // Limit the overlay to one marker per small screen cell at large cohort sizes.
    constexpr int32 CellPixels = 4;
    const int32 Columns = FMath::Max(1, FMath::CeilToInt(Mid / CellPixels));
    const int32 Rows = FMath::Max(1, FMath::CeilToInt(Height / CellPixels));
    MapMarkers.Init(false, Columns * Rows);
    PerspectiveMarkers.Init(false, Columns * Rows);
    const auto DrawDot = [&](FVector2D Point, float Left, TBitArray<>& Cells, const FLinearColor& Color)
    {
        if (Point.X < Left || Point.X >= Left + Mid || Point.Y < MapTop || Point.Y >= MapBottom) return;
        const int32 Cell = int32(Point.Y / CellPixels) * Columns + int32((Point.X - Left) / CellPixels);
        if (!Host.IsFourDemo() && Cells[Cell]) return;
        Cells[Cell] = true;
        const float Radius = Host.IsFourDemo() ? 4 * UI : 1.65f * UI;
        DrawRect(Ink, Point.X - Radius - 1, Point.Y - Radius - 1, 2 * Radius + 2, 2 * Radius + 2);
        DrawRect(Color, Point.X - Radius, Point.Y - Radius, 2 * Radius, 2 * Radius);
    };
    FSceneViewProjectionData Projection;
    auto* Player = GetWorld()->GetFirstPlayerController();
    auto* Local = Player ? Player->GetLocalPlayer() : nullptr;
    const bool bProject3D = Local && Local->ViewportClient && Local->ViewportClient->Viewport &&
        Local->GetProjectionData(Local->ViewportClient->Viewport, Projection);
    const auto Matrix = bProject3D ? Projection.ComputeViewProjectionMatrix() : FMatrix::Identity;
    TArray<FBox2D, TInlineAllocator<4>> Labels;
    for (int32 I = 0; I < Day.GetStudentCount(); ++I)
    {
        if (!Day.IsStudentVisible(I)) continue;
        const auto Position = Day.GetStudentPosition(I);
        const auto Color = Day.DestinationColor(Day.GetStopColorKey(Day.GetTargetStop(I)));
        const auto Point = Project(Position);
        DrawDot(Point, 0.f, MapMarkers, Color);
        FVector2D Screen;
        if (bProject3D && FSceneView::ProjectWorldToScreen(FVector(Position.X * 100, Position.Y * 100, 90),
            Projection.GetConstrainedViewRect(), Matrix, Screen))
            DrawDot(Screen, Mid, PerspectiveMarkers, Color);
        if (Host.IsFourDemo())
        {
            const FString Label = FString::Printf(TEXT("%d  %s"), I + 1, Day.GetDecision(I));
            float TextWidth, TextHeight;
            const float TextScale = UI * .65f;
            GetTextSize(Label, TextWidth, TextHeight, Font, TextScale);
            const float X = FMath::Clamp(float(Point.X - TextWidth * .5), 9.f, Mid - TextWidth - 17.f);
            float Y = FMath::Clamp(float(Point.Y - 24.f * UI), MapTop, MapBottom - TextHeight);
            FBox2D Box(FVector2D(X - 4, Y), FVector2D(X + TextWidth + 4, Y + TextHeight + 3));
            for (const auto& Placed : Labels)
                if (Box.Intersect(Placed))
                {
                    Y = FMath::Max(MapTop, float(Placed.Min.Y - TextHeight - 5));
                    Box = FBox2D(FVector2D(X - 4, Y), FVector2D(X + TextWidth + 4, Y + TextHeight + 3));
                }
            Labels.Add(Box);
            DrawRect(Ink, Box.Min.X, Box.Min.Y, Box.GetSize().X, Box.GetSize().Y);
            DrawText(Label, Color, X, Y, Font, TextScale);
        }
    }

    const auto& Display = Host.GetDisplayStats();
    const bool bAsync = Host.GetMode() == ECrowdInferenceMode::Background;
    const TCHAR* Mode = bAsync ? TEXT("ScholaMass") : TEXT("Synchronous");
    const FString Summary = Day.IsFirstClassOnly()
        ? FString::Printf(TEXT("%s · First class: %d / %d arrived"), Mode, Day.LegsCompleted, Day.GetStudentCount())
        : FString::Printf(TEXT("%s · %d students · %d legs complete"), Mode, Day.GetStudentCount(), Day.LegsCompleted);
    DrawText(Summary, White, 16 * UI, Height - Footer + 7 * UI, Font, UI * .78f);
    DrawText(Host.IsFourDemo() ? TEXT("Lines show shortest routes · Paths © OpenStreetMap") :
        TEXT("Colors follow destination buildings · Paths © OpenStreetMap"), Muted, 16 * UI, Height - 19 * UI, Font, UI * .59f);
    DrawText(FString::Printf(TEXT("%.1f FPS · %.0f turns/s · %.1f rows/call · %.3f ms/frame"), Display.FramesPerSecond,
        Display.DecisionsPerSecond,
        Display.RowsPerModelCall, Display.InferenceMs), White, Mid + 15 * UI, Height - Footer + 7 * UI, Font, UI * .78f);
    DrawText(TEXT("Tab inference mode · Space pause"), Muted, Mid + 15 * UI, Height - 19 * UI, Font, UI * .59f);
}

void ACrowdHUD::DrawNavigationMap(const FCampusMigrationDay& Day, int32 StudentCount)
{
    const auto& Grid = Day.GetNavigationGrid();
    if (!NavigationMapTexture)
    {
        const FColor Walkable(34, 54, 60);
        const FColor Blocked(86, 108, 121);
        TArray64<FColor> Pixels;
        Pixels.SetNumUninitialized(int64(Grid.Width) * Grid.Height);
        for (int32 Y = 0; Y < Grid.Height; ++Y)
            for (int32 X = 0; X < Grid.Width; ++X)
                Pixels[int64(Y) * Grid.Width + X] = Grid.Blocked(X, Y) ? Blocked : Walkable;
        NavigationMapTexture = UTexture2D::CreateTransient(Grid.Width, Grid.Height, PF_B8G8R8A8, NAME_None,
            TConstArrayView64<uint8>(reinterpret_cast<const uint8*>(Pixels.GetData()), Pixels.Num() * sizeof(FColor)));
        if (NavigationMapTexture) NavigationMapTexture->UpdateResource();
    }

    DrawRect(FLinearColor(FColor(13, 24, 32)), 0, 0, Canvas->ClipX, Canvas->ClipY);
    const auto A = Canvas->Project(FVector(100 * Grid.Origin.X, 100 * Grid.Origin.Y, 100));
    const auto B = Canvas->Project(FVector(100 * (Grid.Origin.X + Grid.Width * Grid.Cell),
        100 * (Grid.Origin.Y + Grid.Height * Grid.Cell), 100));
    if (NavigationMapTexture)
        DrawTexture(NavigationMapTexture, A.X, A.Y, B.X - A.X, B.Y - A.Y, 0, 0, 1, 1,
            FLinearColor::White, BLEND_Opaque);

    // The map only draws one marker per screen cell, even for overlapping walkers.
    constexpr int32 MarkerSize = 4;
    const int32 Columns = FMath::DivideAndRoundUp(FMath::CeilToInt(Canvas->ClipX), MarkerSize);
    const int32 Rows = FMath::DivideAndRoundUp(FMath::CeilToInt(Canvas->ClipY), MarkerSize);
    if (Columns <= 0 || Rows <= 0) return;
    MapMarkers.Init(false, Columns * Rows);
    const double ScaleX = (B.X - A.X) / (Grid.Width * Grid.Cell);
    const double ScaleY = (B.Y - A.Y) / (Grid.Height * Grid.Cell);
    const FLinearColor StudentColor(FColor(128, 229, 200));
    for (int32 Student = 0; Student < StudentCount; ++Student)
    {
        if (!Day.IsWalking(Student)) continue;
        const auto Position = Day.GetNavigationPosition(Student);
        const double X = A.X + (Position.X - Grid.Origin.X) * ScaleX;
        const double Y = A.Y + (Position.Y - Grid.Origin.Y) * ScaleY;
        if (X < 0 || X >= Canvas->ClipX || Y < 0 || Y >= Canvas->ClipY) continue;
        const int32 Cell = int32(Y / MarkerSize) * Columns + int32(X / MarkerSize);
        if (MapMarkers[Cell]) continue;
        MapMarkers[Cell] = true;
        DrawRect(StudentColor, X - 2, Y - 2, MarkerSize, MarkerSize);
    }
}

void ACrowdHUD::DrawHUD()
{
    if (FScreenshotRequest::IsScreenshotRequested() && FParse::Param(FCommandLine::Get(), TEXT("CleanScreenshot"))) return;
    Super::DrawHUD();
    const auto* GameMode = GetWorld()->GetAuthGameMode<ACrowdGameMode>();
    const ACrowdComparisonHost* Host = GameMode ? GameMode->Host.Get() : nullptr;
    if (!Host || !Canvas || Host->IsSplitDemo()) return;

    UFont* Font = GEngine->GetLargeFont();
    const float Scale = FMath::Max(Canvas->ClipY / 1080.f, .5f);
    const float Line = 30 * Scale, X = 20 * Scale;
    float Y = 16 * Scale;
    if (const auto* Day = Host->GetMigrationDay(); Day && Host->GetError().IsEmpty())
    {
        const auto* Turns = Day->GetTurns();
        const bool bBackground = Host->GetMode() == ECrowdInferenceMode::Background;
        const TCHAR* Policy = Turns ? (bBackground ? TEXT("ScholaMass background") : TEXT("Synchronous")) :
            !Day->UsesModel() ? TEXT("Rules") : bBackground ? TEXT("Learned / background") : TEXT("Learned / synchronous");
        const bool bShowInference = Day->UsesModel() && !Day->IsFinished();
        const bool bNavigation = Day->IsNavigation();
        const bool bLivePlanner = Day->UsesLivePlanner();
        if (bNavigation && Host->IsNavigationMapView()) DrawNavigationMap(*Day, Host->GetAgentCount());
        if (bNavigation && Host->ShowsRoutes())
        {
            const FLinearColor Colors[] = {{.33f, .95f, .82f}, {1.f, .72f, .25f},
                {.50f, .75f, 1.f}, {.92f, .55f, 1.f}};
            int32 Shown = 0;
            const int32 Limit = FMath::Min(Day->GetActive(), int32(UE_ARRAY_COUNT(Colors)));
            const auto Project = [this](CampusNav::Vec Point)
            {
                return Canvas->Project(FVector(100 * Point.X, 100 * Point.Y, 100), false);
            };
            for (int32 Student = 0; Student < Host->GetAgentCount() && Shown < Limit; ++Student)
            {
                const auto Route = Day->GetNavigationRoute(Student);
                if (Route.IsEmpty()) continue;
                const FLinearColor Color = Colors[Shown++];
                auto Previous = Project(Route[0]);
                for (int32 I = 1; I < Route.Num(); ++I)
                {
                    const auto Current = Project(Route[I]);
                    if (Previous.Z > 0 && Current.Z > 0)
                        DrawLine(Previous.X, Previous.Y, Current.X, Current.Y, Color.CopyWithNewOpacity(.65f), 2.f);
                    Previous = Current;
                }
                const auto Walker = Project(Day->GetNavigationPosition(Student));
                if (Walker.Z > 0) DrawRect(Color, Walker.X - 3, Walker.Y - 3, 6, 6);
            }
        }
        DrawRect(FLinearColor(0, 0, 0, .65f), X - 12 * Scale, Y - 8 * Scale, 535 * Scale,
            Line * (Turns ? 5.6f : bLivePlanner ? 5.6f : bShowInference ? 4.7f : bNavigation ? 4.6f : 3.6f));
        DrawText(Day->IsFinished() ? FString(TEXT("Campus day complete"))
            : Turns ? FString(TEXT("Learned junction turns"))
            : bNavigation ? FString(TEXT("Campus navigation"))
            : FString::Printf(TEXT("Campus migration %d / %d"), Day->GetLeg() + 1, CampusMigration::LegCount),
            FLinearColor(.25f, 1.f, .45f), X, Y, Font, Scale * 1.6f);
        const FString Progress = bNavigation
            ? FString::Printf(TEXT("%d walking   %d finished   %d timed out"), Day->GetActive(), Day->GetArrived(), Day->GetTimedOut())
            : FString::Printf(TEXT("%d walking   %d arrived   %s"), Day->GetActive(), Day->GetArrived(), Policy);
        DrawText(Progress,
            FLinearColor::White, X, Y + Line * 1.2f, Font, Scale * 1.2f);
        if (bShowInference)
        {
            const auto& Inference = Host->GetDisplayStats();
            DrawText(FString::Printf(TEXT("%.0f decisions/s   %.2f ms/frame on game thread   %.1f rows/call"),
                Inference.DecisionsPerSecond, Inference.InferenceMs, Inference.RowsPerModelCall),
                FLinearColor::White, X, Y + Line * 2.15f, Font, Scale);
            DrawText(TEXT("Tab switches inference mode"), FLinearColor(.75f, .78f, .8f),
                X, Y + Line * 3.f, Font, Scale);
        }
        if (bNavigation)
            DrawText(FString::Printf(TEXT("%.1f FPS   ·   N routes   ·   Space pause"), Host->GetDisplayStats().FramesPerSecond),
                FLinearColor::White, X, Y + Line * 2.15f, Font, Scale);
        if (bLivePlanner)
        {
            const auto Plans = Day->GetPlanningStats();
            const TCHAR* Mode = Host->GetMode() == ECrowdInferenceMode::Background ? TEXT("Schola background") : TEXT("Synchronous");
            DrawText(FString::Printf(TEXT("%s   %.2f ms/frame planning   %.1f rows/call"), Mode,
                Host->GetDisplayStats().InferenceMs, Plans.ModelCalls > 0 ? double(Plans.ModelRows) / Plans.ModelCalls : 0.),
                FLinearColor::White, X, Y + Line * 3.15f, Font, Scale);
        }
        if (Turns)
            DrawText(FString::Printf(TEXT("%lld learned turns   %lld retraced edges   %lld retries"),
                Turns->Stats.Applied, Turns->Stats.RetracedEdges, Turns->Stats.Retries),
                FLinearColor::White, X, Y + Line * 3.75f, Font, Scale);
        DrawText(TEXT("Paths © OpenStreetMap contributors"), FLinearColor(.75f, .78f, .8f),
            X, Y + Line * (Turns ? 4.45f : bLivePlanner ? 4.15f : bShowInference ? 3.7f : bNavigation ? 3.15f : 2.15f), Font, Scale);
        return;
    }
    const bool bBackground = Host->GetMode() == ECrowdInferenceMode::Background;
    const FLinearColor Accent = bBackground ? FLinearColor(.25f, 1.f, .45f) : FLinearColor(1.f, .55f, .2f);
    const auto& Stats = Host->GetDisplayStats();

    const FLinearColor Normal = FLinearColor::White, Alert(1.f, .3f, .25f);
    const bool bShort = !Stats.Shortfall.IsEmpty();
    TArray<TTuple<FString, FString, FLinearColor>> Rows;
    Rows.Emplace(TEXT("Mass agents"), FText::AsNumber(Host->GetAgentCount()).ToString(), Normal);
    Rows.Emplace(TEXT("Frame rate"), FString::Printf(TEXT("%.1f FPS"), Stats.FramesPerSecond), Normal);
    Rows.Emplace(TEXT("Frame time"), FString::Printf(TEXT("%.1f ms  (p99 %.1f ms)"), Stats.FrameMs, Stats.FrameP99Ms), Normal);
    Rows.Emplace(TEXT("Inference on game thread"), FString::Printf(TEXT("%.2f ms / frame"), Stats.InferenceMs), Normal);
    Rows.Emplace(TEXT("Crowd movement"), FString::Printf(TEXT("%.2f ms / frame"), Stats.MoveMs), Normal);
    Rows.Emplace(TEXT("Decisions"), FString::Printf(TEXT("%s / s  (%.0f%% of %s)"),
        *FText::AsNumber(FMath::RoundToInt(Stats.DecisionsPerSecond)).ToString(), Stats.TargetFraction * 100,
        *FText::AsNumber(FMath::RoundToInt(Host->GetTargetDecisionsPerSecond())).ToString()), bShort ? Alert : Normal);
    Rows.Emplace(TEXT("Agents per model call"), FString::Printf(TEXT("%.1f"), Stats.RowsPerModelCall), Normal);
    Rows.Emplace(bBackground ? TEXT("Model time on workers") : TEXT("Model time on game thread"),
        FString::Printf(TEXT("%.0f ms / s"), Stats.ModelMsPerSecond), Normal);
    Rows.Emplace(TEXT("Observation to action"), FString::Printf(TEXT("%.1f ms"), Stats.LatencyMs), Normal);
    if (bBackground)
    {
        const auto& L = Stats.LossesPerSecond;
        const bool bLosing = L.Deferred + L.Rejected + L.Invalid + L.Lost() > 0;
        Rows.Emplace(TEXT("Dropped requests"), FText::AsNumber(Stats.DroppedRows).ToString(), Stats.DroppedRows > 0 ? Alert : Normal);
        if (bLosing)
            Rows.Emplace(TEXT("Unserved rows / s"), FString::Printf(TEXT("%s deferred, %s stale, %s failed"),
                *FText::AsNumber(L.Deferred).ToString(), *FText::AsNumber(L.Stale).ToString(),
                *FText::AsNumber(L.Expired + L.Failed).ToString()), Alert);
    }
    if (bShort) Rows.Emplace(TEXT("Below target"), Stats.Shortfall, Alert);

    FString Phase;
    const double Progress = Host->GetCaptureProgress(Phase);
    const FString Footer = !Host->GetError().IsEmpty() ? TEXT("Error: ") + Host->GetError()
        : Progress >= 0 ? FString::Printf(TEXT("%s %.0f%%"), *Phase, Progress * 100) : FString();
    float Width = 590 * Scale;
    for (const auto& Row : Rows)
    {
        float TextWidth = 0, TextHeight = 0;
        GetTextSize(Row.Get<1>(), TextWidth, TextHeight, Font, Scale * 1.25f);
        Width = FMath::Max(Width, (260 + 32) * Scale + TextWidth);
    }
    const float Height = Line * (Rows.Num() + (Footer.IsEmpty() ? 2 : 3)) + 16 * Scale;
    DrawRect(FLinearColor(0, 0, 0, .65f), X - 12 * Scale, Y - 8 * Scale, Width, Height);
    DrawText(bBackground ? TEXT("CAMPUS / background inference") : TEXT("CAMPUS / synchronous inference"),
        Accent, X, Y, Font, Scale * 1.6f);
    Y += Line * 1.5f;
    for (const auto& Row : Rows)
    {
        DrawText(Row.Get<0>(), FLinearColor(.75f, .75f, .75f), X, Y, Font, Scale * 1.25f);
        DrawText(Row.Get<1>(), Row.Get<2>(), X + 260 * Scale, Y, Font, Scale * 1.25f);
        Y += Line;
    }
    if (!Footer.IsEmpty())
        DrawText(Footer, Host->GetError().IsEmpty() ? FLinearColor(.6f, .6f, .6f) : FLinearColor::Red, X, Y + 4 * Scale, Font, Scale * 1.1f);
    const float Bottom = Canvas->ClipY - 64 * Scale;
    DrawRect(FLinearColor(.015f,.025f,.04f,.85f), 0, Bottom, Canvas->ClipX, 64 * Scale);
    DrawText(TEXT("50H NORTH / UNIVERSITY CAMPUS"), FLinearColor(.75f,.85f,.83f), X, Bottom+6*Scale, Font, Scale);
    DrawText(TEXT("1/2/3 cameras   WASD + Q/E fly   RMB look   Shift fast   Space pause   N routes"),
        FLinearColor(.7f,.73f,.76f), X, Bottom+32*Scale, Font, Scale*.9f);

}
