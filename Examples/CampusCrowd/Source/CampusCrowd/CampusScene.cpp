#include "CampusScene.h"
#include "CampusTurnDay.h"
#include "ProceduralMeshComponent.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Materials/MaterialInterface.h"

ACampusScene::ACampusScene()
{
    Mesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("CAD campus geometry"));
    Mesh->SetFlags(RF_Transient);
    RootComponent = Mesh;
    Mesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
}
void ACampusScene::PostRegisterAllComponents()
{
    Super::PostRegisterAllComponents();
    if (!IsTemplate() && Mesh)
    {
        SetRootComponent(Mesh);
        if (Nodes.IsEmpty()) LoadCampus();
    }
}
void ACampusScene::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);
    if (Nodes.IsEmpty()) LoadCampus();
}
void ACampusScene::BeginPlay()
{
    Super::BeginPlay();
    if (Nodes.IsEmpty()) LoadCampus();
}
bool ACampusScene::LoadCampus()
{
    Nodes.Empty();
    FString Text;
    TSharedPtr<FJsonObject> Root;
    if (!FFileHelper::LoadFileToString(Text, *(FPaths::ProjectContentDir() / TEXT("Campus/campus.json"))) ||
        !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root) || !Root.IsValid())
    {
        LoadError = TEXT("Missing or invalid Content/Campus/campus.json. Run Tools/import_campus.py.");
        UE_LOG(LogTemp, Error, TEXT("%s"), *LoadError);
        return false;
    }
    LoadError.Empty();
    Extent = Root->GetNumberField(TEXT("extent"));
    BuildingCount = Root->GetNumberField(TEXT("building_count"));
    for (auto Value : Root->GetArrayField(TEXT("nodes")))
    {
        auto Node = Value->AsObject();
        const auto& P = Node->GetArrayField(TEXT("position"));
        FCampusNode N;
        N.Position = FVector2f(P[0]->AsNumber(), P[1]->AsNumber());
        for (auto Link : Node->GetArrayField(TEXT("links"))) N.Links.Add(Link->AsNumber());
        Nodes.Add(MoveTemp(N));
    }
    Mesh->ClearAllMeshSections();
    auto* Material = LoadObject<UMaterialInterface>(nullptr, TEXT("/Game/Campus/M_Campus.M_Campus"));
    int32 Section = 0;
    for (auto Value : Root->GetArrayField(TEXT("sections")))
    {
        auto Part = Value->AsObject();
        TArray<FVector> Vertices;
        TArray<int32> Triangles;
        for (auto V : Part->GetArrayField(TEXT("vertices")))
        {
            const auto& P = V->AsArray();
            Vertices.Emplace(P[0]->AsNumber(), P[1]->AsNumber(), P[2]->AsNumber());
        }
        for (auto T : Part->GetArrayField(TEXT("triangles"))) Triangles.Add(T->AsNumber());
        const auto& C = Part->GetArrayField(TEXT("color"));
        TArray<FLinearColor> Colors;
        Colors.Init(FLinearColor(C[0]->AsNumber(), C[1]->AsNumber(), C[2]->AsNumber()), Vertices.Num());
        TArray<FVector> Normals;
        TArray<FVector2D> UV;
        TArray<FProcMeshTangent> Tangents;
        // CAD vertices are split per triangle; calculate flat normals in linear time.
        Normals.SetNumZeroed(Vertices.Num());
        for (int32 T=0; T+2<Triangles.Num(); T+=3)
        {
            const int32 A=Triangles[T], B=Triangles[T+1], Third=Triangles[T+2];
            const FVector N = FVector::CrossProduct(Vertices[Third]-Vertices[A], Vertices[B]-Vertices[A]).GetSafeNormal();
            Normals[A] = Normals[B] = Normals[Third] = N;
        }
        Mesh->CreateMeshSection_LinearColor(Section, Vertices, Triangles, Normals, UV, Colors, Tangents, false);
        if (Material) Mesh->SetMaterial(Section, Material);
        ++Section;
    }
    UE_LOG(LogTemp, Display, TEXT("Campus loaded: %d buildings, %d route nodes"), BuildingCount, Nodes.Num());
    return Nodes.Num() > 1;
}

bool ACampusScene::LoadDestinationHighlights(const TMap<FString, int32>& BuildingKeys, FString& Error)
{
    FString Text;
    TSharedPtr<FJsonObject> Data;
    const TArray<TSharedPtr<FJsonValue>> *Buildings, *Map;
    Error = TEXT("Invalid destination highlights. Run Tools/build_destination_highlights.py with the campus OSM extract.");
    if (!FFileHelper::LoadFileToString(Text, *(FPaths::ProjectContentDir() / TEXT("Campus/destination_highlights.json"))) ||
        !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Data) || !Data ||
        !Data->TryGetArrayField(TEXT("map"), Map) || Map->Num() != 4 ||
        !Data->TryGetArrayField(TEXT("buildings"), Buildings))
        return false;
    MapOrigin = FVector2f((*Map)[0]->AsNumber(), (*Map)[1]->AsNumber());
    MapSize = FVector2f((*Map)[2]->AsNumber(), (*Map)[3]->AsNumber());
    if (MapSize.X <= 0 || MapSize.Y <= 0) return false;

    DestinationMesh = NewObject<UProceduralMeshComponent>(this);
    DestinationMesh->SetupAttachment(Mesh);
    DestinationMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    DestinationMesh->SetCastShadow(false);
    DestinationMesh->RegisterComponent();
    auto* Material = LoadObject<UMaterialInterface>(nullptr, TEXT("/Game/Campus/M_DestinationBuildings.M_DestinationBuildings"));
    if (!Material) return false;
    DestinationFootprints.SetNum(BuildingKeys.Num());
    VisibleDestinations.Init(false, BuildingKeys.Num());
    for (const auto& Entry : *Buildings)
    {
        const auto* Building = Entry->AsObject().Get();
        if (!Building) return false;
        const int32* Index = BuildingKeys.Find(Building->GetStringField(TEXT("id")));
        if (!Index) continue;
        const TArray<TSharedPtr<FJsonValue>> *VertexRows, *IndexRows, *FootprintRows;
        if (!Building->TryGetArrayField(TEXT("vertices"), VertexRows) ||
            !Building->TryGetArrayField(TEXT("indices"), IndexRows) || IndexRows->Num() % 3 != 0 ||
            !Building->TryGetArrayField(TEXT("footprint"), FootprintRows))
            return false;

        TArray<FVector> Input;
        for (const auto& Value : *VertexRows)
        {
            const TArray<TSharedPtr<FJsonValue>>* Row;
            if (!Value->TryGetArray(Row) || Row->Num() != 3) return false;
            Input.Emplace((*Row)[0]->AsNumber(), (*Row)[1]->AsNumber(), (*Row)[2]->AsNumber());
        }
        TArray<int32> Indices;
        for (const auto& Value : *IndexRows)
        {
            int32 Vertex;
            if (!Value->TryGetNumber(Vertex) || !Input.IsValidIndex(Vertex)) return false;
            Indices.Add(Vertex);
        }
        TArray<FVector> Vertices, Normals;
        TArray<int32> Triangles;
        const auto Color = FCampusTurnDay::DestinationColor(*Index);
        for (int32 I = 0; I < Indices.Num(); I += 3)
        {
            const auto A = Input[Indices[I]], B = Input[Indices[I+1]], C = Input[Indices[I+2]];
            const FVector Normal = FVector::CrossProduct(B - A, C - A).GetSafeNormal();
            // Move the colored surface along the fixed camera ray without shifting its projection.
            const FVector Offset = -FCampusTurnDay::OverviewRotation().Vector() * 65.0;
            for (const auto& Point : {A, B, C})
            {
                Triangles.Add(Vertices.Add(Point + Offset));
                Normals.Add(Normal);
            }
        }
        TArray<FLinearColor> Colors;
        Colors.Init(Color, Vertices.Num());
        DestinationMesh->CreateMeshSection_LinearColor(*Index, Vertices, Triangles, Normals, {}, Colors, {}, false);
        DestinationMesh->SetMaterial(*Index, Material);
        DestinationMesh->SetMeshSectionVisible(*Index, false);
        auto& Footprint = DestinationFootprints[*Index];
        for (const auto& Value : *FootprintRows)
        {
            const TArray<TSharedPtr<FJsonValue>>* Row;
            if (!Value->TryGetArray(Row) || Row->Num() != 3) return false;
            for (const auto& Point : *Row)
            {
                const TArray<TSharedPtr<FJsonValue>>* Pair;
                if (!Point->TryGetArray(Pair) || Pair->Num() != 2) return false;
                Footprint.Emplace((*Pair)[0]->AsNumber(), (*Pair)[1]->AsNumber());
            }
        }
    }
    for (const auto& Footprint : DestinationFootprints)
        if (Footprint.IsEmpty()) return false;
    Error.Reset();
    return true;
}

void ACampusScene::SetActiveDestinations(const TBitArray<>& Active)
{
    if (!DestinationMesh) return;
    for (int32 I = 0; I < Active.Num(); ++I)
    {
        if (VisibleDestinations[I] == Active[I]) continue;
        VisibleDestinations[I] = Active[I];
        DestinationMesh->SetMeshSectionVisible(I, Active[I]);
    }
}
