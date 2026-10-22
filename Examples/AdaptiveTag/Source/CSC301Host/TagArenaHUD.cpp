// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "TagArena.h"
#include "CanvasItem.h"
#include "Engine/Canvas.h"
#include "Engine/Engine.h"
#include "GameFramework/PlayerController.h"

void ATagArenaHUD::DrawHUD()
{
    Super::DrawHUD();
    auto* Arena=GetWorld()?GetWorld()->GetAuthGameMode<ATagArenaGameMode>():nullptr;
    if(!Canvas || !Arena || !PlayerOwner)return;
    const float S=FMath::Min(Canvas->SizeX/1600.0f,Canvas->SizeY/1000.0f);
    const float W=Canvas->SizeX/S, H=Canvas->SizeY/S;
    const FLinearColor Ink(0.86f,0.92f,0.98f), Muted(0.48f,0.59f,0.70f), Cyan(0.1f,0.85f,1);
    const FLinearColor Coral(1,0.32f,0.38f), Mint(0.25f,0.95f,0.65f), Gold(1,0.76f,0.3f), Panel(0.012f,0.022f,0.037f,0.97f);
    auto Rect=[&](float X,float Y,float Width,float Height,FLinearColor Color){DrawRect(Color,X*S,Y*S,Width*S,Height*S);};
    auto Text=[&](const FString& Value,float X,float Y,FLinearColor Color,float Size=0.75f){
        FCanvasTextItem Item(FVector2D(X*S,Y*S),FText::FromString(Value),GEngine->GetMediumFont(),Color);
        Item.Scale=FVector2D(S*Size*1.3f); Item.EnableShadow(FLinearColor(0,0,0,0.6f)); Canvas->DrawItem(Item);
    };
    auto Line=[&](float X,float Y,float X2,float Y2,FLinearColor Color,float Thickness=1){DrawLine(X*S,Y*S,X2*S,Y2*S,Color,Thickness*S);};
    auto WorldText=[&](const FString& Value,const FVector& Position,FLinearColor Color,float Offset=0){
        FVector2D Screen;
        if(PlayerOwner->ProjectWorldLocationToScreen(Position,Screen)) {
            float Width,Height; GetTextSize(Value,Width,Height,GEngine->GetMediumFont(),S*0.75f*1.3f);
            Text(Value,(Screen.X-Width/2)/S,Screen.Y/S+Offset,Color);
        }
    };
    const auto& Stats=Arena->LearningStats;
    const auto& Decision=Arena->CurrentDecision;
    Rect(24,18,W-48,76,Panel); Rect(24,18,4,76,Cyan);
    Text(TEXT("ADAPTIVE TAG"),44,28,Ink,1.6f);
    Text(TEXT("UNREAL + RLTOOLS / NATIVE ONLINE LEARNING"),46,67,Muted,0.66f);
    Text(TEXT("LOCAL CPU / NO PYTHON BACKEND"),W-395,34,Cyan,0.72f);
    Text(FString::Printf(TEXT("%.0f FPS | SIM %.0fx | %s"),Arena->DisplayFrameMs>0?1000/Arena->DisplayFrameMs:0,Arena->SimulationSpeed,Arena->bPaused?TEXT("PAUSED"):TEXT("RUNNING")),W-395,63,Gold,0.67f);
    Rect(24,112,330,680,Panel);
    Text(TEXT("TAGGER / LEARNING"),42,126,Coral,0.85f);
    Text(Arena->LearningStatus,42,158,Stats.healthy?Mint:Coral,0.64f);
    const float P=Arena->bUseLearner?Stats.expectedLeft:0.5f;
    Text(FString::Printf(TEXT("LEFT %.0f%%               RIGHT %.0f%%"),100*P,100*(1-P)),42,192,Ink,0.72f);
    Rect(42,220,292,10,Coral); Rect(42,220,292*P,10,Cyan);
    Text(TEXT("Actor's own LEFT / RIGHT distribution"),42,240,Muted,0.59f);
    Text(FString::Printf(TEXT("EXPERIENCES %u   /   ONLINE UPDATES"),Stats.experiences),42,274,Ink,0.68f);
    Text(FString::Printf(TEXT("TRAINING UPDATES  %u"),Stats.updates),42,304,Ink,0.72f);
    Text(FString::Printf(TEXT("POLICY v%u   TRAINED v%u"),Stats.appliedVersion,Stats.trainedVersion),42,334,Muted,0.63f);
    Text(FString::Printf(TEXT("QUEUE %u/128  DROPPED %u  INVALID %u"),Stats.queued,Stats.dropped,Stats.rejected),42,362,Muted,0.59f);
    Text(TEXT("P(LEFT) / LAST 30 WALL-CLOCK SECONDS"),42,394,Muted,0.60f);
    Rect(42,420,292,66,FLinearColor(0.025f,0.044f,0.065f)); Line(42,453,334,453,Muted);
    const auto& History=Arena->ProbabilityHistory;
    for(int32 i=1;i<History.Num();++i)Line(42+292.0f*(i-1)/119,486-66*History[i-1],42+292.0f*i/119,486-66*History[i],Cyan,2);
    Text(FString::Printf(TEXT("Value %+.2f   Advantage %+.2f"),Stats.value,Stats.advantage),42,502,Muted,0.66f);
    Text(FString::Printf(TEXT("Critic MSE %.3f   Entropy %.3f"),Stats.criticMse,Stats.entropy),42,530,Ink,0.63f);
    Text(FString::Printf(TEXT("Train %.3f ms   Inference %.3f ms"),Stats.batchMs,Stats.inferenceMs),42,558,Muted,0.63f);
    Line(42,590,334,590,Muted);
    Text(TEXT("WHAT THE AGENT SEES"),42,604,Cyan,0.73f);
    Text(TEXT("Observation [1.0] / constant bias only"),42,636,Ink,0.64f);
    Text(TEXT("No runner position, route or mode input."),42,662,Muted,0.59f);
    Text(FString::Printf(TEXT("Actor P(LEFT)=%.0f%%  P(RIGHT)=%.0f%%"),100*Decision.leftProbability,100*(1-Decision.leftProbability)),42,693,Ink,0.66f);
    Text(!Arena->bUseLearner?TEXT("Comparison: fixed 50/50 baseline"):TEXT("Action source: actor only / every round"),42,722,Gold,0.60f);
    Text(Arena->bTaggerRevealed?FString::Printf(TEXT("Sampled route: %s"),*ATagArenaGameMode::SideLabel(Arena->TaggerSide)):TEXT("Sampled route: HIDDEN until you commit"),42,751,Coral,0.61f);
    const float R=W-354;
    Rect(R,112,330,320,Panel);
    Text(FString::Printf(TEXT("ROUND %02d"),Arena->RoundNumber),R+18,127,Ink,1.15f);
    Text(Arena->GetModeLabel(),R+18,169,Cyan,0.78f);
    Text(FString::Printf(TEXT("%.1f / 12.0 game seconds"),Arena->RoundSeconds),R+18,201,Muted,0.68f);
    Text(FString::Printf(TEXT("CATCHES %d     ESCAPES %d"),Arena->Catches,Arena->Escapes),R+18,242,Ink,0.75f);
    Text(Arena->GetRecentCount()?FString::Printf(TEXT("RECENT CATCH RATE %.0f%% / %d rounds"),100*Arena->GetRecentCatchRate(),Arena->GetRecentCount()):TEXT("RECENT CATCH RATE --"),R+18,280,Mint,0.65f);
    Text(TEXT("Fixed 50/50 reference: 50% in expectation"),R+18,308,Muted,0.58f);
    Text(FString::Printf(TEXT("RUNNER ROUTES  LEFT %d  RIGHT %d"),Arena->LeftRoutes,Arena->RightRoutes),R+18,347,Ink,0.64f);
    Text(FString::Printf(TEXT("Timeouts %d   Skipped samples %d"),Arena->Timeouts,Arena->SkippedExperiences),R+18,382,Muted,0.64f);
    Rect(R,450,330,342,Panel);
    Text(TEXT("EXPERIMENT CONTROLS"),R+18,464,Muted,0.73f);
    const TCHAR* Controls[]={TEXT("1  Human / WASD runner"),TEXT("2  Auto / 80% left"),TEXT("3  Auto / 80% right"),TEXT("4  Auto / balanced"),TEXT("F  Speed 1x / 4x / 12x"),TEXT("L  Freeze / resume training"),TEXT("B  Learned / fixed 50:50 policy")};
    for(int32 i=0;i<7;++i)Text(Controls[i],R+18,500+33*i,i==4?Gold:Ink,0.70f);
    Text(TEXT("Mode changes preserve learned weights."),R+18,750,Muted,0.58f);
    WorldText(TEXT("ESCAPE"),FVector(575,0,10),Mint); WorldText(TEXT("LEFT"),FVector(0,-500,10),Cyan);
    WorldText(TEXT("RIGHT"),FVector(0,500,10),Cyan); WorldText(TEXT("COVER"),FVector(20,0,130),Ink);
    if(Arena->GetRunner())WorldText(Arena->RunnerMode==ETagRunnerMode::Human?TEXT("YOU / WASD"):TEXT("RUNNER"),Arena->GetRunner()->GetActorLocation(),Cyan,30);
    if(Arena->GetTagger())WorldText(Arena->bTaggerRevealed?TEXT("TAGGER"):TEXT("CHOICE HIDDEN"),Arena->GetTagger()->GetActorLocation(),Coral,30);
    if(Arena->Phase!=ETagRoundPhase::Active || Arena->bPaused) {
        const float X=W/2-205,Y=H/2+84;
        Rect(X,Y,410,92,Panel);
        Text(Arena->bPaused?TEXT("PAUSED"):Arena->Phase==ETagRoundPhase::Countdown?TEXT("GET READY"):Arena->ResultText,X+20,Y+12,Gold,1.4f);
        Text(Arena->bPaused?TEXT("Space to continue"):TEXT("Tagger commits before the runner chooses."),X+20,Y+58,Muted,0.65f);
    }
    Rect(24,H-192,W-48,118,Panel);
    Text(TEXT("LIVE EXPERIENCE / TRAINING EVENTS"),42,H-182,Cyan,0.67f);
    Text(Arena->LastExperienceText,520,H-182,Ink,0.66f);
    for(int32 i=0;i<FMath::Min(3,Arena->LiveEvents.Num());++i)Text(Arena->LiveEvents[i],42,H-151+24*i,i==0?Ink:Muted,0.64f);
    Rect(24,H-60,W-48,40,Panel);
    Text(TEXT("WASD MOVE   R RESTART ROUND   SPACE PAUSE   T RESET MODEL + SCORES   |   +1 CATCH / -1 ESCAPE   |   MEMORY ONLY"),42,H-49,Ink,0.65f);
}
