using Godot;
using System;

namespace Schola.Godot;

[GlobalClass]
public partial class ScholaEnvironmentNode : Node
{

	[ExportCategory("Rewards")]
	[Export] public float ProgressReward { get; set; } = 1.0f;
	[Export] public float StepReward { get; set; } = -0.001f;
	[Export] public float GoalReward { get; set; } = 10.0f;
	[Export] public float FailurePenalty { get; set; } = -10.0f;

	[ExportCategory("Episodes")]
	// Zero disables the limit.
	[Export(PropertyHint.Range, "0,100000,1,or_greater")]
	public int EpisodeLimit { get; set; } = 0;

	public float TotalReward { get; private set; }
	public bool EpisodeFinished { get; private set; }
	public int CompletedEpisodeCount { get; private set; }
	public bool EpisodeLimitReached => EpisodeLimit > 0 && CompletedEpisodeCount >= EpisodeLimit;

	// Notifies listeners when total reward changes
	[Signal]
	public delegate void RewardChangedEventHandler(float totalReward, string reason);
	// Notifies listeners when an episode ends
	[Signal]
	public delegate void EpisodeEndedEventHandler(bool succeeded);
	// Notifies listeners when no further episodes may be started.
	[Signal]
	public delegate void EpisodesExhaustedEventHandler();


	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		AddToGroup("schola_environment");
	}


	// Adds a reward amount to the current episode total
	public void AddReward(float amount, string reason = "")
	{
		if (EpisodeFinished)
			return;

		TotalReward += amount;
		EmitSignal(SignalName.RewardChanged, TotalReward, reason);
	}

	// Converts progress into a scaled reward
	public void AddProgressReward(float progressAmount)
	{
		AddReward(progressAmount * ProgressReward, "Progress");
	}

	// Finish episode successfully -> grant the goal reward
	public void CompleteEpisode()
	{
		if (EpisodeFinished)
			return;

		AddReward(GoalReward, "Goal reached");
		FinishEpisode(true);
	}

	// Finish episode unsuccessfully -> apply the failure penalty
	public void FailEpisode(string reason = "Failed")
	{
		if (EpisodeFinished)
			return;

		AddReward(FailurePenalty, reason);
		FinishEpisode(false);
	}

	// Clears the current reward total and starts a new episode
	public bool ResetEpisode()
	{
		if (EpisodeLimitReached)
			return false;

		TotalReward = 0.0f;
		EpisodeFinished = false;
		EmitSignal(SignalName.RewardChanged, TotalReward, "Episode reset");
		return true;
	}

	// Clears the completed-episode counter so a new limited run can begin.
	public void ResetEpisodeLimit()
	{
		CompletedEpisodeCount = 0;
	}

	private void FinishEpisode(bool succeeded)
	{
		EpisodeFinished = true;
		CompletedEpisodeCount++;
		EmitSignal(SignalName.EpisodeEnded, succeeded);

		if (EpisodeLimitReached)
			EmitSignal(SignalName.EpisodesExhausted);
	}
}
