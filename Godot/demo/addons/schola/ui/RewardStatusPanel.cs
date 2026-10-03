using Godot;
using System;

namespace Schola.Godot;

[GlobalClass]
public partial class RewardStatusPanel : Control
{
	[Export] public NodePath EnvironmentPath { get; set; }

	private Label _label;
	private ScholaEnvironmentNode _environment;

	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		// Label setup:
		_label = GetNodeOrNull<Label>("Label");		

		// Environment setup:
		if (EnvironmentPath.IsEmpty)
		{
			// fallback
			_environment = GetTree().GetFirstNodeInGroup("schola_environment") as ScholaEnvironmentNode;
		} else
		{
			_environment = GetNodeOrNull<ScholaEnvironmentNode>(EnvironmentPath);
		}

		if (_environment == null)
		{
			_label.Text = "No Schola environment found.";
			return;
		}

		_environment.RewardChanged += OnRewardChanged;
		_environment.EpisodeEnded += OnEpisodeEnded;

		OnRewardChanged(_environment.TotalReward, "Ready");
	}


	// runs when node is removed from the scene tree
	public override void _ExitTree()
	{
		// unsubscribe from environment's signals
		if (_environment != null)
		{
			_environment.RewardChanged -= OnRewardChanged;
			_environment.EpisodeEnded -= OnEpisodeEnded;
		}
	}


	private void OnRewardChanged(float totalReward, string reason)
	{
		_label.Text = $"Reward: {totalReward:F3}\n{reason}\n{GetEpisodeStatus()}";
	}
	
	private void OnEpisodeEnded(bool succeeded)
	{
		if (succeeded == true)
		{
			_label.Text += "\nEpisode complete";
		} else
		{
			_label.Text += "\nEpisode failed";
		}
	}

	private string GetEpisodeStatus()
	{
		var currentEpisode = _environment.CompletedEpisodeCount;
		if (!_environment.EpisodeFinished)
			currentEpisode++;

		if (_environment.EpisodeLimit > 0)
			return $"Episode: {currentEpisode} / {_environment.EpisodeLimit}";

		return $"Episode: {currentEpisode} / inf";
	}
}
