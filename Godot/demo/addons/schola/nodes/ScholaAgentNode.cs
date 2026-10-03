using Godot;
using System;

namespace Schola.Godot;

[GlobalClass]
public partial class ScholaAgentNode : Node
{
	[Export] public NodePath EnvironmentPath { get; set; }

	public ScholaEnvironmentNode Environment { get; private set; }

	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		if (EnvironmentPath.IsEmpty)
		{
			// fallback
			Environment = GetTree().GetFirstNodeInGroup("schola_environment") as ScholaEnvironmentNode;
		} else
		{
			Environment = GetNodeOrNull<ScholaEnvironmentNode>(EnvironmentPath);
		}

		if (Environment == null)
			GD.PushWarning("ScholaAgentNode needs a ScholaEnvironmentNode");
	}


	public void AddReward(float amount, string reason = "")
	{
		Environment.AddReward(amount, reason);
	}

	public void CompleteEpisode()
	{
		Environment.CompleteEpisode();
	}

	public void FailEpisode(string reason = "Failed")
	{
		Environment.FailEpisode(reason);
	}

	public bool ResetEpisode()
	{
		return Environment != null && Environment.ResetEpisode();
	}
}
