using System.Threading.Tasks;
using Godot;
using Schola.Godot;

public partial class PlayerController : CharacterBody3D
{
	[ExportCategory("Movement")]
	[Export] public float Speed { get; set; } = 5.0f;

	[ExportCategory("Goal")]
	[Export] public NodePath GoalPath { get; set; }
	[Export] public float GoalRadius { get; set; } = 1.5f;
	[Export] public float GoalHoldDelay { get; set; } = 1.0f;

	[ExportCategory("Episode")]
	[Export] public float EpisodeResetDelay { get; set; } = 2.0f;

	private Node3D _goal;
	private ScholaAgentNode _agent;
	private Vector3 _spawnPosition;
	private float _goalOverlapTime;
	private bool _resetScheduled;

	public override void _Ready()
	{
		_spawnPosition = GlobalPosition;
		_goal = GetNodeOrNull<Node3D>(GoalPath);
		_agent = GetNodeOrNull<ScholaAgentNode>("ScholaAgentNode");

		if (_goal == null)
			GD.PushWarning("PlayerController needs a GoalPath.");

		if (_agent == null)
			GD.PushWarning("Player needs a ScholaAgentNode child.");
	}

	public override void _PhysicsProcess(double delta)
	{
		if (_resetScheduled)
		{
			Velocity = Vector3.Zero;
			return;
		}

		Vector2 input = Input.GetVector(
			"move_left",
			"move_right",
			"move_forward",
			"move_back");

		Vector3 direction = new Vector3(input.X, 0.0f, input.Y).Normalized();

		Velocity = new Vector3(
			direction.X * Speed,
			Velocity.Y,
			direction.Z * Speed);

		MoveAndSlide();

		CheckGoal((float)delta);
	}

	private void CheckGoal(float delta)
	{
		if (_goal == null || _agent?.Environment == null)
			return;

		float distanceToGoal = GlobalPosition.DistanceTo(_goal.GlobalPosition);

		if (distanceToGoal > GoalRadius)
		{
			_goalOverlapTime = 0.0f;
			return;
		}

		_goalOverlapTime += delta;

		if (_goalOverlapTime < GoalHoldDelay)
			return;

		_agent.CompleteEpisode();
		_resetScheduled = true;
		_ = ResetAfterDelay();
	}

	private async Task ResetAfterDelay()
	{
		await ToSignal(
			GetTree().CreateTimer(EpisodeResetDelay),
			SceneTreeTimer.SignalName.Timeout);

		_agent?.Environment?.ResetEpisode();

		GlobalPosition = _spawnPosition;
		Velocity = Vector3.Zero;
		_goalOverlapTime = 0.0f;
		_resetScheduled = false;
	}
}
