using Godot;
using Schola.Godot;

public partial class DemoController : Node
{
	[Export] public NodePath EnvironmentPath { get; set; }
	private ScholaEnvironmentNode _environment;

	public override void _Ready()
	{
		_environment = GetNodeOrNull<ScholaEnvironmentNode>(EnvironmentPath);
	}

	public override void _UnhandledInput(InputEvent @event)
	{
		if (_environment == null)
			return;

		if (@event.IsActionPressed("ui_accept"))
			_environment.AddReward(1.0f, "Test reward");

		if (@event.IsActionPressed("ui_cancel"))
			_environment.FailEpisode("Test failure");
	}
}
