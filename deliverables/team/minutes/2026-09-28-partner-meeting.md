# AMD Schola – Partner Meeting 1

**Date:** 2026.9.28 
**Location:** Microsoft Teams  
**Attendees:** Everyone in the team and the partner.

## 1. Agenda

- Discuss the scope and expectations of the AMD Schola project.
- Explore potential directions for extending Schola.
- Clarify D1 requirements.
- Discuss communication and next steps.

## 2. Discussion

### Project Direction

The partner explained that two other teams are working on Unity and Godot ports of Schola. Given our team's RL background, our team was encouraged to focus on developing an extension for Schola instead.

Three potential directions were discussed:

1. **Python-Based Environment Generation**
   - Allow developers to define RL environments through Python, similar to Isaac Lab and MuJoCo.
   - Simplify environment setup and training workflows.

2. **Advanced Inference and NPC Management**
   - Improve inference efficiency through batching, asynchronous processing, and agent scheduling.
   - Explore concurrency, selective inference, and Unreal's MassEntity system.

3. **Agent Benchmarking**
   - Build an application to benchmark AI agents in Unreal Engine environments.
   - Potentially use Unreal's Lyra demo to evaluate models in gameplay scenarios.

Other topics included CARLA and the feasibility of implementing these features within a semester.

These ideas were preliminary, and no specific feature was finalized during the meeting.

### D1 Requirements

- For a developer-facing library, the UI mock-up can focus on developer experience, APIs, and workflows rather than a traditional graphical interface.
- Schola aims to support engine-independent RL workflows, allowing developers to reuse training infrastructure across game engines.

### Communication

- Microsoft Teams was proposed for day-to-day communication.
- Email would be used for formal updates.
- The team should designate a primary point of contact.

## 3. Decisions

- Explore a Schola extension rather than the originally proposed Unity port.
- Discuss the proposed directions internally before committing to a specific feature.
- Arrange a follow-up meeting to refine the project scope.

## 4. Action Items

| Task | Responsible |
|------|-------------|
| Evaluate potential extension directions | Student Team |
| Discuss technical feasibility and scope | Student Team |
| Arrange a follow-up meeting | AMD Partner |
| Notify the professor about the proposed scope change | AMD Partner |
| Designate a primary point of contact | Student Team |

## 5. Next Steps

- Review the proposed project directions internally.
- Prepare questions and feedback for the next partner meeting.
- Work toward defining a feasible project scope for the semester.