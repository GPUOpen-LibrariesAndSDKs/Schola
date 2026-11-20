# AMD SCHOLA/John SWEs
> _Note:_ This document will evolve throughout your project. You commit regularly to this file while working on the project (especially edits/additions/deletions to the _Highlights_ section). 
 > **This document will serve as a master plan between your team, your partner and your TA.**

## Product Details
 
#### Q1: What is the product?

An extension to AMD Schola, an Unreal Engine library for training and running reinforcement learning (RL) agents, to build smarter and more optimized NPCs in games.

Schola lets developers train RL agents in Unreal Engine and use them in games. Today it assumes each agent runs inference every frame, one at a time. This is an undesirable pattern in many scenarios:
- In turn based games, per-frame inference wastes compute.
- A simulations with thousands of pedestrains can't afford a separate inference call per character (building on top of Unreal MassEntity).

We are building an extension to Schola, delivered as part of an Unreal Engine plugin with Python-side support, that adds optimization and support for five NPC types:
1. **Adaptive NPCs** learn online from player behavior and adjust during play.
2. **Event-Based NPCs** run inference only when subscribed game events fire, such as a turn starting or a player entering a zone.
3. **Background NPCs** make large crowds affordable through batched inference, culling of off-screen agents, and integration with Unreal's Mass Entity framework.
4. **Shared-Policy NPCs** reuse a common network base with specialized outputs for different roles, so related behaviors run faster than separate models.
5. **Advanced Strategy NPCs** *(optional stretch goal)* look ahead by simulating opponent moves, as top strategy-game AIs do. This is hard to do with Schola's current inference interface.

Our project lets adapt and deploy our NPC implementations in their games, instead of writing training and inference boilerplate. We target game developers using Unreal Engine who want learned NPC behavior across genres, especially turn-based strategy, FPS, and large-scale simulations, without building custom infrastructure.

Alexander Cann, Member of Technical Staff at AMD, is our partner and lead developer of Schola. His interests cover deep neural networks, reinforcement learning, and the intersection of AI and games.

Success looks like when a developer can seamlessly choose the correct NPC type, attach to a trained model, and get efficient behavior for their game genre and usecase.

#### Q2: Who are your target users?

Primary user: A gameplay or AI programmer at an indie or mid-sized studio. They work in Unreal Engine daily using C++ and Blueprints, and know behavior trees and navigation well. They have basic machine learning knowledge, such as having followed a tutorial or trained a simple model, but they are not ML researchers.

What they want: NPCs that feel smarter than hand-scripted behavior trees, without building a custom ML pipeline. For example, a programmer on a turn-based strategy game who wants an AI opponent that learns good tactics, but can't spend weeks on infrastructure.

Their pain points: Running RL models in a shipping game means writing custom code for batching and scheduling. Frame budgets are tight, so per-frame inference for every NPC is too costly.

Secondary users:
- Technical or AI designers who want an streamlined UI/API to tune advanced NPC behavior in Unreal.
- Machine learning engineers on game teams who train policies in Python (PyTorch, Stable Baselines3, Ray RLlib) and need an easy path to deploy them in Unreal.
- Solo developers and game jam teams experimenting with learned NPCs, who benefit most from ready-made examples.

#### Q3: Why would your users choose your product? What are they using today to solve their problem/need?

Most teams today write custom inference code for each NPC type, run every agent's model every frame, or stick with behavior trees. Schola covers training and basic inference, but turn-based triggers, crowds, shared networks and online adaptation are left to the developer. Our project:

- Saves time and money, helping developers avoid implementing writing boilerplate
- Provides optimized NPC implementations, increasing performance and lowering demands on hardware
- Makes advanced ML/RL techniques more accessible, such as through pretrained weights for common cases (e.g. basic crowd movement)
- Enables completely new behavior, such as online learning powered NPCs

Existing alternatives: Unreal's behavior trees and Mass Entity cover scripted behavior and large-scale simulation, not learned policies. Generic ML runtimes run models but do not typically offer support for turns, culling or entity batches.

Partner fit: GPUOpen is AMD's open-source initiative giving game developers free tools to get the most out of GPU hardware. Efficient, batched, open-source inference for NPCs is squarely in that mission.

#### Q4: What are the user stories that make up the Minumum Viable Product (MVP)?

* Event-based NPCs: As a game developer, I want NPCs to subscribe to selected gameplay events and perform inference when those events occur, so that I can coordinate their decisions with turn changes or other relevant game events.
* Learning during gameplay: As a game developer, I want NPC policies to learn from player interactions during gameplay, so that NPCs can adapt their strategies to the player’s behavior.
* Efficient background agents: As a researcher, I want to batch inference requests across many background agents, including agents managed through Unreal’s Mass Entity framework, so that I can study large-scale interactions between ML-controlled agents while staying within the simulation’s performance budget.
* Shared-policy NPCs: As a machine learning engineer, I want NPC policies with a shared network backbone and specialized output heads to reuse the backbone’s computation for the same observation, so that I can produce multiple specialized decisions with less computation than running each policy independently.
* Advanced strategy NPCs: As a game developer, I want NPCs to evaluate possible future actions and opponent responses using a search strategy such as Monte Carlo Tree Search, so that they can make decisions that account for longer-term consequences.
<img width="1686" height="728" alt="image" src="https://github.com/user-attachments/assets/12e21c4b-d702-4e88-98a8-31cfb19497be" />



#### Q5: Have you decided on how you will build it? Share what you know now or tell us the options you are considering.

We extend Schola's existing architecture. Schola already pairs a Python training side with an Unreal C++ runtime, connected by gRPC. We add new runtime modules on the Unreal side and small additions on the Python side.

(Existing) Stack:
- C++ and Blueprints for Unreal Engine 5 plugin modules
- Python for training and model export, using Schola's existing Stable Baselines3, Ray RLlib and Gymnasium adapters, plus PyTorch
- Protocol Buffers and gRPC for Python-to-Unreal communication (already in Schola)
- Unreal's Neural Network Engine (NNE), via Schola's ScholaNNE module, for in-game inference

Architecture: We simply add new modules on top of the inference and interactor layers, roughly one per NPC type:
- Event-based scheduler: triggers inference from subscribed game events
- Batched inference manager: groups many agents into one call, with culling and Mass Entity integration
- Shared-policy runner: runs a common network base once, then branch out
- Online adaptation: streams player-behavior data over gRPC to Python, which updates the policy and returns new weights
- Search-based strategy support (optional): a lookahead interface for simulating opponent moves

----
## Intellectual Property Confidentiality Agreement 

Our code will be released under the MIT License, consistent with the existing Schola codebase, and AMD will have access to the deployed system during the course.

----

## Teamwork Details

#### Q6: Have you met with your team?

For our team-building activity, we decided to play Wordle together online. During this activity, we were able to get to know each other better and it was a lot of fun. Some fun facts we collected were that Chloe used to have a pet turtle, and both Olivia and Jace can type 160+ wpm.

<img width="455" height="345" alt="Screenshot 2026-10-08 at 12 46 17 AM" src="https://github.com/user-attachments/assets/7df4454a-9ee6-4445-b7cf-16124ce3c45a" />

#### Q7: What are the roles & responsibilities on the team?

Our team is divided into four development roles, each responsible for a different class of NPC behavior. Each role is responsible for implementing the respective NPC detailed in **Q1.**

1. **Adaptive NPCs/Online learning:** 
2. **Event-Based NPCs:** 
3. **Background NPCs:** 
4. **Shared-Policy NPCs:** 

* Max - Partner liaison, Background NPCs
* Prithvi - Event-Based and Background NPCs.
* Clementine - Shared Policy NPCs: She's interested in shared-policy inference because it combines machine learning, inference optimization, and system integration. 
* Jason - Adaptive NPCs/Online learning: He is interested in evaluating ML algorithms, as well as learning more about model updating and data pipelines.
* Jaela - Background/shared policy Npcs (inference): She is interested in machine learning engineering and wants to learn more about efficient model inference.
* Jace - Adaptive NPCs/Online learning: Similarly to Olivia, he is interested in this because game dev has always been a direction he’s wanted to explore.
* Olivia - Adaptive NPCs/Online learning: She is interested in this because of her strong gamedev background: online learning has a more tangible effect on games which is something she would like to explore.
* Chloe - Adaptive NPCs/Online learning: Chloe is most interested in the online learning portion because she has an interest in video games and game development, which the online learning portion is has large implications for.

#### Q8: How will you work as a team?

We plan to meet weekly (ad hoc) as a team over discord to share progress, discuss any issues, and plan our next steps. We will also schedule additional meetings or coding sessions when needed, especially for debugging and integrating different parts of the project. We will also use GitHub issues to track tasks and pull requests to review each other's code.

We have been meeting with our AMD project partner to discuss the project scope and expectations. Our first meeting focused on exploring possible directions for extending Schola, including advanced inference, environment generation, and agent benchmarking. A follow-up meeting was planned to narrow down these ideas and agree on a realistic scope for the semester.

For the rest of the term, we plan to meet with our partner regularly over Microsoft Teams to share updates, ask questions, and get feedback on our progress. We will use Teams for day-to-day communication and email for more formal updates.

#### Q9: How will you organize your team?

We use Discord to communicate, with separate channels for things like meeting summaries, announcements, important documents and general discussions. We will also set up a shared Google folder. For to-dos, assigning task and prioritizing tasks, we will use the GitHub Projects feature to keep track of issues, tasks, blockers and ideas. We will also use this to track the flow of a task from inception to completion.

#### Q10: What are the rules regarding how your team works?

**Communications:**

**What is the expected frequency? What methods/channels will be used?**

Ping each other on discord for quick internal communication, discord calls for team meetings; Microsoft teams for partner communication/ weekly meetings with AMD members/ more formal project discussions

**If you have a partner project, what is your process for communicating with your partner? Who is responsible?**

Through Microsoft Teams. The partner liaison (Max) is responsible for following up and communicating on behalf of the team. We take meeting notes as well and formalize ideas and expectations regularly after meetings.

**Collaboration:**

**How are people held accountable for attending meetings, completing action items? What is your process?**

If a group member is late/not showing up for a meeting, we will @ping them on Discord (our main communication platform). In our server, we organize by channels, such as announcements and to-do’s, and we ping members according to the message content. 

**How will you address the issue if one person doesn't contribute or is not responsive?**

First of all, we will try to resolve the situation through contacting the person and sorting out the reason behind their lack of contribution. We will reiterate the tasks they were assigned and the communication standards we agreed upon. However, if they are not responsive for a period of time, we will escalate the issue to the TA, outlining the efforts we made to reconcile the issue ourselves.

## Organisation Details

#### Q11. How does your team fit within the overall team organisation of the partner?

Our team will be taking on a product development role as we are developing new features for AMD’s existing Schola framework. We have had discussions with our partner and have decided this is a good scope for the project and AMD’s needs. We will be working on infrastructure for several NPC cases, including as mentioned as prior: adaptive, event-driven, background, shared-policy, and advanced strategy NPCs [described in more detail in **Q1**]. We fit this role because the project requires a combination of machine-learning, game-engine development and software and system architecture. All members of the team have experience with C++, RL, and building software systems through coursework, projects and internships.

#### Q12. How does your project fit within the overall product from the partner?

Schola is AMD’s open-source reinforcement learning framework that connects game engines with Python-based training and inference tools. Our team is responsible for building on the existing framework by adding support for more advanced NPC behaviours and inference patterns. Specifically, we will focus on four main features: adaptive NPCs that learn from player behavior, event-based NPCs that perform inference when specific events occur, background NPCs that use batched inference for better efficiency, and shared-policy NPCs that reuse parts of neural networks. Two other student teams are working on Unity Godot ports of Schola.

A successful project would deliver working, tested, and documented implementations of the 4 core NPC features that can be integrated into Schola.

## Potential Risks

#### Q13. What are some potential risks to your project?

* **Scope**
  - This project involves several NPC features, each with different requirements. Implementing and testing all four features might be challenging.

* **Exisiting codebase constraints**
  - Since we are not creating a new project and simply adding onto an existing framework, we will not have as much control over the scope/implementation. 

* **Concept complexity**
  - Online learning is also a topic that is difficult and theory-heavy, so there may be a learning curve. 

#### Q14. What are some potential mitigation strategies for the risks you identified?

One issue noted was the scope, with multiple features with unique requirements. We could potentially lower this risk by focusing on the planning portion, creating tasks, estimating time per task and reflecting on prediction accuracy every sprint. If we notice that there are too many tasks for the time remaining, we will communicate quickly with each other and the company to determine next steps for prioritization. We may also add/remove user stories depending on how we are progressing with exploring/implementing new topics.

