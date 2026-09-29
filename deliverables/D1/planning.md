# AMD-Schola/The hard workers

## Product Details
 
#### Q1: What is the product?
We are building a Godot Engine port of AMD Schola, an open-source cross-platform reinforcement learning library currently built for Unreal Engine. We are partnering with AMD to extend their tool to support Godot. Our partners are: Alexander Cann (Member of Technical Staff) and Michael Liu (Senior Software Engineer). This tool will allow developers to natively define Reinforcement Learning (RL) Environments and Agents within Godot, attaching modular sensors and actuators, and connecting them to Python-based RL frameworks like Gymnasium, RLlib, or Stable-Baselines3. For example, a developer can create a racing car in Godot and train it to navigate a track using RL, without having to write the complex engine-to-Python communication logic from scratch.

#### Q2: Who are your target users?
- Game developers building NPCs or AI gameplay systems using Godot.
- AI / RL researchers needing a lightweight, accessible engine (Godot) as a training environment.
- Robotics and sim-to-real practitioners leveraging game engines for prototyping simulation environments before transferring to hardware.

#### Q3: Why would your users choose your product? What are they using today to solve their problem/need?
Currently, developers wanting to use Godot for RL either have to write custom sockets/RPC layers from scratch or rely on unsupported/unofficial Godot plugins which may not support the latest Python framework updates. Our product brings the official, actively supported AMD Schola architecture to Godot. It saves significant time by providing pre-built modular sensors, actuators, and an established gRPC communication layer to connect Godot directly to robust Python RL frameworks. This aligns with AMD's goal of broadening access to machine learning tools.

#### Q4: What are the user stories that make up the Minumum Viable Product (MVP)?

 * At least 5 user stories concerning the main features of the application - note that this can broken down further
 * You must follow proper user story format (as taught in lecture) ```As a <user of the app>, I want to <do something in the app> in order to <accomplish some goal>```
 * User stories must contain acceptance criteria. Examples of user stories with different formats can be found here: https://www.justinmind.com/blog/user-story-examples/. **It is important that you provide a link to an artifact containing your user stories**.
 * If you have a partner, these must be reviewed and accepted by them. You need to include the evidence of partner approval (e.g., screenshot from email) or at least communication to the partner (e.g., email you sent)

US1: Defining an Environment

As a Godot developer, I want to create an RL environment by implementing a simple interface, in order to train an agent without writing networking code.

US2: Declaring Spaces

As a Godot developer, I want to declare my agent's observation and action spaces using reusable types, in order to tell Python the shape of my problem.

US3: Transport - gRPC Server

As an ML practitioner, I want my Godot game to answer the same gRPC calls Unreal does, in order to reuse Python's existing training tools unmodified.

US4: Core - The Connector Loop

As a Godot developer, I want reset/step/auto-reset to behave exactly like Unreal's, in order for training to work identically across engines.

US5: Godot Bindings + The Demo Environment

As a Godot developer, I want to build an RL environment using normal Godot nodes and the Inspector, in order to work the way I already work in Godot.

US6: ONNX Export (Python)

As a Godot developer, I want to export a trained policy to ONNX, in order to run it later without Python.

US7: ONNX Inference + Shipping

As a Godot developer, I want a trained policy to drive my agent with Python closed, and to ship my game without training bloat.

#### Q5: Have you decided on how you will build it? Share what you know now or tell us the options you are considering.

> Short (1-2 min' read max)
 * What is the technology stack? Specify languages, frameworks, libraries, PaaS products or tools to be used or being considered. 
 * How will you deploy the application?
 * Describe the architecture - what are the high level components or patterns you will use? Diagrams are useful here. 
 * Will you be using third party applications or APIs? If so, what are they?

----
## Intellectual Property Confidentiality Agreement 
> Note this section is **not marked** but must be completed briefly if you have a partner. If you have any questions, please ask on Piazza.
>  
**By default, you own any work that you do as part of your coursework.** However, some partners may want you to keep the project confidential after the course is complete. As part of your first deliverable, you should discuss and agree upon an option with your partner. Examples include:
1. You can share the software and the code freely with anyone with or without a license, regardless of domain, for any use.
2. You can upload the code to GitHub or other similar publicly available domains.
3. You will only share the code under an open-source license with the partner but agree to not distribute it in any way to any other entity or individual. 
4. You will share the code under an open-source license and distribute it as you wish but only the partner can access the system deployed during the course.
5. You will only reference the work you did in your resume, interviews, etc. You agree to not share the code or software in any capacity with anyone unless your partner has agreed to it.

**Your partner cannot ask you to sign any legal agreements or documents pertaining to non-disclosure, confidentiality, IP ownership, etc.**

Briefly describe which option you have agreed to.

----

## Teamwork Details

#### Q6: Have you met with your team?

Do a team-building activity in-person or online. This can be playing an online game, meeting for bubble tea, lunch, or any other activity you all enjoy.
* Get to know each other on a more personal level.
* Provide a few sentences on what you did and share a picture or other evidence of your team building activity.
* Share at least three fun facts from members of you team (total not 3 for each member).


#### Q7: What are the roles & responsibilities on the team?

Describe the different roles on the team and the responsibilities associated with each role (e.g., frontend, database). 
 * Roles should reflect the structure of your team and be appropriate for your project. One person may have multiple roles.  
 * Add role(s) to your Team-[Team_Number]-[Team_Name].csv file on the main folder.
 * At least one person must be identified as the dedicated partner liaison. They need to have great organization and communication skills.
 * Everyone must contribute to code. Students who don't contribute to code enough will receive a lower mark at the end of the term.

List each team member and:
 * A description of their role(s) and responsibilities including the components they'll work on and non-software related work
 * Why did you choose them to take that role? Specify if they are interested in learning that part, experienced in it, or any other reasons. Do no make things up. This part is not graded but may be reviewed later.


#### Q8: How will you work as a team?

Describe meetings (and other events) you are planning to have. 
 * When and where? Recurring or ad hoc? In-person or online?
 * What's the purpose of each meeting?
 * Other events could be coding sessions, code reviews, quick weekly sync meeting online, etc.
 * You should have 2 meetings with your project partner (if you have one) before D1 is due. Describe them here:
   * You must keep track of meeting minutes and add them to your repo under "deliverables/minutes" folder
   * You must have a regular meeting schedule established for the rest of the term.  
  
#### Q9: How will you organize your team?

List/describe the artifacts you will produce to organize your team. (We strongly recommend that you use standard collaboration tools like Linear.app, Jira, Slack, Discord, GitHub.)       

 * Artifacts can be To-Do lists, Task boards, schedule(s), meeting minutes, etc.
 * We want to understand:
   * How do you keep track of what needs to get done? (You must grant your TA and partner access to systems you use to manage work)
   * **How do you prioritize tasks?**
   * How do tasks get assigned to team members?
   * How do you determine the status of work from inception to completion?

#### Q10: What are the rules regarding how your team works?

**Communications:**
 * What is the expected frequency? What methods/channels will be used? 
 * If you have a partner project, what is your process for communicating with your partner? Who is responsible?
 
**Collaboration:**
 * How are people held accountable for attending meetings, completing action items? What is your process?
 * How will you address the issue if one person doesn't contribute or is not responsive?

## Organisation Details

#### Q11. How does your team fit within the overall team organisation of the partner?
Our team functions as an external feature expansion team for AMD. The partner's team developed the core Schola library and the Unreal implementation. We are taking the role of porting this functionality to a new engine (Godot), effectively opening up a new platform for their product. We act semi-autonomously, relying on their Unreal plugin as a reference architecture, and contributing back to their open-source ecosystem.

#### Q12. How does your project fit within the overall product from the partner?
Our project is a horizontal expansion of the AMD Schola product. Currently, Schola provides an Unreal Engine plugin and an engine-agnostic Python package. Our project provides a Godot plugin that interfaces with the exact same Python package. It fits seamlessly alongside the Unreal plugin, offering developers a choice of game engine while keeping the RL training pipeline identical. The partner considers success to be achieving feature parity with the Unreal reference on Godot and validating end-to-end training on at least one benchmark environment.

## Potential Risks

#### Q13. What are some potential risks to your project?
* Now that you have defined your project, what risks can you identify that might impact it?
* Some examples of risks at this planning stage could include:
  * Uncertainties regarding a specific feature
  * Misaligned expectations or conflicts
  * Lack of clarity in execution or decision-making
  * Limited access to data, systems, or other dependencies
  * User stories that are too abstract or too simple
* For each risk, provide a brief bullet point and then explain the risk in detail. 

#### Q14. What are some potential mitigation strategies for the risks you identified?
* Examples of mitigation strategies:
  * More communication with the partner might help with improving clarity.
  * Adding more details for an user story might make it less abstract.
  * Adding an extra user story might increase the project complexity, making it less simple.
* It's ok if you are unable to find mitigation strategies for all the risks right now.
