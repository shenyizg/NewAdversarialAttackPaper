# Latest Large Language Model Attack Papers
**update at 2026-01-19 16:15:10**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Membership Inference on LLMs in the Wild**

野外LLM会员推断 cs.CL

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11314v1) [paper-pdf](https://arxiv.org/pdf/2601.11314v1)

**Authors**: Jiatong Yi, Yanyang Li

**Abstract**: Membership Inference Attacks (MIAs) act as a crucial auditing tool for the opaque training data of Large Language Models (LLMs). However, existing techniques predominantly rely on inaccessible model internals (e.g., logits) or suffer from poor generalization across domains in strict black-box settings where only generated text is available. In this work, we propose SimMIA, a robust MIA framework tailored for this text-only regime by leveraging an advanced sampling strategy and scoring mechanism. Furthermore, we present WikiMIA-25, a new benchmark curated to evaluate MIA performance on modern proprietary LLMs. Experiments demonstrate that SimMIA achieves state-of-the-art results in the black-box setting, rivaling baselines that exploit internal model information.

摘要: 成员资格推理攻击（MIA）是大型语言模型（LLM）不透明训练数据的重要审计工具。然而，现有技术主要依赖于不可访问的模型内部（例如，logits）或在严格的黑匣子设置中跨域的普遍性较差，只有生成的文本可用。在这项工作中，我们提出了SimMIA，这是一个强大的MIA框架，通过利用先进的抽样策略和评分机制，专为这种纯文本制度量身定制。此外，我们还推出了WikiMIA-25，这是一个新基准，旨在评估现代专有LLM上的MIA性能。实验表明，SimMIA在黑匣子设置中实现了最先进的结果，与利用内部模型信息的基线相媲美。



## **2. SD-RAG: A Prompt-Injection-Resilient Framework for Selective Disclosure in Retrieval-Augmented Generation**

SD-RAG：用于检索增强一代中选择性披露的预算注入弹性框架 cs.CR

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11199v1) [paper-pdf](https://arxiv.org/pdf/2601.11199v1)

**Authors**: Aiman Al Masoud, Marco Arazzi, Antonino Nocera

**Abstract**: Retrieval-Augmented Generation (RAG) has attracted significant attention due to its ability to combine the generative capabilities of Large Language Models (LLMs) with knowledge obtained through efficient retrieval mechanisms over large-scale data collections. Currently, the majority of existing approaches overlook the risks associated with exposing sensitive or access-controlled information directly to the generation model. Only a few approaches propose techniques to instruct the generative model to refrain from disclosing sensitive information; however, recent studies have also demonstrated that LLMs remain vulnerable to prompt injection attacks that can override intended behavioral constraints. For these reasons, we propose a novel approach to Selective Disclosure in Retrieval-Augmented Generation, called SD-RAG, which decouples the enforcement of security and privacy constraints from the generation process itself. Rather than relying on prompt-level safeguards, SD-RAG applies sanitization and disclosure controls during the retrieval phase, prior to augmenting the language model's input. Moreover, we introduce a semantic mechanism to allow the ingestion of human-readable dynamic security and privacy constraints together with an optimized graph-based data model that supports fine-grained, policy-aware retrieval. Our experimental evaluation demonstrates the superiority of SD-RAG over baseline existing approaches, achieving up to a $58\%$ improvement in the privacy score, while also showing a strong resilience to prompt injection attacks targeting the generative model.

摘要: 检索增强生成（RAG）因其能够将大型语言模型（LLM）的生成能力与通过大规模数据集合的高效检索机制获得的知识相结合而受到广泛关注。目前，大多数现有方法都忽视了与将敏感或访问控制信息直接暴露给生成模型相关的风险。只有少数方法提出了指示生成模型避免披露敏感信息的技术;然而，最近的研究也表明，LLM仍然容易受到可以推翻预期行为约束的提示注入攻击。出于这些原因，我们提出了一种新型的检索增强生成中的选择性披露方法，称为SD-RAG，它将安全和隐私约束的实施与生成过程本身分开。SD-RAG不是依赖预算级别的保障措施，而是在增强语言模型的输入之前在检索阶段应用清理和披露控制。此外，我们引入了一种语义机制，允许吸收人类可读的动态安全和隐私约束，以及一个优化的基于图形的数据模型，该模型支持细粒度、策略感知检索。我们的实验评估证明了SD-RAG相对于现有基线方法的优越性，在隐私评分方面实现了高达58%的改进，同时还表现出了强大的韧性，可以引发针对生成模型的注入攻击。



## **3. AJAR: Adaptive Jailbreak Architecture for Red-teaming**

Asimmon：红色团队的自适应越狱架构 cs.CR

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.10971v1) [paper-pdf](https://arxiv.org/pdf/2601.10971v1)

**Authors**: Yipu Dou, Wang Yang

**Abstract**: As Large Language Models (LLMs) evolve from static chatbots into autonomous agents capable of tool execution, the landscape of AI safety is shifting from content moderation to action security. However, existing red-teaming frameworks remain bifurcated: they either focus on rigid, script-based text attacks or lack the architectural modularity to simulate complex, multi-turn agentic exploitations. In this paper, we introduce AJAR (Adaptive Jailbreak Architecture for Red-teaming), a proof-of-concept framework designed to bridge this gap through Protocol-driven Cognitive Orchestration. Built upon the robust runtime of Petri, AJAR leverages the Model Context Protocol (MCP) to decouple adversarial logic from the execution loop, encapsulating state-of-the-art algorithms like X-Teaming as standardized, plug-and-play services. We validate the architectural feasibility of AJAR through a controlled qualitative case study, demonstrating its ability to perform stateful backtracking within a tool-use environment. Furthermore, our preliminary exploration of the "Agentic Gap" reveals a complex safety dynamic: while tool usage introduces new injection vectors via code execution, the cognitive load of parameter formatting can inadvertently disrupt persona-based attacks. AJAR is open-sourced to facilitate the standardized, environment-aware evaluation of this emerging attack surface. The code and data are available at https://github.com/douyipu/ajar.

摘要: 随着大型语言模型（LLM）从静态聊天机器人演变为能够执行工具的自主代理，人工智能安全的格局正在从内容审核转向动作安全。然而，现有的红队框架仍然存在分歧：它们要么专注于严格的基于脚本的文本攻击，要么缺乏模拟复杂的多回合代理开发的架构模块化。在本文中，我们介绍了April（自适应越狱架构红队），一个概念验证框架，旨在通过协议驱动的认知演示来弥合这一差距。基于Petri的强大运行时，April利用模型上下文协议（MCP）将对抗逻辑从执行循环中解耦，将X-Teaming等最先进的算法封装为标准化的即插即用服务。我们通过受控定性案例研究验证了Atomic的架构可行性，展示了其在工具使用环境中执行有状态回溯的能力。此外，我们对“统计差距”的初步探索揭示了一个复杂的安全动态：虽然工具使用通过代码执行引入新的注入载体，但参数格式的认知负载可能会无意中扰乱基于人物的攻击。Aspects是开源的，以促进对这一新兴攻击表面进行标准化、环境意识的评估。代码和数据可在https://github.com/douyipu/ajar上获取。



## **4. Beyond Max Tokens: Stealthy Resource Amplification via Tool Calling Chains in LLM Agents**

Beyond Max Tokens：通过LLM代理中的工具调用链秘密地扩大资源 cs.CR

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.10955v1) [paper-pdf](https://arxiv.org/pdf/2601.10955v1)

**Authors**: Kaiyu Zhou, Yongsen Zheng, Yicheng He, Meng Xue, Xueluan Gong, Yuji Wang, Kwok-Yan Lam

**Abstract**: The agent-tool communication loop is a critical attack surface in modern Large Language Model (LLM) agents. Existing Denial-of-Service (DoS) attacks, primarily triggered via user prompts or injected retrieval-augmented generation (RAG) context, are ineffective for this new paradigm. They are fundamentally single-turn and often lack a task-oriented approach, making them conspicuous in goal-oriented workflows and unable to exploit the compounding costs of multi-turn agent-tool interactions. We introduce a stealthy, multi-turn economic DoS attack that operates at the tool layer under the guise of a correctly completed task. Our method adjusts text-visible fields and a template-governed return policy in a benign, Model Context Protocol (MCP)-compatible tool server, optimizing these edits with a Monte Carlo Tree Search (MCTS) optimizer. These adjustments leave function signatures unchanged and preserve the final payload, steering the agent into prolonged, verbose tool-calling sequences using text-only notices. This compounds costs across turns, escaping single-turn caps while keeping the final answer correct to evade validation. Across six LLMs on the ToolBench and BFCL benchmarks, our attack expands tasks into trajectories exceeding 60,000 tokens, inflates costs by up to 658x, and raises energy by 100-560x. It drives GPU KV cache occupancy from <1% to 35-74% and cuts co-running throughput by approximately 50%. Because the server remains protocol-compatible and task outcomes are correct, conventional checks fail. These results elevate the agent-tool interface to a first-class security frontier, demanding a paradigm shift from validating final answers to monitoring the economic and computational cost of the entire agentic process.

摘要: 代理-工具通信循环是现代大型语言模型（LLM）代理中的一个关键攻击面。现有的拒绝服务（Dock）攻击主要通过用户提示或注入的检索增强生成（RAG）上下文触发，对于这种新范式来说无效。它们基本上是单轮，并且通常缺乏面向任务的方法，这使得它们在面向目标的工作流程中引人注目，并且无法利用多轮代理工具交互的复合成本。我们引入了一种隐蔽的、多轮经济性的拒绝服务攻击，以正确完成的任务为幌子在工具层运行。我们的方法在良性的、模型上下文协议（HCP）兼容的工具服务器中调整文本可见字段和模板管理的返回策略，并使用蒙特卡洛树搜索（MCTS）优化器优化这些编辑。这些调整使函数签名保持不变并保留最终有效负载，使用纯文本通知引导代理进入延长、冗长的工具调用序列。这会增加轮流成本，避免单圈上限，同时保持最终答案的正确性以逃避验证。在Tools Bench和BFCL基准的六个LLM中，我们的攻击将任务扩展到超过60，000个代币的轨迹，使成本增加高达658倍，并将能量增加100- 560倍。它将GDPKV缓存占用率从<1%提高到35-74%，并将共运行吞吐量降低约50%。由于服务器保持协议兼容并且任务结果正确，因此传统检查会失败。这些结果将代理-工具界面提升到一流的安全前沿，要求范式从验证最终答案转向监控整个代理流程的经济和计算成本。



## **5. Detecting Winning Arguments with Large Language Models and Persuasion Strategies**

使用大型语言模型和说服策略检测获胜论点 cs.CL

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10660v1) [paper-pdf](https://arxiv.org/pdf/2601.10660v1)

**Authors**: Tiziano Labruna, Arkadiusz Modzelewski, Giorgio Satta, Giovanni Da San Martino

**Abstract**: Detecting persuasion in argumentative text is a challenging task with important implications for understanding human communication. This work investigates the role of persuasion strategies - such as Attack on reputation, Distraction, and Manipulative wording - in determining the persuasiveness of a text. We conduct experiments on three annotated argument datasets: Winning Arguments (built from the Change My View subreddit), Anthropic/Persuasion, and Persuasion for Good. Our approach leverages large language models (LLMs) with a Multi-Strategy Persuasion Scoring approach that guides reasoning over six persuasion strategies. Results show that strategy-guided reasoning improves the prediction of persuasiveness. To better understand the influence of content, we organize the Winning Argument dataset into broad discussion topics and analyze performance across them. We publicly release this topic-annotated version of the dataset to facilitate future research. Overall, our methodology demonstrates the value of structured, strategy-aware prompting for enhancing interpretability and robustness in argument quality assessment.

摘要: 检测论点文本中的说服力是一项具有挑战性的任务，对于理解人类沟通具有重要意义。这项工作调查了说服策略（例如对声誉的攻击、干扰和操纵性措辞）在确定文本说服力方面的作用。我们对三个带注释的论点数据集进行了实验：Winning Arguments（从Change My View子reddit构建）、Anthropic/Persuspect和Persuspect for Good。我们的方法利用大型语言模型（LLM）和多策略说服评分方法，指导六种说服策略的推理。结果表明，策略引导推理提高了说服力的预测。为了更好地了解内容的影响力，我们将Winning Arrangement数据集组织为广泛的讨论主题，并分析各个主题的表现。我们公开发布该数据集的主题注释版本，以促进未来的研究。总体而言，我们的方法论展示了结构化、策略感知的提示对于增强论点质量评估的可解释性和稳健性的价值。



## **6. Be Your Own Red Teamer: Safety Alignment via Self-Play and Reflective Experience Replay**

成为你自己的红色团队：通过自我游戏和反思体验重播实现安全调整 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10589v1) [paper-pdf](https://arxiv.org/pdf/2601.10589v1)

**Authors**: Hao Wang, Yanting Wang, Hao Li, Rui Li, Lei Sha

**Abstract**: Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial ``jailbreak'' attacks designed to bypass safety guardrails. Current safety alignment methods depend heavily on static external red teaming, utilizing fixed defense prompts or pre-collected adversarial datasets. This leads to a rigid defense that overfits known patterns and fails to generalize to novel, sophisticated threats. To address this critical limitation, we propose empowering the model to be its own red teamer, capable of achieving autonomous and evolving adversarial attacks. Specifically, we introduce Safety Self- Play (SSP), a system that utilizes a single LLM to act concurrently as both the Attacker (generating jailbreaks) and the Defender (refusing harmful requests) within a unified Reinforcement Learning (RL) loop, dynamically evolving attack strategies to uncover vulnerabilities while simultaneously strengthening defense mechanisms. To ensure the Defender effectively addresses critical safety issues during the self-play, we introduce an advanced Reflective Experience Replay Mechanism, which uses an experience pool accumulated throughout the process. The mechanism employs a Upper Confidence Bound (UCB) sampling strategy to focus on failure cases with low rewards, helping the model learn from past hard mistakes while balancing exploration and exploitation. Extensive experiments demonstrate that our SSP approach autonomously evolves robust defense capabilities, significantly outperforming baselines trained on static adversarial datasets and establishing a new benchmark for proactive safety alignment.

摘要: 大型语言模型（LLM）已实现非凡的功能，但仍然容易受到旨在绕过安全护栏的对抗性“越狱”攻击的影响。当前的安全对齐方法严重依赖于静态外部红色团队，利用固定的防御提示或预先收集的对抗数据集。这导致了一种过于适合已知模式的严格防御，并且未能概括为新颖、复杂的威胁。为了解决这一关键局限性，我们建议让模型成为自己的红色团队，能够实现自主和不断发展的对抗性攻击。具体来说，我们引入了安全自助游戏（STP），这是一个利用单个LLM在统一的强化学习（RL）循环中同时充当攻击者（生成越狱）和防御者（拒绝有害请求）的系统，动态进化攻击策略以发现漏洞，同时加强防御机制。为了确保Defender有效解决自助游戏期间的关键安全问题，我们引入了先进的反思体验重播机制，该机制利用整个过程中积累的经验库。该机制采用上置信界（UCB）抽样策略来关注回报较低的失败案例，帮助模型从过去的严重错误中学习，同时平衡探索和利用。大量实验表明，我们的STP方法可以自主进化强大的防御能力，显着优于在静态对抗数据集上训练的基线，并为主动安全调整建立了新的基准。



## **7. Defending Large Language Models Against Jailbreak Attacks via In-Decoding Safety-Awareness Probing**

通过解码内安全意识探测保护大型语言模型免受越狱攻击 cs.AI

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10543v1) [paper-pdf](https://arxiv.org/pdf/2601.10543v1)

**Authors**: Yinzhi Zhao, Ming Wang, Shi Feng, Xiaocui Yang, Daling Wang, Yifei Zhang

**Abstract**: Large language models (LLMs) have achieved impressive performance across natural language tasks and are increasingly deployed in real-world applications. Despite extensive safety alignment efforts, recent studies show that such alignment is often shallow and remains vulnerable to jailbreak attacks. Existing defense mechanisms, including decoding-based constraints and post-hoc content detectors, struggle against sophisticated jailbreaks, often intervening robust detection or excessively degrading model utility. In this work, we examine the decoding process of LLMs and make a key observation: even when successfully jailbroken, models internally exhibit latent safety-related signals during generation. However, these signals are overridden by the model's drive for fluent continuation, preventing timely self-correction or refusal. Building on this observation, we propose a simple yet effective approach that explicitly surfaces and leverages these latent safety signals for early detection of unsafe content during decoding. Experiments across diverse jailbreak attacks demonstrate that our approach significantly enhances safety, while maintaining low over-refusal rates on benign inputs and preserving response quality. Our results suggest that activating intrinsic safety-awareness during decoding offers a promising and complementary direction for defending against jailbreak attacks. Code is available at: https://github.com/zyz13590/SafeProbing.

摘要: 大型语言模型（LLM）在自然语言任务中取得了令人印象深刻的性能，并且越来越多地部署在现实世界的应用程序中。尽管做出了广泛的安全调整，但最近的研究表明，这种调整往往很浅，并且仍然容易受到越狱攻击。现有的防御机制，包括基于解码的约束和事后内容检测器，难以应对复杂的越狱，通常会干预稳健的检测或过度降低模型效用。在这项工作中，我们研究了LLM的解码过程，并进行了一个关键观察：即使成功越狱，模型在生成期间内部也会表现出潜在的安全相关信号。然而，这些信号被模型流畅延续的驱动力所覆盖，从而阻止了及时的自我纠正或拒绝。在这一观察的基础上，我们提出了一种简单而有效的方法，该方法明确地揭示并利用这些潜在的安全信号，以便在解码期间早期检测不安全内容。针对各种越狱攻击的实验表明，我们的方法显着提高了安全性，同时在良性输入上保持较低的过度拒绝率并保持响应质量。我们的结果表明，在解码过程中激活内在的安全意识为防御越狱攻击提供了一个有希望且补充的方向。代码可访问：https://github.com/zyz13590/SafeProbing。



## **8. ReasAlign: Reasoning Enhanced Safety Alignment against Prompt Injection Attack**

ReasAlign：推理增强的安全一致性以对抗即时注入攻击 cs.CR

15 pages, 10 figures

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10173v1) [paper-pdf](https://arxiv.org/pdf/2601.10173v1)

**Authors**: Hao Li, Yankai Yang, G. Edward Suh, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) have enabled the development of powerful agentic systems capable of automating complex workflows across various fields. However, these systems are highly vulnerable to indirect prompt injection attacks, where malicious instructions embedded in external data can hijack agent behavior. In this work, we present ReasAlign, a model-level solution to improve safety alignment against indirect prompt injection attacks. The core idea of ReasAlign is to incorporate structured reasoning steps to analyze user queries, detect conflicting instructions, and preserve the continuity of the user's intended tasks to defend against indirect injection attacks. To further ensure reasoning logic and accuracy, we introduce a test-time scaling mechanism with a preference-optimized judge model that scores reasoning steps and selects the best trajectory. Comprehensive evaluations across various benchmarks show that ReasAlign maintains utility comparable to an undefended model while consistently outperforming Meta SecAlign, the strongest prior guardrail. On the representative open-ended CyberSecEval2 benchmark, which includes multiple prompt-injected tasks, ReasAlign achieves 94.6% utility and only 3.6% ASR, far surpassing the state-of-the-art defensive model of Meta SecAlign (56.4% utility and 74.4% ASR). These results demonstrate that ReasAlign achieves the best trade-off between security and utility, establishing a robust and practical defense against prompt injection attacks in real-world agentic systems. Our code and experimental results could be found at https://github.com/leolee99/ReasAlign.

摘要: 大型语言模型（LLM）使得能够开发强大的代理系统，能够自动化跨各个领域的复杂工作流。然而，这些系统非常容易受到间接提示注入攻击，其中嵌入在外部数据中的恶意指令可以劫持代理行为。在这项工作中，我们提出了ReasAlign，一个模型级的解决方案，以提高对间接提示注入攻击的安全对齐。ReasAlign的核心思想是结合结构化推理步骤来分析用户查询，检测冲突指令，并保持用户预期任务的连续性，以抵御间接注入攻击。为了进一步确保推理逻辑和准确性，我们引入了测试时缩放机制，该机制具有偏好优化的判断模型，该模型对推理步骤进行评分并选择最佳轨迹。各种基准的综合评估表明，ReasAlign保持了与无防护模型相当的实用性，同时始终优于Meta SecAlign（先前最强的护栏）。在代表性的开放式CyberSecEval2基准测试中，其中包括多个注入任务，ReasAlign实现了94.6%的实用性和仅3.6%的ASR，远远超过了Meta SecAlign的最先进防御模型（56.4%的实用性和74.4%的ASR）。这些结果表明，ReasAlign实现了安全性和实用性之间的最佳权衡，在现实世界的代理系统中建立了针对即时注入攻击的强大而实用的防御。我们的代码和实验结果可以在https://github.com/leolee99/ReasAlign上找到。



## **9. Understanding and Preserving Safety in Fine-Tuned LLMs**

了解和维护精调LLM的安全性 cs.LG

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10141v1) [paper-pdf](https://arxiv.org/pdf/2601.10141v1)

**Authors**: Jiawen Zhang, Yangfan Hu, Kejia Chen, Lipeng He, Jiachen Ma, Jian Lou, Dan Li, Jian Liu, Xiaohu Yang, Ruoxi Jia

**Abstract**: Fine-tuning is an essential and pervasive functionality for applying large language models (LLMs) to downstream tasks. However, it has the potential to substantially degrade safety alignment, e.g., by greatly increasing susceptibility to jailbreak attacks, even when the fine-tuning data is entirely harmless. Despite garnering growing attention in defense efforts during the fine-tuning stage, existing methods struggle with a persistent safety-utility dilemma: emphasizing safety compromises task performance, whereas prioritizing utility typically requires deep fine-tuning that inevitably leads to steep safety declination.   In this work, we address this dilemma by shedding new light on the geometric interaction between safety- and utility-oriented gradients in safety-aligned LLMs. Through systematic empirical analysis, we uncover three key insights: (I) safety gradients lie in a low-rank subspace, while utility gradients span a broader high-dimensional space; (II) these subspaces are often negatively correlated, causing directional conflicts during fine-tuning; and (III) the dominant safety direction can be efficiently estimated from a single sample. Building upon these novel insights, we propose safety-preserving fine-tuning (SPF), a lightweight approach that explicitly removes gradient components conflicting with the low-rank safety subspace. Theoretically, we show that SPF guarantees utility convergence while bounding safety drift. Empirically, SPF consistently maintains downstream task performance and recovers nearly all pre-trained safety alignment, even under adversarial fine-tuning scenarios. Furthermore, SPF exhibits robust resistance to both deep fine-tuning and dynamic jailbreak attacks. Together, our findings provide new mechanistic understanding and practical guidance toward always-aligned LLM fine-tuning.

摘要: 微调是将大型语言模型（LLM）应用于下游任务的基本且普遍的功能。然而，它有可能大幅降低安全对准，例如，即使微调数据完全无害，也可以极大地增加越狱攻击的易感性。尽管在微调阶段在国防工作中受到越来越多的关注，但现有的方法仍面临着持续的安全-效用困境：强调安全性会损害任务性能，而优先考虑效用通常需要深度微调，这不可避免地会导致安全性急剧下降。   在这项工作中，我们通过对安全对齐的LLM中安全导向梯度和实用导向梯度之间的几何相互作用提出新的见解来解决这一困境。通过系统的实证分析，我们揭示了三个关键见解：（一）安全梯度位于低阶子空间，而效用梯度跨越更广泛的多维空间;（二）这些子空间通常呈负相关，导致微调过程中的方向冲突;（三）可以有效地估计主导安全方向从单个样本。在这些新颖见解的基础上，我们提出了安全保护微调（SPF），这是一种轻量级方法，可以显式地去除与低等级安全子空间冲突的梯度分量。从理论上讲，我们表明SPF保证了效用收敛，同时限制了安全漂移。从经验上看，即使在敌对的微调场景下，SPF也能始终如一地保持下游任务性能，并恢复几乎所有预先训练的安全对齐。此外，SPF对深度微调和动态越狱攻击都表现出强大的抵抗力。我们的研究结果共同为始终一致的LLM微调提供了新的机制理解和实践指导。



## **10. Privacy Enhanced PEFT: Tensor Train Decomposition Improves Privacy Utility Tradeoffs under DP-SGD**

隐私增强PEFT：张量列车分解改善DP-Singapore下的隐私效用权衡 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10045v1) [paper-pdf](https://arxiv.org/pdf/2601.10045v1)

**Authors**: Pradip Kunwar, Minh Vu, Maanak Gupta, Manish Bhattarai

**Abstract**: Fine-tuning large language models on sensitive data poses significant privacy risks, as membership inference attacks can reveal whether individual records were used during training. While Differential Privacy (DP) provides formal protection, applying DP to conventional Parameter-Efficient Fine-Tuning (PEFT) methods such as Low-Rank Adaptation (LoRA) often incurs substantial utility loss. In this work, we show that a more structurally constrained PEFT architecture, Tensor Train Low-Rank Adaptation (TTLoRA), can improve the privacy-utility tradeoff by shrinking the effective parameter space while preserving expressivity. To this end, we develop TTLoRA-DP, a differentially private training framework for TTLoRA. Specifically, we extend the ghost clipping algorithm to Tensor Train cores via cached contraction states, enabling efficient Differentially Private Stochastic Gradient Descent (DP-SGD) with exact per-example gradient norm computation without materializing full per-example gradients. Experiments on GPT-2 fine-tuning over the Enron and Penn Treebank datasets show that TTLoRA-DP consistently strengthens privacy protection relative to LoRA-DP while maintaining comparable or better downstream utility. Moreover, TTLoRA exhibits lower membership leakage even without DP training, using substantially smaller adapters and requiring on average 7.6X fewer parameters than LoRA. Overall, our results demonstrate that TTLoRA offers a practical path to improving the privacy-utility tradeoff in parameter-efficient language model adaptation.

摘要: 微调敏感数据上的大型语言模型会带来巨大的隐私风险，因为成员资格推断攻击可以揭示训练期间是否使用了个别记录。虽然差异隐私（DP）提供了形式的保护，但将DP应用于传统的参数高效微调（PEFT）方法（例如低等级自适应（LoRA））通常会导致巨大的效用损失。在这项工作中，我们表明，结构上限制更大的PEFT架构，张量列车低等级自适应（TTLoRA），可以通过缩小有效参数空间同时保留表达性来改善隐私与效用的权衡。为此，我们开发了TTLoRA-DP，这是TTLoRA的差异化私人培训框架。具体来说，我们通过缓存的收缩状态将幽灵剪辑算法扩展到张量列车核心，通过精确的每示例梯度规范计算来实现高效的差异私有随机梯度下降（DP-BCD），而无需具体化完整的每示例梯度。在Enron和Penn Treebank数据集上进行GPT-2微调的实验表明，TTLoRA-DP相对于LoRA-DP持续加强隐私保护，同时保持相当或更好的下游效用。此外，即使没有DP训练，TTLoRA也表现出更低的成员泄漏，使用小得多的适配器，所需参数平均比LoRA少7.6倍。总体而言，我们的结果表明TTLoRA为改善参数高效语言模型适应中的隐私与效用权衡提供了一种实用途径。



## **11. SoK: Privacy-aware LLM in Healthcare: Threat Model, Privacy Techniques, Challenges and Recommendations**

SoK：隐私意识LLM在医疗保健：威胁模型，隐私技术，挑战和建议 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10004v1) [paper-pdf](https://arxiv.org/pdf/2601.10004v1)

**Authors**: Mohoshin Ara Tahera, Karamveer Singh Sidhu, Shuvalaxmi Dass, Sajal Saha

**Abstract**: Large Language Models (LLMs) are increasingly adopted in healthcare to support clinical decision-making, summarize electronic health records (EHRs), and enhance patient care. However, this integration introduces significant privacy and security challenges, driven by the sensitivity of clinical data and the high-stakes nature of medical workflows. These risks become even more pronounced across heterogeneous deployment environments, ranging from small on-premise hospital systems to regional health networks, each with unique resource limitations and regulatory demands. This Systematization of Knowledge (SoK) examines the evolving threat landscape across the three core LLM phases: Data preprocessing, Fine-tuning, and Inference within realistic healthcare settings. We present a detailed threat model that characterizes adversaries, capabilities, and attack surfaces at each phase, and we systematize how existing privacy-preserving techniques (PPTs) attempt to mitigate these vulnerabilities. While existing defenses show promise, our analysis identifies persistent limitations in securing sensitive clinical data across diverse operational tiers. We conclude with phase-aware recommendations and future research directions aimed at strengthening privacy guarantees for LLMs in regulated environments. This work provides a foundation for understanding the intersection of LLMs, threats, and privacy in healthcare, offering a roadmap toward more robust and clinically trustworthy AI systems.

摘要: 大型语言模型（LLM）在医疗保健中越来越多地采用，以支持临床决策、总结电子健康记录（EHR）并加强患者护理。然而，由于临床数据的敏感性和医疗工作流程的高风险性质，这种集成带来了重大的隐私和安全挑战。这些风险在不同的部署环境中变得更加明显，从小型本地医院系统到区域医疗网络，每个环境都有独特的资源限制和监管要求。知识系统化（SoK）研究了LLM三个核心阶段不断变化的威胁格局：数据预处理、微调和现实医疗保健环境中的推理。我们提出了一个详细的威胁模型，描述了每个阶段的对手、能力和攻击表面，并系统化了现有的隐私保护技术（PPT）如何尝试缓解这些漏洞。虽然现有的防御措施显示出希望，但我们的分析发现，在保护不同运营层级的敏感临床数据方面存在持续的局限性。我们最后提出了阶段感知建议和未来研究方向，旨在加强受监管环境中LLM的隐私保障。这项工作为了解医疗保健领域LLM、威胁和隐私的交叉点提供了基础，并为迈向更强大、临床值得信赖的人工智能系统提供了路线图。



## **12. The Promptware Kill Chain: How Prompt Injections Gradually Evolved Into a Multi-Step Malware**

黑客软件杀死链：提示注射如何逐渐演变为多步骤恶意软件 cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09625v1) [paper-pdf](https://arxiv.org/pdf/2601.09625v1)

**Authors**: Ben Nassi, Bruce Schneier, Oleg Brodt

**Abstract**: The rapid adoption of large language model (LLM)-based systems -- from chatbots to autonomous agents capable of executing code and financial transactions -- has created a new attack surface that existing security frameworks inadequately address. The dominant framing of these threats as "prompt injection" -- a catch-all phrase for security failures in LLM-based systems -- obscures a more complex reality: Attacks on LLM-based systems increasingly involve multi-step sequences that mirror traditional malware campaigns. In this paper, we propose that attacks targeting LLM-based applications constitute a distinct class of malware, which we term \textit{promptware}, and introduce a five-step kill chain model for analyzing these threats. The framework comprises Initial Access (prompt injection), Privilege Escalation (jailbreaking), Persistence (memory and retrieval poisoning), Lateral Movement (cross-system and cross-user propagation), and Actions on Objective (ranging from data exfiltration to unauthorized transactions). By mapping recent attacks to this structure, we demonstrate that LLM-related attacks follow systematic sequences analogous to traditional malware campaigns. The promptware kill chain offers security practitioners a structured methodology for threat modeling and provides a common vocabulary for researchers across AI safety and cybersecurity to address a rapidly evolving threat landscape.

摘要: 基于大型语言模型（LLM）的系统的快速采用-从聊天机器人到能够执行代码和金融交易的自治代理-已经创建了一个新的攻击面，现有的安全框架无法充分解决。这些威胁的主要框架是“即时注入”-基于LLM的系统中安全故障的通用短语-掩盖了一个更复杂的现实：对基于LLM的系统的攻击越来越多地涉及反映传统恶意软件活动的多步序列。在本文中，我们提出，针对基于LLM的应用程序的攻击构成了一种独特的恶意软件，我们称之为\textit{恶意软件}，并介绍了一个五步杀伤链模型来分析这些威胁。该框架包括初始访问（即时注入），安全升级（越狱），持久性（内存和检索中毒），横向移动（跨系统和跨用户传播）和目标操作（从数据泄露到未经授权的交易）。通过将最近的攻击映射到这个结构，我们证明了LLM相关的攻击遵循类似于传统恶意软件活动的系统序列。软件杀伤链为安全从业人员提供了一种结构化的威胁建模方法，并为AI安全和网络安全领域的研究人员提供了一个通用词汇表，以应对快速发展的威胁环境。



## **13. SpatialJB: How Text Distribution Art Becomes the "Jailbreak Key" for LLM Guardrails**

SpatialJB：文本分发艺术如何成为法学硕士护栏的“越狱钥匙” cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09321v1) [paper-pdf](https://arxiv.org/pdf/2601.09321v1)

**Authors**: Zhiyi Mou, Jingyuan Yang, Zeheng Qian, Wangze Ni, Tianfang Xiao, Ning Liu, Chen Zhang, Zhan Qin, Kui Ren

**Abstract**: While Large Language Models (LLMs) have powerful capabilities, they remain vulnerable to jailbreak attacks, which is a critical barrier to their safe web real-time application. Current commercial LLM providers deploy output guardrails to filter harmful outputs, yet these defenses are not impenetrable. Due to LLMs' reliance on autoregressive, token-by-token inference, their semantic representations lack robustness to spatially structured perturbations, such as redistributing tokens across different rows, columns, or diagonals. Exploiting the Transformer's spatial weakness, we propose SpatialJB to disrupt the model's output generation process, allowing harmful content to bypass guardrails without detection. Comprehensive experiments conducted on leading LLMs get nearly 100% ASR, demonstrating the high effectiveness of SpatialJB. Even after adding advanced output guardrails, like the OpenAI Moderation API, SpatialJB consistently maintains a success rate exceeding 75%, outperforming current jailbreak techniques by a significant margin. The proposal of SpatialJB exposes a key weakness in current guardrails and emphasizes the importance of spatial semantics, offering new insights to advance LLM safety research. To prevent potential misuse, we also present baseline defense strategies against SpatialJB and evaluate their effectiveness in mitigating such attacks. The code for the attack, baseline defenses, and a demo are available at https://anonymous.4open.science/r/SpatialJailbreak-8E63.

摘要: 虽然大型语言模型（LLM）具有强大的功能，但它们仍然容易受到越狱攻击，这是其安全Web实时应用程序的关键障碍。目前的商业LLM提供商部署了输出护栏来过滤有害输出，但这些防御措施并非无懈可击。由于LLM依赖于自回归、逐个标记的推理，它们的语义表示对空间结构化扰动缺乏鲁棒性，例如跨不同的行、列或对角线重新分布标记。利用Transformer的空间弱点，我们建议SpatialJB扰乱模型的输出生成过程，允许有害内容绕过护栏而不被检测。在领先的LLM上进行的综合实验获得了近100%的ASB，证明了SpatialJB的高效性。即使在添加了OpenAI Moderation API等高级输出护栏后，SpatialJB仍始终保持超过75%的成功率，大幅优于当前的越狱技术。SpatialJB的提案暴露了当前护栏的一个关键弱点，并强调了空间语义的重要性，为推进LLM安全研究提供了新的见解。为了防止潜在的滥用，我们还提供了针对SpatialJB的基线防御策略，并评估其减轻此类攻击的有效性。攻击代码、基线防御和演示可在https://anonymous.4open.science/r/SpatialJailbreak-8E63上获取。



## **14. STaR: Sensitive Trajectory Regulation for Unlearning in Large Reasoning Models**

STaR：大型推理模型中取消学习的敏感轨迹调节 cs.AI

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09281v1) [paper-pdf](https://arxiv.org/pdf/2601.09281v1)

**Authors**: Jingjing Zhou, Gaoxiang Cong, Li Su, Liang Li

**Abstract**: Large Reasoning Models (LRMs) have advanced automated multi-step reasoning, but their ability to generate complex Chain-of-Thought (CoT) trajectories introduces severe privacy risks, as sensitive information may be deeply embedded throughout the reasoning process. Existing Large Language Models (LLMs) unlearning approaches that typically focus on modifying only final answers are insufficient for LRMs, as they fail to remove sensitive content from intermediate steps, leading to persistent privacy leakage and degraded security. To address these challenges, we propose Sensitive Trajectory Regulation (STaR), a parameter-free, inference-time unlearning framework that achieves robust privacy protection throughout the reasoning process. Specifically, we first identify sensitive content via semantic-aware detection. Then, we inject global safety constraints through secure prompt prefix. Next, we perform trajectory-aware suppression to dynamically block sensitive content across the entire reasoning chain. Finally, we apply token-level adaptive filtering to prevent both exact and paraphrased sensitive tokens during generation. Furthermore, to overcome the inadequacies of existing evaluation protocols, we introduce two metrics: Multi-Decoding Consistency Assessment (MCS), which measures the consistency of unlearning across diverse decoding strategies, and Multi-Granularity Membership Inference Attack (MIA) Evaluation, which quantifies privacy protection at both answer and reasoning-chain levels. Experiments on the R-TOFU benchmark demonstrate that STaR achieves comprehensive and stable unlearning with minimal utility loss, setting a new standard for privacy-preserving reasoning in LRMs.

摘要: 大型推理模型（LRM）具有先进的自动化多步推理，但它们生成复杂的思想链（CoT）轨迹的能力会带来严重的隐私风险，因为敏感信息可能会深入嵌入整个推理过程。现有的大型语言模型（LLM）的学习方法通常只关注修改最终答案，这对LRM来说是不够的，因为它们无法从中间步骤中删除敏感内容，导致持续的隐私泄露和安全性下降。为了应对这些挑战，我们提出了敏感轨迹调节（STaR），这是一个无参数的推理时间遗忘框架，可以在整个推理过程中实现强大的隐私保护。具体来说，我们首先通过语义感知检测来识别敏感内容。然后，我们通过安全提示前置注入全球安全约束。接下来，我们执行个体感知抑制，以动态阻止整个推理链中的敏感内容。最后，我们应用令牌级自适应过滤来防止生成期间出现精确和解释的敏感令牌。此外，为了克服现有评估协议的不足，我们引入了两个指标：多解码一致性评估（MC），衡量不同解码策略之间取消学习的一致性，以及多粒度成员资格推断攻击（MIA）评估，量化答案和推理链级别的隐私保护。R-TOFU基准测试的实验表明，STaR以最小的效用损失实现了全面、稳定的去学习，为LRM中的隐私保护推理设定了新标准。



## **15. KryptoPilot: An Open-World Knowledge-Augmented LLM Agent for Automated Cryptographic Exploitation**

KryptoPilot：一个开放世界知识增强的LLM代理，用于自动加密技术开发 cs.CR

14 Pages,4 figures

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09129v1) [paper-pdf](https://arxiv.org/pdf/2601.09129v1)

**Authors**: Xiaonan Liu, Zhihao Li, Xiao Lan, Hao Ren, Haizhou Wang, Xingshu Chen

**Abstract**: Capture-the-Flag (CTF) competitions play a central role in modern cybersecurity as a platform for training practitioners and evaluating offensive and defensive techniques derived from real-world vulnerabilities. Despite recent advances in large language models (LLMs), existing LLM-based agents remain ineffective on high-difficulty cryptographic CTF challenges, which require precise cryptanalytic knowledge, stable long-horizon reasoning, and disciplined interaction with specialized toolchains. Through a systematic exploratory study, we show that insufficient knowledge granularity, rather than model reasoning capacity, is a primary factor limiting successful cryptographic exploitation: coarse or abstracted external knowledge often fails to support correct attack modeling and implementation. Motivated by this observation, we propose KryptoPilot, an open-world knowledge-augmented LLM agent for automated cryptographic exploitation. KryptoPilot integrates dynamic open-world knowledge acquisition via a Deep Research pipeline, a persistent workspace for structured knowledge reuse, and a governance subsystem that stabilizes reasoning through behavioral constraints and cost-aware model routing. This design enables precise knowledge alignment while maintaining efficient reasoning across heterogeneous subtasks. We evaluate KryptoPilot on two established CTF benchmarks and in six real-world CTF competitions. KryptoPilot achieves a complete solve rate on InterCode-CTF, solves between 56 and 60 percent of cryptographic challenges on the NYU-CTF benchmark, and successfully solves 26 out of 33 cryptographic challenges in live competitions, including multiple earliest-solved and uniquely-solved instances. These results demonstrate the necessity of open-world, fine-grained knowledge augmentation and governed reasoning for scaling LLM-based agents to real-world cryptographic exploitation.

摘要: 夺旗（CTF）比赛作为培训从业者和评估源自现实世界漏洞的进攻和防御技术的平台，在现代网络安全中发挥着核心作用。尽管大型语言模型（LLM）最近取得了进展，但现有的基于LLM的代理在高难度加密CTF挑战中仍然无效，而这些挑战需要精确的密码分析知识、稳定的长期推理以及与专业工具链的纪律性交互。通过系统的探索性研究，我们表明，知识粒度不足，而不是模型推理能力，是限制成功加密技术利用的主要因素：粗糙或抽象的外部知识往往无法支持正确的攻击建模和实施。受这一观察的启发，我们提出了KryptoPilot，这是一种开放世界知识增强的LLM代理，用于自动加密技术利用。KryptoPilot通过深度研究管道、用于结构化知识重用的持久工作空间以及通过行为约束和成本感知模型路由稳定推理的治理子系统集成了动态开放世界知识获取。该设计可以实现精确的知识对齐，同时保持跨异类子任务的高效推理。我们根据两个既定的CTF基准和六场现实世界的CTF比赛评估KryptoPilot。KryptoPilot在InterCode-CTF上实现了完全解决率，在NYU-CTF基准上解决了56%至60%的加密挑战，并在现场比赛中成功解决了33个加密挑战中的26个，包括多个最早解决和最早解决的实例。这些结果证明了开放世界、细粒度知识增强和受管辖推理将基于LLM的代理扩展到现实世界的加密利用的必要性。



## **16. Too Helpful to Be Safe: User-Mediated Attacks on Planning and Web-Use Agents**

太有帮助而不安全：用户调解的对规划和网络使用代理的攻击 cs.CR

Keywords: LLM Agents; User-Mediated Attack; Agent Security; Human Factors in Cybersecurity; Web-Use Agents; Planning Agents; Benchmark

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.10758v1) [paper-pdf](https://arxiv.org/pdf/2601.10758v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Large Language Models (LLMs) have enabled agents to move beyond conversation toward end-to-end task execution and become more helpful. However, this helpfulness introduces new security risks stem less from direct interface abuse than from acting on user-provided content. Existing studies on agent security largely focus on model-internal vulnerabilities or adversarial access to agent interfaces, overlooking attacks that exploit users as unintended conduits. In this paper, we study user-mediated attacks, where benign users are tricked into relaying untrusted or attacker-controlled content to agents, and analyze how commercial LLM agents respond under such conditions. We conduct a systematic evaluation of 12 commercial agents in a sandboxed environment, covering 6 trip-planning agents and 6 web-use agents, and compare agent behavior across scenarios with no, soft, and hard user-requested safety checks. Our results show that agents are too helpful to be safe by default. Without explicit safety requests, trip-planning agents bypass safety constraints in over 92% of cases, converting unverified content into confident booking guidance. Web-use agents exhibit near-deterministic execution of risky actions, with 9 out of 17 supported tests reaching a 100% bypass rate. Even when users express soft or hard safety intent, constraint bypass remains substantial, reaching up to 54.7% and 7% for trip-planning agents, respectively. These findings reveal that the primary issue is not a lack of safety capability, but its prioritization. Agents invoke safety checks only conditionally when explicitly prompted, and otherwise default to goal-driven execution. Moreover, agents lack clear task boundaries and stopping rules, frequently over-executing workflows in ways that lead to unnecessary data disclosure and real-world harm.

摘要: 大型语言模型（LLM）使代理能够超越对话，走向端到端的任务执行，变得更有帮助。然而，这种有益的做法引入了新的安全风险，这些风险更多地来自于对用户提供的内容的操作，而不是直接的界面滥用。现有的代理安全研究主要集中在模型内部的漏洞或对代理接口的对抗性访问，忽略了利用用户作为意外管道的攻击。在本文中，我们研究了用户介导的攻击，其中良性用户被诱骗将不受信任或攻击者控制的内容中继给代理，并分析了商业LLM代理在这种情况下如何响应。我们在沙箱环境中对12个商业代理进行了系统评估，涵盖6个旅行规划代理和6个网络使用代理，并在无、软和硬用户请求的安全检查的情况下比较代理行为。我们的结果表明，代理太有帮助，默认情况下是安全的。在没有明确的安全要求的情况下，旅行计划代理在超过92%的情况下绕过了安全限制，将未经验证的内容转化为自信的预订指导。Web使用代理几乎确定地执行危险操作，支持的17个测试中有9个达到了100%的绕过率。即使用户表达了软或硬安全意图，约束绕过仍然很大，出行规划代理的比例分别高达54.7%和7%。这些调查结果表明，主要问题不是缺乏安全能力，而是其优先顺序。代理仅在显式提示时有条件地调用安全检查，否则默认为目标驱动执行。此外，代理缺乏明确的任务边界和停止规则，经常过度执行工作流程，导致不必要的数据泄露和现实世界的伤害。



## **17. Integrating APK Image and Text Data for Enhanced Threat Detection: A Multimodal Deep Learning Approach to Android Malware**

集成APK图像和文本数据以增强威胁检测：针对Android恶意软件的多模式深度学习方法 cs.CR

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08959v1) [paper-pdf](https://arxiv.org/pdf/2601.08959v1)

**Authors**: Md Mashrur Arifin, Maqsudur Rahman, Nasir U. Eisty

**Abstract**: As zero-day Android malware attacks grow more sophisticated, recent research highlights the effectiveness of using image-based representations of malware bytecode to detect previously unseen threats. However, existing studies often overlook how image type and resolution affect detection and ignore valuable textual data in Android Application Packages (APKs), such as permissions and metadata, limiting their ability to fully capture malicious behavior. The integration of multimodality, which combines image and text data, has gained momentum as a promising approach to address these limitations. This paper proposes a multimodal deep learning framework integrating APK images and textual features to enhance Android malware detection. We systematically evaluate various image types and resolutions across different Convolutional Neural Networks (CNN) architectures, including VGG, ResNet-152, MobileNet, DenseNet, EfficientNet-B4, and use LLaMA-2, a large language model, to extract and annotate textual features for improved analysis. The findings demonstrate that RGB images at higher resolutions (e.g., 256x256, 512x512) achieve superior classification performance, while the multimodal integration of image and text using the CLIP model reveals limited potential. Overall, this research highlights the importance of systematically evaluating image attributes and integrating multimodal data to develop effective malware detection for Android systems.

摘要: 随着零日Android恶意软件攻击变得越来越复杂，最近的研究强调了使用基于图像的恶意软件字节码表示来检测以前未见过的威胁的有效性。然而，现有的研究经常忽视图像类型和分辨率如何影响检测，并忽视Android应用程序包（APK）中有价值的文本数据，例如权限和元数据，从而限制了其完全捕获恶意行为的能力。结合图像和文本数据的多模式集成已成为解决这些限制的一种有希望的方法。本文提出了一个集成APK图像和文本特征的多模式深度学习框架，以增强Android恶意软件检测。我们系统地评估不同卷积神经网络（CNN）架构中的各种图像类型和分辨率，包括VGG、ResNet-152、MobileNet、DenseNet、EfficientNet-B4，并使用LLaMA-2（一种大型语言模型）来提取和注释文本特征以改进分析。研究结果表明，更高分辨率的Ruby图像（例如，256 x256、512 x512）实现了卓越的分类性能，而使用CLIP模型的图像和文本的多模式集成显示出有限的潜力。总体而言，这项研究强调了系统性评估图像属性和集成多模式数据以为Android系统开发有效的恶意软件检测的重要性。



## **18. Robust CAPTCHA Using Audio Illusions in the Era of Large Language Models: from Evaluation to Advances**

在大型语言模型时代使用音频幻象的稳健验证码：从评估到进步 cs.SD

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08516v1) [paper-pdf](https://arxiv.org/pdf/2601.08516v1)

**Authors**: Ziqi Ding, Yunfeng Wan, Wei Song, Yi Liu, Gelei Deng, Nan Sun, Huadong Mo, Jingling Xue, Shidong Pan, Yuekang Li

**Abstract**: CAPTCHAs are widely used by websites to block bots and spam by presenting challenges that are easy for humans but difficult for automated programs to solve. To improve accessibility, audio CAPTCHAs are designed to complement visual ones. However, the robustness of audio CAPTCHAs against advanced Large Audio Language Models (LALMs) and Automatic Speech Recognition (ASR) models remains unclear.   In this paper, we introduce AI-CAPTCHA, a unified framework that offers (i) an evaluation framework, ACEval, which includes advanced LALM- and ASR-based solvers, and (ii) a novel audio CAPTCHA approach, IllusionAudio, leveraging audio illusions. Through extensive evaluations of seven widely deployed audio CAPTCHAs, we show that most existing methods can be solved with high success rates by advanced LALMs and ASR models, exposing critical security weaknesses.   To address these vulnerabilities, we design a new audio CAPTCHA approach, IllusionAudio, which exploits perceptual illusion cues rooted in human auditory mechanisms. Extensive experiments demonstrate that our method defeats all tested LALM- and ASR-based attacks while achieving a 100% human pass rate, significantly outperforming existing audio CAPTCHA methods.

摘要: 网站广泛使用验证码来阻止机器人和垃圾邮件，因为它提出了人类容易解决但自动化程序难以解决的挑战。为了提高可访问性，音频验证码旨在补充视觉验证码。然而，音频CAPTCHA对高级大型音频语言模型（LALM）和自动语音识别（ASB）模型的稳健性仍不清楚。   在本文中，我们介绍了AI-CAPTCHA，这是一个统一的框架，它提供了（i）一个评估框架ACEval，其中包括先进的基于LALM和ASR的求解器，以及（ii）一种新的音频CAPTCHA方法IllusionAudio，利用音频错觉。通过对七种广泛部署的音频CAPTCHA的广泛评估，我们表明，大多数现有方法都可以通过先进的LALM和ASB模型以很高的成功率解决，从而暴露了关键的安全弱点。   为了解决这些漏洞，我们设计了一种新的音频验证码方法IllusionAudio，它利用植根于人类听觉机制的感知错觉线索。大量实验表明，我们的方法可以击败所有测试的基于LALM和SVR的攻击，同时实现100%的人类通过率，显着优于现有的音频验证码方法。



## **19. Evaluating Role-Consistency in LLMs for Counselor Training**

评估LLM辅导员培训的角色一致性 cs.CL

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08892v1) [paper-pdf](https://arxiv.org/pdf/2601.08892v1)

**Authors**: Eric Rudolph, Natalie Engert, Jens Albrecht

**Abstract**: The rise of online counseling services has highlighted the need for effective training methods for future counselors. This paper extends research on VirCo, a Virtual Client for Online Counseling, designed to complement traditional role-playing methods in academic training by simulating realistic client interactions. Building on previous work, we introduce a new dataset incorporating adversarial attacks to test the ability of large language models (LLMs) to maintain their assigned roles (role-consistency). The study focuses on evaluating the role consistency and coherence of the Vicuna model's responses, comparing these findings with earlier research. Additionally, we assess and compare various open-source LLMs for their performance in sustaining role consistency during virtual client interactions. Our contributions include creating an adversarial dataset, evaluating conversation coherence and persona consistency, and providing a comparative analysis of different LLMs.

摘要: 在线咨询服务的兴起凸显了未来咨询师对有效培训方法的需求。本文扩展了对VirCo的研究，VirCo是一个在线咨询虚拟客户端，旨在通过模拟现实的客户互动来补充学术培训中的传统角色扮演方法。在之前的工作的基础上，我们引入了一个包含对抗攻击的新数据集，以测试大型语言模型（LLM）维护其分配角色（角色一致性）的能力。该研究的重点是评估Vicuna模型反应的角色一致性和一致性，并将这些发现与早期研究进行比较。此外，我们还评估和比较各种开源LLM在虚拟客户端交互期间维持角色一致性方面的性能。我们的贡献包括创建对抗数据集、评估对话一致性和角色一致性，以及提供不同LLM的比较分析。



## **20. BenchOverflow: Measuring Overflow in Large Language Models via Plain-Text Prompts**

BenchOverFlow：通过纯文本预算来测量大型语言模型中的溢出 cs.CL

Accepted at TMLR 2026

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08490v1) [paper-pdf](https://arxiv.org/pdf/2601.08490v1)

**Authors**: Erin Feiglin, Nir Hutnik, Raz Lapid

**Abstract**: We investigate a failure mode of large language models (LLMs) in which plain-text prompts elicit excessive outputs, a phenomenon we term Overflow. Unlike jailbreaks or prompt injection, Overflow arises under ordinary interaction settings and can lead to elevated serving cost, latency, and cross-user performance degradation, particularly when scaled across many requests. Beyond usability, the stakes are economic and environmental: unnecessary tokens increase per-request cost and energy consumption, compounding into substantial operational spend and carbon footprint at scale. Moreover, Overflow represents a practical vector for compute amplification and service degradation in shared environments. We introduce BenchOverflow, a model-agnostic benchmark of nine plain-text prompting strategies that amplify output volume without adversarial suffixes or policy circumvention. Using a standardized protocol with a fixed budget of 5000 new tokens, we evaluate nine open- and closed-source models and observe pronounced rightward shifts and heavy tails in length distributions. Cap-saturation rates (CSR@1k/3k/5k) and empirical cumulative distribution functions (ECDFs) quantify tail risk; within-prompt variance and cross-model correlations show that Overflow is broadly reproducible yet heterogeneous across families and attack vectors. A lightweight mitigation-a fixed conciseness reminder-attenuates right tails and lowers CSR for all strategies across the majority of models. Our findings position length control as a measurable reliability, cost, and sustainability concern rather than a stylistic quirk. By enabling standardized comparison of length-control robustness across models, BenchOverflow provides a practical basis for selecting deployments that minimize resource waste and operating expense, and for evaluating defenses that curb compute amplification without eroding task performance.

摘要: 我们研究了大型语言模型（LLM）的一种失败模式，其中纯文本提示会引发过多的输出，我们将这种现象称为“溢出”。与越狱或提示注入不同，溢出在普通交互设置下发生，并可能导致服务成本、延迟和跨用户性能下降，特别是在跨多个请求扩展时。除了可用性之外，还有经济和环境方面的利害关系：不必要的代币会增加每次请求的成本和能源消耗，从而导致大量运营支出和大规模碳足迹。此外，溢出代表了共享环境中计算放大和服务降级的实用载体。我们引入BenchOverflow，这是一个与模型无关的基准测试，包含九种纯文本提示策略，可以在没有对抗性后缀或规避政策的情况下放大输出量。使用一个标准化的协议，固定预算为5000个新的令牌，我们评估了9个开源和闭源模型，并观察到明显的长度变化和重尾分布。上限饱和率（CSR@1k/3 k/5 k）和经验累积分布函数（ECDF）量化了尾部风险;即时内方差和跨模型相关性表明，溢出具有广泛的可重复性，但在家族和攻击向量之间具有异质性。一个轻量级的缓解措施-一个固定的简洁性衰减器-衰减右尾并降低大多数模型中所有策略的CSR。我们的研究结果将长度控制定位为可衡量的可靠性，成本和可持续性问题，而不是风格上的怪癖。BenchOverflow通过对不同模型的长度控制鲁棒性进行标准化比较，为选择最大限度地减少资源浪费和运营费用的部署以及评估在不影响任务性能的情况下抑制计算放大的防御措施提供了实用基础。



## **21. DNF: Dual-Layer Nested Fingerprinting for Large Language Model Intellectual Property Protection**

DNF：用于大语言模型知识产权保护的双层嵌套指纹 cs.CR

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08223v1) [paper-pdf](https://arxiv.org/pdf/2601.08223v1)

**Authors**: Zhenhua Xu, Yiran Zhao, Mengting Zhong, Dezhang Kong, Changting Lin, Tong Qiao, Meng Han

**Abstract**: The rapid growth of large language models raises pressing concerns about intellectual property protection under black-box deployment. Existing backdoor-based fingerprints either rely on rare tokens -- leading to high-perplexity inputs susceptible to filtering -- or use fixed trigger-response mappings that are brittle to leakage and post-hoc adaptation. We propose \textsc{Dual-Layer Nested Fingerprinting} (DNF), a black-box method that embeds a hierarchical backdoor by coupling domain-specific stylistic cues with implicit semantic triggers. Across Mistral-7B, LLaMA-3-8B-Instruct, and Falcon3-7B-Instruct, DNF achieves perfect fingerprint activation while preserving downstream utility. Compared with existing methods, it uses lower-perplexity triggers, remains undetectable under fingerprint detection attacks, and is relatively robust to incremental fine-tuning and model merging. These results position DNF as a practical, stealthy, and resilient solution for LLM ownership verification and intellectual property protection.

摘要: 大型语言模型的快速增长引发了人们对黑匣子部署下知识产权保护的紧迫担忧。现有的基于后门的指纹要么依赖于罕见的令牌（导致容易受到过滤的高困惑度输入），要么使用容易泄露和事后适应的固定的响应者映射。我们提出了\textsk {双层嵌套指纹识别}（DNF），这是一种黑匣子方法，通过将特定领域的风格线索与隐式语义触发器相结合来嵌入分层后门。在Mistral-7 B、LLaMA-3- 8B-Direct和Falcon 3 - 7 B-Direct中，DNF实现了完美的指纹激活，同时保留了下游实用性。与现有方法相比，它使用较低的触发器，在指纹检测攻击下仍然不可检测，并且对增量微调和模型合并相对鲁棒。这些结果使DNF成为LLM所有权验证和知识产权保护的实用、隐蔽和弹性解决方案。



## **22. Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety**

与法规一起对先例进行推理：案例增强的深思熟虑调整以确保LLM安全 cs.AI

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.08000v1) [paper-pdf](https://arxiv.org/pdf/2601.08000v1)

**Authors**: Can Jin, Rui Wu, Tong Che, Qixin Zhang, Hongwu Peng, Jiahui Zhao, Zhenting Wang, Wenqi Wei, Ligong Han, Zhao Zhang, Yuan Cao, Ruixiang Tang, Dimitris N. Metaxas

**Abstract**: Ensuring that Large Language Models (LLMs) adhere to safety principles without refusing benign requests remains a significant challenge. While OpenAI introduces deliberative alignment (DA) to enhance the safety of its o-series models through reasoning over detailed ``code-like'' safety rules, the effectiveness of this approach in open-source LLMs, which typically lack advanced reasoning capabilities, is understudied. In this work, we systematically evaluate the impact of explicitly specifying extensive safety codes versus demonstrating them through illustrative cases. We find that referencing explicit codes inconsistently improves harmlessness and systematically degrades helpfulness, whereas training on case-augmented simple codes yields more robust and generalized safety behaviors. By guiding LLMs with case-augmented reasoning instead of extensive code-like safety rules, we avoid rigid adherence to narrowly enumerated rules and enable broader adaptability. Building on these insights, we propose CADA, a case-augmented deliberative alignment method for LLMs utilizing reinforcement learning on self-generated safety reasoning chains. CADA effectively enhances harmlessness, improves robustness against attacks, and reduces over-refusal while preserving utility across diverse benchmarks, offering a practical alternative to rule-only DA for improving safety while maintaining helpfulness.

摘要: 确保大型语言模型（LLM）遵守安全原则而不拒绝善意请求仍然是一个重大挑战。虽然OpenAI引入了审慎对齐（DA），通过推理详细的“类似代码”安全规则来增强其o系列模型的安全性，但这种方法在通常缺乏高级推理能力的开源LLM中的有效性研究不足。在这项工作中，我们系统地评估了明确规定广泛的安全规范与通过说明性案例展示它们的影响。我们发现，不一致地引用显式代码会改善无害性并系统性地降低帮助性，而对案例增强的简单代码进行训练会产生更稳健和更普遍的安全行为。通过使用案例增强推理而不是广泛的类似代码的安全规则来指导LLM，我们可以避免严格遵守狭隘列举的规则，并实现更广泛的适应性。在这些见解的基础上，我们提出了CADA，这是一种针对LLM的案例增强审慎对齐方法，利用自生成的安全推理链上的强化学习。CADA有效地增强了无害性，提高了针对攻击的鲁棒性，并减少了过度拒绝，同时保留了不同基准的实用性，为纯规则DA提供了一种实用的替代方案，以提高安全性，同时保持有用性。



## **23. SecureCAI: Injection-Resilient LLM Assistants for Cybersecurity Operations**

SecureCAE：具有注射弹性的网络安全运营法学硕士助理 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07835v1) [paper-pdf](https://arxiv.org/pdf/2601.07835v1)

**Authors**: Mohammed Himayath Ali, Mohammed Aqib Abdullah, Mohammed Mudassir Uddin, Shahnawaz Alam

**Abstract**: Large Language Models have emerged as transformative tools for Security Operations Centers, enabling automated log analysis, phishing triage, and malware explanation; however, deployment in adversarial cybersecurity environments exposes critical vulnerabilities to prompt injection attacks where malicious instructions embedded in security artifacts manipulate model behavior. This paper introduces SecureCAI, a novel defense framework extending Constitutional AI principles with security-aware guardrails, adaptive constitution evolution, and Direct Preference Optimization for unlearning unsafe response patterns, addressing the unique challenges of high-stakes security contexts where traditional safety mechanisms prove insufficient against sophisticated adversarial manipulation. Experimental evaluation demonstrates that SecureCAI reduces attack success rates by 94.7% compared to baseline models while maintaining 95.1% accuracy on benign security analysis tasks, with the framework incorporating continuous red-teaming feedback loops enabling dynamic adaptation to emerging attack strategies and achieving constitution adherence scores exceeding 0.92 under sustained adversarial pressure, thereby establishing a foundation for trustworthy integration of language model capabilities into operational cybersecurity workflows and addressing a critical gap in current approaches to AI safety within adversarial domains.

摘要: 大型语言模型已成为安全运营中心的变革性工具，可以实现自动化日志分析、网络钓鱼分类和恶意软件解释;然而，在对抗性网络安全环境中的部署暴露了关键漏洞，从而引发注入攻击，其中嵌入安全制品中的恶意指令操纵模型行为。本文介绍了SecureCAE，这是一种新型防御框架，通过安全感知护栏、自适应宪法进化和直接偏好优化来扩展宪法人工智能原则，用于消除不安全的响应模式，解决高风险安全环境中传统安全机制不足以应对复杂的对抗性操纵的独特挑战。实验评估表明，与基线模型相比，SecureCAE将攻击成功率降低了94.7%，同时在良性安全分析任务上保持了95.1%的准确性，该框架结合了连续的红色团队反馈循环，能够动态适应新兴的攻击策略，并在持续的对抗压力下实现宪法遵守分数超过0.92。从而为将语言模型能力可信地集成到运营网络安全工作流程中奠定基础，并解决当前对抗领域人工智能安全方法中的关键差距。



## **24. A Visual Semantic Adaptive Watermark grounded by Prefix-Tuning for Large Vision-Language Model**

基于大型视觉语言模型的前置调整的视觉语义自适应水印 cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07291v1) [paper-pdf](https://arxiv.org/pdf/2601.07291v1)

**Authors**: Qi Zheng, Shuliang Liu, Yu Huang, Sihang Jia, Jungang Li, Lyuhao Chen, Junhao Chen, Hanqian Li, Aiwei Liu, Yibo Yan, Xuming Hu

**Abstract**: Watermarking has emerged as a pivotal solution for content traceability and intellectual property protection in Large Vision-Language Models (LVLMs). However, vision-agnostic watermarks introduce visually irrelevant tokens and disrupt visual grounding by enforcing indiscriminate pseudo-random biases, while some semantic-aware methods incur prohibitive inference latency due to rejection sampling. In this paper, we propose the VIsual Semantic Adaptive Watermark (VISA-Mark), a novel framework that embeds detectable signals while strictly preserving visual fidelity. Our approach employs a lightweight, efficiently trained prefix-tuner to extract dynamic Visual-Evidence Weights, which quantify the evidentiary support for candidate tokens based on the visual input. These weights guide an adaptive vocabulary partitioning and logits perturbation mechanism, concentrating watermark strength specifically on visually-supported tokens. By actively aligning the watermark with visual evidence, VISA-Mark effectively maintains visual fidelity. Empirical results confirm that VISA-Mark outperforms conventional methods with a 7.8% improvement in visual consistency (Chair-I) and superior semantic fidelity. The framework maintains highly competitive detection accuracy (96.88% AUC) and robust attack resilience (99.3%) without sacrificing inference efficiency, effectively establishing a new standard for reliability-preserving multimodal watermarking.

摘要: 水印已成为大型视觉语言模型（LVLM）中内容可追溯性和知识产权保护的关键解决方案。然而，视觉不可知的水印引入了视觉上不相关的标记，并通过强制实施不加区别的伪随机偏差来破坏视觉基础，而一些语义感知方法则会因拒绝抽样而招致令人望而却步的推理延迟。在本文中，我们提出了视觉语义自适应水印（VISA-Mark），这是一种新颖的框架，可以嵌入可检测信号，同时严格保留视觉保真度。我们的方法采用轻量级、经过高效训练的前置调整器来提取动态视觉证据权重，该权重根据视觉输入量化候选令牌的证据支持。这些权重引导自适应词汇划分和logits扰动机制，将水印强度专门集中在视觉支持的令牌上。通过主动将水印与视觉证据对齐，VISA-Mark有效地保持了视觉保真度。实证结果证实，VISA-Mark优于传统方法，在视觉一致性（Chair-I）和优越的语义保真度方面提高了7.8%。该框架在不牺牲推理效率的情况下保持了极具竞争力的检测准确率（96.88%AUC）和强大的攻击弹性（99.3%），有效地建立了可靠性保护的多模态水印的新标准。



## **25. When Bots Take the Bait: Exposing and Mitigating the Emerging Social Engineering Attack in Web Automation Agent**

当机器人上钩时：揭露和缓解Web自动化代理中新兴的社会工程攻击 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07263v1) [paper-pdf](https://arxiv.org/pdf/2601.07263v1)

**Authors**: Xinyi Wu, Geng Hong, Yueyue Chen, MingXuan Liu, Feier Jin, Xudong Pan, Jiarun Dai, Baojun Liu

**Abstract**: Web agents, powered by large language models (LLMs), are increasingly deployed to automate complex web interactions. The rise of open-source frameworks (e.g., Browser Use, Skyvern-AI) has accelerated adoption, but also broadened the attack surface. While prior research has focused on model threats such as prompt injection and backdoors, the risks of social engineering remain largely unexplored. We present the first systematic study of social engineering attacks against web automation agents and design a pluggable runtime mitigation solution. On the attack side, we introduce the AgentBait paradigm, which exploits intrinsic weaknesses in agent execution: inducement contexts can distort the agent's reasoning and steer it toward malicious objectives misaligned with the intended task. On the defense side, we propose SUPERVISOR, a lightweight runtime module that enforces environment and intention consistency alignment between webpage context and intended goals to mitigate unsafe operations before execution.   Empirical results show that mainstream frameworks are highly vulnerable to AgentBait, with an average attack success rate of 67.5% and peaks above 80% under specific strategies (e.g., trusted identity forgery). Compared with existing lightweight defenses, our module can be seamlessly integrated across different web automation frameworks and reduces attack success rates by up to 78.1% on average while incurring only a 7.7% runtime overhead and preserving usability. This work reveals AgentBait as a critical new threat surface for web agents and establishes a practical, generalizable defense, advancing the security of this rapidly emerging ecosystem. We reported the details of this attack to the framework developers and received acknowledgment before submission.

摘要: 由大型语言模型（LLM）支持的Web代理越来越多地被部署来自动化复杂的Web交互。开源框架的兴起（例如，浏览器使用，Skyvern-AI）加速了采用，但也拓宽了攻击面。虽然之前的研究集中在即时注入和后门等模型威胁上，但社会工程的风险在很大程度上仍未被探索。我们首次对针对Web自动化代理的社会工程攻击进行了系统性研究，并设计了可插入的运行时缓解解决方案。在攻击方面，我们引入了AgentBait范式，该范式利用了代理执行中的内在弱点：诱导上下文可能会扭曲代理的推理，并将其引导到与预期任务不一致的恶意目标。在防御方面，我们提出了SUPERVISOR，这是一个轻量级运行时模块，它强制网页上下文和预期目标之间的环境和意图一致性一致性，以在执行前减轻不安全操作。   经验结果表明，主流框架极易受到AgentBait的影响，平均攻击成功率为67.5%，在特定策略下（例如，可信身份伪造）。与现有的轻量级防御相比，我们的模块可以在不同的Web自动化框架之间无缝集成，平均可将攻击成功率降低高达78.1%，同时仅产生7.7%的运行时开销并保持可用性。这项工作揭示了AgentBait作为Web代理的一个关键的新威胁表面，并建立了一个实用的，可推广的防御，提高了这个迅速崛起的生态系统的安全性。我们向框架开发人员报告了这次攻击的细节，并在提交之前收到了确认。



## **26. Defenses Against Prompt Attacks Learn Surface Heuristics**

防御即时攻击学习表面启发法 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07185v1) [paper-pdf](https://arxiv.org/pdf/2601.07185v1)

**Authors**: Shawn Li, Chenxiao Yu, Zhiyu Ni, Hao Li, Charith Peris, Chaowei Xiao, Yue Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in security-sensitive applications, where they must follow system- or developer-specified instructions that define the intended task behavior, while completing benign user requests. When adversarial instructions appear in user queries or externally retrieved content, models may override intended logic. Recent defenses rely on supervised fine-tuning with benign and malicious labels. Although these methods achieve high attack rejection rates, we find that they rely on narrow correlations in defense data rather than harmful intent, leading to systematic rejection of safe inputs. We analyze three recurring shortcut behaviors induced by defense fine-tuning. \emph{Position bias} arises when benign content placed later in a prompt is rejected at much higher rates; across reasoning benchmarks, suffix-task rejection rises from below \textbf{10\%} to as high as \textbf{90\%}. \emph{Token trigger bias} occurs when strings common in attack data raise rejection probability even in benign contexts; inserting a single trigger token increases false refusals by up to \textbf{50\%}. \emph{Topic generalization bias} reflects poor generalization beyond the defense data distribution, with defended models suffering test-time accuracy drops of up to \textbf{40\%}. These findings suggest that current prompt-injection defenses frequently respond to attack-like surface patterns rather than the underlying intent. We introduce controlled diagnostic datasets and a systematic evaluation across two base models and multiple defense pipelines, highlighting limitations of supervised fine-tuning for reliable LLM security.

摘要: 大型语言模型（LLM）越来越多地部署在安全敏感的应用程序中，它们必须遵循系统或开发人员指定的指令，这些指令定义了预期的任务行为，同时完成良性的用户请求。当对抗性指令出现在用户查询或外部检索的内容中时，模型可能会覆盖预期的逻辑。最近的防御依赖于良性和恶意标签的监督微调。虽然这些方法实现了高攻击拒绝率，但我们发现它们依赖于防御数据中的窄相关性，而不是有害的意图，从而导致系统拒绝安全输入。我们分析了防御微调引发的三种反复出现的捷径行为。\{位置偏差}当提示中稍后放置的良性内容被拒绝率高得多时，就会出现;在推理基准中，后缀任务拒绝率从低于\textBF{10\%}上升到高达\textBF{90\%}。\当攻击数据中常见的字符串即使在良性上下文中也会提高拒绝概率时，就会发生{Token触发偏差};插入单个触发令牌会增加错误拒绝最多\textBF{50\%}。\{主题概括偏差}反映了防御数据分布之外的较差概括，防御模型的测试时准确性下降高达\textBF{40\%}。这些发现表明，当前的预算注射防御经常对类似攻击的表面模式做出反应，而不是潜在意图。我们引入了受控诊断数据集和跨两个基本模型和多个防御管道的系统评估，强调了监督式微调以实现可靠的LLM安全性的局限性。



## **27. Safe-FedLLM: Delving into the Safety of Federated Large Language Models**

Safe-FedLLM：深入研究联邦大型语言模型的安全性 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07177v1) [paper-pdf](https://arxiv.org/pdf/2601.07177v1)

**Authors**: Mingxiang Tao, Yu Tian, Wenxuan Tu, Yue Yang, Xue Yang, Xiangyan Tang

**Abstract**: Federated learning (FL) addresses data privacy and silo issues in large language models (LLMs). Most prior work focuses on improving the training efficiency of federated LLMs. However, security in open environments is overlooked, particularly defenses against malicious clients. To investigate the safety of LLMs during FL, we conduct preliminary experiments to analyze potential attack surfaces and defensible characteristics from the perspective of Low-Rank Adaptation (LoRA) weights. We find two key properties of FL: 1) LLMs are vulnerable to attacks from malicious clients in FL, and 2) LoRA weights exhibit distinct behavioral patterns that can be filtered through simple classifiers. Based on these properties, we propose Safe-FedLLM, a probe-based defense framework for federated LLMs, constructing defenses across three dimensions: Step-Level, Client-Level, and Shadow-Level. The core concept of Safe-FedLLM is to perform probe-based discrimination on the LoRA weights locally trained by each client during FL, treating them as high-dimensional behavioral features and using lightweight classification models to determine whether they possess malicious attributes. Extensive experiments demonstrate that Safe-FedLLM effectively enhances the defense capability of federated LLMs without compromising performance on benign data. Notably, our method effectively suppresses malicious data impact without significant impact on training speed, and remains effective even with many malicious clients. Our code is available at: https://github.com/dmqx/Safe-FedLLM.

摘要: 联合学习（FL）解决大型语言模型（LLM）中的数据隐私和筒仓问题。之前的大多数工作都集中在提高联邦LLM的培训效率。然而，开放环境中的安全性被忽视了，特别是针对恶意客户端的防御。为了研究FL期间LLM的安全性，我们进行了初步实验，从低等级适应（LoRA）权重的角度分析潜在的攻击面和防御特征。我们发现FL的两个关键属性：1）LLM容易受到FL中恶意客户端的攻击，2）LoRA权重表现出可以通过简单分类器过滤的独特行为模式。基于这些属性，我们提出了Safe-FedLLM，这是一个针对联邦LLM的基于探针的防御框架，跨越三个维度构建防御：步骤级、客户端级和影子级。Safe-FedLLM的核心理念是对FL期间每个客户端本地训练的LoRA权重进行基于探针的辨别，将其视为多维行为特征，并使用轻量级分类模型来确定其是否具有恶意属性。大量实验表明，Safe-FedLLM有效增强了联邦LLM的防御能力，而不会损害良性数据的性能。值得注意的是，我们的方法有效地抑制了恶意数据的影响，而不会对训练速度产生重大影响，即使有许多恶意客户端也仍然有效。我们的代码可访问：https://github.com/dmqx/Safe-FedLLM。



## **28. Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework**

通过强大的LLM授权的多Agent强化学习框架增强云网络弹性 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07122v1) [paper-pdf](https://arxiv.org/pdf/2601.07122v1)

**Authors**: Yixiao Peng, Hao Hu, Feiyang Li, Xinye Cao, Yingchang Jiang, Jipeng Tang, Guoshun Nan, Yuling Liu

**Abstract**: While virtualization and resource pooling empower cloud networks with structural flexibility and elastic scalability, they inevitably expand the attack surface and challenge cyber resilience. Reinforcement Learning (RL)-based defense strategies have been developed to optimize resource deployment and isolation policies under adversarial conditions, aiming to enhance system resilience by maintaining and restoring network availability. However, existing approaches lack robustness as they require retraining to adapt to dynamic changes in network structure, node scale, attack strategies, and attack intensity. Furthermore, the lack of Human-in-the-Loop (HITL) support limits interpretability and flexibility. To address these limitations, we propose CyberOps-Bots, a hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs). Inspired by MITRE ATT&CK's Tactics-Techniques model, CyberOps-Bots features a two-layer architecture: (1) An upper-level LLM agent with four modules--ReAct planning, IPDRR-based perception, long-short term memory, and action/tool integration--performs global awareness, human intent recognition, and tactical planning; (2) Lower-level RL agents, developed via heterogeneous separated pre-training, execute atomic defense actions within localized network regions. This synergy preserves LLM adaptability and interpretability while ensuring reliable RL execution. Experiments on real cloud datasets show that, compared to state-of-the-art algorithms, CyberOps-Bots maintains network availability 68.5% higher and achieves a 34.7% jumpstart performance gain when shifting the scenarios without retraining. To our knowledge, this is the first study to establish a robust LLM-RL framework with HITL support for cloud defense. We will release our framework to the community, facilitating the advancement of robust and autonomous defense in cloud networks.

摘要: 虽然虚拟化和资源池赋予云网络结构灵活性和弹性可扩展性，但它们不可避免地扩大了攻击面并挑战网络弹性。基于强化学习（RL）的防御策略被开发出来，以优化对抗条件下的资源部署和隔离政策，旨在通过维护和恢复网络可用性来增强系统的弹性。然而，现有的方法缺乏鲁棒性，因为它们需要重新训练以适应网络结构、节点规模、攻击策略和攻击强度的动态变化。此外，缺乏人在环（HITL）支持限制了可解释性和灵活性。为了解决这些限制，我们提出了CyberOps-Bots，这是一个由大型语言模型（LLM）支持的分层多智能体强化学习框架。CyberOps-Bots受到MITRE ATA & CK的Tactics-Techniques模型的启发，具有两层架构：（1）上层LLM代理，具有四个模块--ReAct规划、基于IPDRR的感知、长短期记忆和动作/工具集成--执行全球感知、人类意图识别和战术规划;（2）通过异类分离预训练开发的低级RL代理，在局部网络区域内执行原子防御动作。这种协同作用保留了LLM的适应性和可解释性，同时确保可靠的RL执行。对真实云数据集的实验表明，与最先进的算法相比，CyberOps-Bots的网络可用性提高了68.5%，并且在无需重新训练的情况下改变场景时实现了34.7%的启动性能提升。据我们所知，这是第一项建立具有HITL支持的强大LLM-RL框架的研究。我们将向社区发布我们的框架，促进云网络中稳健和自主防御的发展。



## **29. Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems**

克服检索障碍：LLM系统的野外间接即时注入 cs.CR

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.07072v1) [paper-pdf](https://arxiv.org/pdf/2601.07072v1)

**Authors**: Hongyan Chang, Ergute Bao, Xinjian Luo, Ting Yu

**Abstract**: Large language models (LLMs) increasingly rely on retrieving information from external corpora. This creates a new attack surface: indirect prompt injection (IPI), where hidden instructions are planted in the corpora and hijack model behavior once retrieved. Previous studies have highlighted this risk but often avoid the hardest step: ensuring that malicious content is actually retrieved. In practice, unoptimized IPI is rarely retrieved under natural queries, which leaves its real-world impact unclear.   We address this challenge by decomposing the malicious content into a trigger fragment that guarantees retrieval and an attack fragment that encodes arbitrary attack objectives. Based on this idea, we design an efficient and effective black-box attack algorithm that constructs a compact trigger fragment to guarantee retrieval for any attack fragment. Our attack requires only API access to embedding models, is cost-efficient (as little as $0.21 per target user query on OpenAI's embedding models), and achieves near-100% retrieval across 11 benchmarks and 8 embedding models (including both open-source models and proprietary services).   Based on this attack, we present the first end-to-end IPI exploits under natural queries and realistic external corpora, spanning both RAG and agentic systems with diverse attack objectives. These results establish IPI as a practical and severe threat: when a user issued a natural query to summarize emails on frequently asked topics, a single poisoned email was sufficient to coerce GPT-4o into exfiltrating SSH keys with over 80% success in a multi-agent workflow. We further evaluate several defenses and find that they are insufficient to prevent the retrieval of malicious text, highlighting retrieval as a critical open vulnerability.

摘要: 大型语言模型（LLM）越来越依赖于从外部库中检索信息。这创建了一个新的攻击表面：间接提示注入（IPI），其中隐藏指令被植入到库中，并在检索到后劫持模型行为。之前的研究强调了这种风险，但通常避免了最困难的步骤：确保恶意内容被真正检索到。在实践中，未经优化的IPI很少在自然查询下被检索，这使得其现实世界的影响不清楚。   我们通过将恶意内容分解为保证检索的触发片段和编码任意攻击目标的攻击片段来解决这一挑战。基于这个想法，我们设计了一种高效有效的黑匣子攻击算法，该算法构造了一个紧凑的触发片段，以保证对任何攻击片段的检索。我们的攻击仅需要API访问嵌入模型，具有成本效益（OpenAI嵌入模型上的每个目标用户查询低至0.21美元），并且在11个基准测试和8个嵌入模型（包括开源模型和专有服务）中实现了近100%的检索。   基于这次攻击，我们在自然查询和现实外部库下展示了第一个端到端IPI漏洞，跨越RAG和具有不同攻击目标的代理系统。这些结果将IPI确立为一种实际且严重的威胁：当用户发出自然查询来总结有关常见主题的电子邮件时，一封有毒电子邮件足以迫使GPT-4 o提取出SSH密钥，在多代理工作流程中成功率超过80%。我们进一步评估了几种防御措施，发现它们不足以阻止恶意文本的检索，并强调检索是一个关键的开放漏洞。



## **30. PenForge: On-the-Fly Expert Agent Construction for Automated Penetration Testing**

PenForge：用于自动渗透测试的实时专家代理构建 cs.SE

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.06910v1) [paper-pdf](https://arxiv.org/pdf/2601.06910v1)

**Authors**: Huihui Huang, Jieke Shi, Junkai Chen, Ting Zhang, Yikun Li, Chengran Yang, Eng Lieh Ouh, Lwin Khin Shar, David Lo

**Abstract**: Penetration testing is essential for identifying vulnerabilities in web applications before real adversaries can exploit them. Recent work has explored automating this process with Large Language Model (LLM)-powered agents, but existing approaches either rely on a single generic agent that struggles in complex scenarios or narrowly specialized agents that cannot adapt to diverse vulnerability types. We therefore introduce PenForge, a framework that dynamically constructs expert agents during testing rather than relying on those prepared beforehand. By integrating automated reconnaissance of potential attack surfaces with agents instantiated on the fly for context-aware exploitation, PenForge achieves a 30.0% exploit success rate (12/40) on CVE-Bench in the particularly challenging zero-day setting, which is a 3 times improvement over the state-of-the-art. Our analysis also identifies three opportunities for future work: (1) supplying richer tool-usage knowledge to improve exploitation effectiveness; (2) extending benchmarks to include more vulnerabilities and attack types; and (3) fostering developer trust by incorporating explainable mechanisms and human review. As an emerging result with substantial potential impact, PenForge embodies the early-stage yet paradigm-shifting idea of on-the-fly agent construction, marking its promise as a step toward scalable and effective LLM-driven penetration testing.

摘要: 渗透测试对于在真正的对手利用Web应用程序中的漏洞之前识别它们至关重要。最近的工作探索了使用大型语言模型（LLM）驱动的代理自动化这一过程，但现有方法要么依赖于在复杂场景中挣扎的单个通用代理，要么依赖于无法适应不同漏洞类型的狭隘专业代理。因此，我们引入了PenForge，这是一个在测试期间动态构建专家代理的框架，而不是依赖于预先准备的代理。通过将对潜在攻击面的自动侦察与动态实例化以进行上下文感知利用的代理集成，PenForge在特别具有挑战性的零日环境中在CVE-Bench上实现了30.0%的利用成功率（12/40），这比最先进技术提高了3倍。我们的分析还确定了未来工作的三个机会：（1）提供更丰富的工具使用知识以提高利用效率;（2）扩展基准以包括更多漏洞和攻击类型;以及（3）通过结合可解释的机制和人类审查来促进开发人员的信任。作为一项具有重大潜在影响的新兴成果，PenForge体现了动态代理构建的早期但范式转变理念，标志着其有望成为迈向可扩展和有效的LLM驱动渗透测试的一步。



## **31. Paraphrasing Adversarial Attack on LLM-as-a-Reviewer**

解释对LLM作为评审员的对抗攻击 cs.CL

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.06884v1) [paper-pdf](https://arxiv.org/pdf/2601.06884v1)

**Authors**: Masahiro Kaneko

**Abstract**: The use of large language models (LLMs) in peer review systems has attracted growing attention, making it essential to examine their potential vulnerabilities. Prior attacks rely on prompt injection, which alters manuscript content and conflates injection susceptibility with evaluation robustness. We propose the Paraphrasing Adversarial Attack (PAA), a black-box optimization method that searches for paraphrased sequences yielding higher review scores while preserving semantic equivalence and linguistic naturalness. PAA leverages in-context learning, using previous paraphrases and their scores to guide candidate generation. Experiments across five ML and NLP conferences with three LLM reviewers and five attacking models show that PAA consistently increases review scores without changing the paper's claims. Human evaluation confirms that generated paraphrases maintain meaning and naturalness. We also find that attacked papers exhibit increased perplexity in reviews, offering a potential detection signal, and that paraphrasing submissions can partially mitigate attacks.

摘要: 同行评审系统中使用大型语言模型（LLM）引起了越来越多的关注，因此检查其潜在的漏洞至关重要。之前的攻击依赖于即时注入，这会改变手稿内容并将注入敏感性与评估稳健性混为一谈。我们提出了曲解对抗攻击（PPA），这是一种黑匣子优化方法，可以搜索曲解后的序列，以产生更高的评论分数，同时保持语义等效性和语言自然性。PPA利用上下文学习，使用之前的解释及其分数来指导候选人的生成。在五次ML和NLP会议上进行的实验显示，由三名LLM评审员和五个攻击模型组成，PPA在不改变论文主张的情况下持续提高评审分数。人类评估证实，生成的重述保持了意义和自然性。我们还发现，受攻击的论文在评论中表现出更多的困惑，提供了潜在的检测信号，并且重述提交的内容可以部分减轻攻击。



## **32. CHASE: LLM Agents for Dissecting Malicious PyPI Packages**

CHASE：剖析恶意PyPI包的LLM代理 cs.CR

Accepted for publication and presented at the 2nd IEEE International Conference on AI-powered Software (AIware 2025). 10 pages, 3 figures

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.06838v1) [paper-pdf](https://arxiv.org/pdf/2601.06838v1)

**Authors**: Takaaki Toda, Tatsuya Mori

**Abstract**: Modern software package registries like PyPI have become critical infrastructure for software development, but are increasingly exploited by threat actors distributing malicious packages with sophisticated multi-stage attack chains. While Large Language Models (LLMs) offer promising capabilities for automated code analysis, their application to security-critical malware detection faces fundamental challenges, including hallucination and context confusion, which can lead to missed detections or false alarms. We present CHASE (Collaborative Hierarchical Agents for Security Exploration), a high-reliability multi-agent architecture that addresses these limitations through a Plan-and-Execute coordination model, specialized Worker Agents focused on specific analysis aspects, and integration with deterministic security tools for critical operations. Our key insight is that reliability in LLM-based security analysis emerges not from improving individual model capabilities but from architecting systems that compensate for LLM weaknesses while leveraging their semantic understanding strengths. Evaluation on a dataset of 3,000 packages (500 malicious, 2,500 benign) demonstrates that CHASE achieves 98.4% recall with only 0.08% false positive rate, while maintaining a practical median analysis time of 4.5 minutes per package, making it suitable for operational deployment in automated package screening. Furthermore, we conducted a survey with cybersecurity professionals to evaluate the generated analysis reports, identifying their key strengths and areas for improvement. This work provides a blueprint for building reliable AI-powered security tools that can scale with the growing complexity of modern software supply chains. Our project page is available at https://t0d4.github.io/CHASE-AIware25/

摘要: PyPI等现代软件包注册表已成为软件开发的重要基础设施，但越来越多地被威胁行为者利用复杂的多阶段攻击链分发恶意包。虽然大型语言模型（LLM）为自动代码分析提供了前景光明的功能，但其应用于安全关键恶意软件检测面临着根本性挑战，包括幻觉和上下文混乱，这可能会导致漏报或误报。我们介绍了CHASE（用于安全探索的协作分层代理），这是一种高可靠性多代理架构，通过计划和执行协调模型、专注于特定分析方面的专业工作者代理以及与关键操作的确定性安全工具集成来解决这些限制。我们的主要见解是，基于LLM的安全分析的可靠性不是来自提高单个模型能力，而是来自构建能够弥补LLM弱点、同时利用其语义理解优势的系统。对3，000个包裹（500个恶意包裹，2，500个良性包裹）的数据集的评估表明，CHASE的召回率达到了98.4%，假阳性率仅为0.08%，同时每个包裹的实际分析时间中位数保持在4.5分钟，适合自动化包裹筛查中的操作部署。此外，我们还对网络安全专业人员进行了一项调查，以评估生成的分析报告，确定其主要优势和需要改进的领域。这项工作为构建可靠的人工智能驱动的安全工具提供了蓝图，这些工具可以随着现代软件供应链日益增长的复杂性进行扩展。我们的项目页面可访问https://t0d4.github.io/CHASE-AIware25/



## **33. CyberLLM-FINDS 2025: Instruction-Tuned Fine-tuning of Domain-Specific LLMs with Retrieval-Augmented Generation and Graph Integration for MITRE Evaluation**

CyberLLM-Finds 2025：具有检索增强生成和图形集成的领域特定LLM的指令调整微调，以实现MITRE评估 cs.CR

12 pages

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.06779v1) [paper-pdf](https://arxiv.org/pdf/2601.06779v1)

**Authors**: Vasanth Iyer, Leonardo Bobadilla, S. S. Iyengar

**Abstract**: Large Language Models (LLMs) such as Gemma-2B have shown strong performance in various natural language processing tasks. However, general-purpose models often lack the domain expertise required for cybersecurity applications. This work presents a methodology to fine-tune the Gemma-2B model into a domain-specific cybersecurity LLM. We detail the processes of dataset preparation, fine-tuning, and synthetic data generation, along with implications for real-world applications in threat detection, forensic investigation, and attack analysis.   Experiments highlight challenges in prompt length distribution during domain-specific fine-tuning. Uneven prompt lengths limit the model's effective use of the context window, constraining local inference to 200-400 tokens despite hardware support for longer sequences. Chain-of-thought styled prompts, paired with quantized weights, yielded the best performance under these constraints. To address context limitations, we employed a hybrid strategy using cloud LLMs for synthetic data generation and local fine-tuning for deployment efficiency.   To extend the evaluation, we introduce a Retrieval-Augmented Generation (RAG) pipeline and graph-based reasoning framework. This approach enables structured alignment with MITRE ATT&CK techniques through STIX-based threat intelligence, enhancing recall in multi-hop and long-context scenarios. Graph modules encode entity-neighborhood context and tactic chains, helping mitigate the constraints of short prompt windows. Results demonstrate improved model alignment with tactic, technique, and procedure (TTP) coverage, validating the utility of graph-augmented LLMs in cybersecurity threat intelligence applications.

摘要: Gemma-2B等大型语言模型（LLM）在各种自然语言处理任务中表现出出色的性能。然而，通用模型通常缺乏网络安全应用所需的领域专业知识。这项工作提出了一种将Gemma-2B模型微调为特定领域的网络安全LLM的方法。我们详细介绍了数据集准备、微调和合成数据生成的过程，以及对威胁检测、法医调查和攻击分析等现实世界应用的影响。   实验凸显了特定领域微调期间即时长度分布的挑战。不均匀的提示长度限制了模型对上下文窗口的有效使用，尽管硬件支持更长的序列，但仍将局部推断限制在200-400个令牌。思想链风格的提示与量化权重相结合，在这些限制下产生了最佳性能。为了解决上下文限制，我们采用了混合策略，使用云LLM来生成合成数据并进行本地微调以提高部署效率。   为了扩展评估，我们引入了检索增强生成（RAG）管道和基于图形的推理框架。这种方法通过基于STIX的威胁情报实现与MITRE ATA & CK技术的结构化一致，增强多跳和长上下文场景中的召回。图形模块对实体邻居上下文和策略链进行编码，有助于减轻短提示窗口的限制。结果表明，模型与策略、技术和程序（TTP）覆盖范围的改进一致性，验证了图形增强的LLM在网络安全威胁情报应用中的实用性。



## **34. Memory Poisoning Attack and Defense on Memory Based LLM-Agents**

基于内存的LLM-Agents的内存中毒攻击与防御 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.05504v2) [paper-pdf](https://arxiv.org/pdf/2601.05504v2)

**Authors**: Balachandra Devarangadi Sunil, Isheeta Sinha, Piyush Maheshwari, Shantanu Todmal, Shreyan Mallik, Shuchi Mishra

**Abstract**: Large language model agents equipped with persistent memory are vulnerable to memory poisoning attacks, where adversaries inject malicious instructions through query only interactions that corrupt the agents long term memory and influence future responses. Recent work demonstrated that the MINJA (Memory Injection Attack) achieves over 95 % injection success rate and 70 % attack success rate under idealized conditions. However, the robustness of these attacks in realistic deployments and effective defensive mechanisms remain understudied. This work addresses these gaps through systematic empirical evaluation of memory poisoning attacks and defenses in Electronic Health Record (EHR) agents. We investigate attack robustness by varying three critical dimensions: initial memory state, number of indication prompts, and retrieval parameters. Our experiments on GPT-4o-mini, Gemini-2.0-Flash and Llama-3.1-8B-Instruct models using MIMIC-III clinical data reveal that realistic conditions with pre-existing legitimate memories dramatically reduce attack effectiveness. We then propose and evaluate two novel defense mechanisms: (1) Input/Output Moderation using composite trust scoring across multiple orthogonal signals, and (2) Memory Sanitization with trust-aware retrieval employing temporal decay and pattern-based filtering. Our defense evaluation reveals that effective memory sanitization requires careful trust threshold calibration to prevent both overly conservative rejection (blocking all entries) and insufficient filtering (missing subtle attacks), establishing important baselines for future adaptive defense mechanisms. These findings provide crucial insights for securing memory-augmented LLM agents in production environments.

摘要: 配备持久内存的大型语言模型代理很容易受到内存中毒攻击，对手通过仅查询的交互注入恶意指令，从而破坏代理的长期内存并影响未来的响应。最近的工作表明，MINJA（内存注入攻击）在理想化条件下实现了超过95%的注入成功率和70%的攻击成功率。然而，这些攻击在现实部署中的稳健性和有效的防御机制仍然没有得到充分的研究。这项工作通过对电子健康记录（EHR）代理中的记忆中毒攻击和防御的系统性实证评估来解决这些差距。我们通过改变三个关键维度来研究攻击的稳健性：初始存储状态、指示提示数量和检索参数。我们使用MIIC-III临床数据对GPT-4 o-mini、Gemini-2.0-Flash和Llama-3.1- 8B-Direct模型进行的实验表明，具有预先存在的合法记忆的现实条件会显着降低攻击有效性。然后，我们提出并评估了两种新型防御机制：（1）使用多个垂直信号的复合信任评分进行输入/输出调节，以及（2）使用时间衰减和基于模式的过滤的信任感知检索进行记忆净化。我们的防御评估表明，有效的内存清理需要仔细的信任阈值校准，以防止过于保守的拒绝（阻止所有条目）和过滤不足（错过微妙攻击），为未来的自适应防御机制建立重要的基线。这些发现为在生产环境中保护内存增强的LLM代理提供了重要见解。



## **35. STELP: Secure Transpilation and Execution of LLM-Generated Programs**

STELP：LLM生成的程序的安全移植和执行 cs.SE

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.05467v3) [paper-pdf](https://arxiv.org/pdf/2601.05467v3)

**Authors**: Swapnil Shinde, Sahil Wadhwa, Andy Luo, Akshay Gupta, Mohammad Shahed Sorower

**Abstract**: Rapid evolution of Large Language Models (LLMs) has achieved major advances in reasoning, planning, and function-calling capabilities. Multi-agentic collaborative frameworks using such LLMs place them at the center of solving software development-related tasks such as code generation. However, direct use of LLM generated code in production software development systems is problematic. The code could be unstable or erroneous and contain vulnerabilities such as data poisoning, malicious attacks, and hallucinations that could lead to widespread system malfunctions. This prohibits the adoption of LLM generated code in production AI systems where human code reviews and traditional secure testing tools are impractical or untrustworthy. In this paper, we discuss safety and reliability problems with the execution of LLM generated code and propose a Secure Transpiler and Executor of LLM-Generated Program (STELP), capable of executing LLM-generated code in a controlled and safe manner. STELP secures autonomous production AI systems involving code generation, filling the critical void left by the impracticality or limitations of traditional secure testing methodologies and human oversight. This includes applications such as headless code generation-execution and LLMs that produce executable code snippets as an action plan to be executed in real time. We contribute a human-validated dataset of insecure code snippets and benchmark our approach on publicly available datasets for correctness, safety, and latency. Our results demonstrate that our approach outperforms an existing method by a significant margin, particularly in its ability to safely execute risky code snippets. Warning: This paper contains malicious code snippets that should be run with caution.

摘要: 大型语言模型（LLM）的快速发展在推理、规划和函数调用能力方面取得了重大进步。使用此类LLM的多代理协作框架将其置于解决代码生成等软件开发相关任务的中心。然而，在生产软件开发系统中直接使用LLM生成的代码是有问题的。该代码可能不稳定或错误，并且包含数据中毒、恶意攻击和幻觉等漏洞，可能导致广泛的系统故障。这禁止在人工代码审查和传统安全测试工具不切实际或不可信的生产AI系统中采用LLM生成的代码。在本文中，我们讨论的安全性和可靠性问题与LLM生成的代码的执行，并提出了一个安全的传输器和执行器的LLM生成的程序（STELP），能够执行LLM生成的代码在一个受控的和安全的方式。STELP保护涉及代码生成的自主生产AI系统，填补了传统安全测试方法和人类监督的不切实际或局限性所留下的关键空白。这包括诸如无头代码生成-执行和LLM之类的应用程序，这些应用程序生成可执行代码片段作为要实时执行的行动计划。我们提供了一个经过人类验证的不安全代码片段数据集，并在公开可用的数据集上对我们的方法进行正确性、安全性和延迟的基准测试。我们的结果表明，我们的方法比现有方法的性能有很大的差距，特别是在其安全执行有风险的代码片段的能力方面。警告：本文包含恶意代码片段，应谨慎运行。



## **36. BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents**

BackdoorAgent：针对基于LLM的代理进行后门攻击的统一框架 cs.AI

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.04566v2) [paper-pdf](https://arxiv.org/pdf/2601.04566v2)

**Authors**: Yunhao Feng, Yige Li, Yutao Wu, Yingshui Tan, Yanming Guo, Yifan Ding, Kun Zhai, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large language model (LLM) agents execute tasks through multi-step workflows that combine planning, memory, and tool use. While this design enables autonomy, it also expands the attack surface for backdoor threats. Backdoor triggers injected into specific stages of an agent workflow can persist through multiple intermediate states and adversely influence downstream outputs. However, existing studies remain fragmented and typically analyze individual attack vectors in isolation, leaving the cross-stage interaction and propagation of backdoor triggers poorly understood from an agent-centric perspective. To fill this gap, we propose \textbf{BackdoorAgent}, a modular and stage-aware framework that provides a unified, agent-centric view of backdoor threats in LLM agents. BackdoorAgent structures the attack surface into three functional stages of agentic workflows, including \textbf{planning attacks}, \textbf{memory attacks}, and \textbf{tool-use attacks}, and instruments agent execution to enable systematic analysis of trigger activation and propagation across different stages. Building on this framework, we construct a standardized benchmark spanning four representative agent applications: \textbf{Agent QA}, \textbf{Agent Code}, \textbf{Agent Web}, and \textbf{Agent Drive}, covering both language-only and multimodal settings. Our empirical analysis shows that \textit{triggers implanted at a single stage can persist across multiple steps and propagate through intermediate states.} For instance, when using a GPT-based backbone, we observe trigger persistence in 43.58\% of planning attacks, 77.97\% of memory attacks, and 60.28\% of tool-stage attacks, highlighting the vulnerabilities of the agentic workflow itself to backdoor threats. To facilitate reproducibility and future research, our code and benchmark are publicly available at GitHub.

摘要: 大型语言模型（LLM）代理通过结合规划、内存和工具使用的多步骤工作流程执行任务。虽然这种设计实现了自主性，但它也扩大了后门威胁的攻击面。注入到代理工作流程特定阶段的后门触发器可能会持续存在多个中间状态，并对下游输出产生不利影响。然而，现有的研究仍然支离破碎，通常是孤立地分析单个攻击载体，从以代理为中心的角度来看，人们对后门触发器的跨阶段相互作用和传播知之甚少。为了填补这一空白，我们提出了\textBF{BackdoorAgent}，这是一个模块化和阶段感知框架，它提供了LLM代理中后门威胁的统一、以代理为中心的视图。BackdoorAgent将攻击表面构建为代理工作流程的三个功能阶段，包括\textBF{规划攻击}、\textBF{内存攻击}和\textBF{tool-use attacks}，并工具代理执行，以实现对触发激活和跨不同阶段传播的系统分析。在此框架的基础上，我们构建了一个跨越四个代表性代理应用程序的标准化基准：\textBF{Agent QA}、\textBF{Agent Code}、\textBF{Agent Web}和\textBF{Agent Drive}，涵盖纯语言和多模式设置。我们的经验分析表明，\textit{在单个阶段植入的触发器可以持续存在多个步骤并通过中间状态传播。}例如，当使用基于GPT的主干网时，我们观察到43.58%的规划攻击、77.97%的内存攻击和60.28%的工具阶段攻击的触发持续性，凸显了代理工作流程本身对后门威胁的脆弱性。为了促进可重复性和未来的研究，我们的代码和基准已在GitHub上公开。



## **37. TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards**

TROJail：多回合大型语言模型越狱的轨迹级优化，并给予流程奖励 cs.AI

21 pages, 15 figures

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2512.07761v2) [paper-pdf](https://arxiv.org/pdf/2512.07761v2)

**Authors**: Xiqiao Xiong, Ouxiang Li, Zhuo Liu, Moxin Li, Wentao Shi, Fengbin Zhu, Qifan Wang, Fuli Feng

**Abstract**: Large language models have seen widespread adoption, yet they remain vulnerable to multi-turn jailbreak attacks, threatening their safe deployment. This has led to the task of training automated multi-turn attackers to probe model safety vulnerabilities. However, existing approaches typically rely on turn-level optimization, which is insufficient for learning long-term attack strategies. To bridge this gap, we formulate this task as a multi-turn reinforcement learning problem, directly optimizing the harmfulness of the final-turn response as the outcome reward. To address the sparse supervision of the outcome reward, we introduce TROJail, which employs two process rewards to evaluate the utility of intermediate prompts and integrate them into advantage estimation. These rewards (1) penalize overly harmful prompts that trigger the model's refusal mechanism, and (2) encourage steering the semantic relevance of responses toward the targeted harmful content. Experimental results show improved attack success rates across multiple models and benchmarks, highlighting the effectiveness of our approach. The code is available at https://github.com/xxiqiao/TROJail. Warning: This paper contains examples of harmful content.

摘要: 大型语言模型已经被广泛采用，但它们仍然容易受到多轮越狱攻击，威胁到它们的安全部署。这导致了训练自动化多轮攻击者来探测模型安全漏洞的任务。然而，现有的方法通常依赖于回合级优化，这是不足以学习长期的攻击策略。为了弥合这一差距，我们将此任务制定为多回合强化学习问题，直接优化最终回合反应的危害性作为结果奖励。为了解决结果奖励的稀疏监督问题，我们引入了TROJail，它采用两个流程奖励来评估中间提示的效用，并将其集成到优势估计中。这些奖励（1）惩罚触发模型拒绝机制的过于有害的提示，以及（2）鼓励将响应的语义相关性引导到目标有害内容。实验结果表明，多个模型和基准的攻击成功率有所提高，凸显了我们方法的有效性。该代码可在https://github.com/xxiqiao/TROJail上获取。警告：本文包含有害内容的示例。



## **38. From static to adaptive: immune memory-based jailbreak detection for large language models**

从静态到适应性：基于免疫记忆的大型语言模型越狱检测 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2512.03356v2) [paper-pdf](https://arxiv.org/pdf/2512.03356v2)

**Authors**: Jun Leng, Yu Liu, Litian Zhang, Ruihan Hu, Zhuting Fang, Xi Zhang

**Abstract**: Large Language Models (LLMs) serve as the backbone of modern AI systems, yet they remain susceptible to adversarial jailbreak attacks. Consequently, robust detection of such malicious inputs is paramount for ensuring model safety. Traditional detection methods typically rely on external models trained on fixed, large-scale datasets, which often incur significant computational overhead. While recent methods shift toward leveraging internal safety signals of models to enable more lightweight and efficient detection. However, these methods remain inherently static and struggle to adapt to the evolving nature of jailbreak attacks. Drawing inspiration from the biological immune mechanism, we introduce the Immune Memory Adaptive Guard (IMAG) framework. By distilling and encoding safety patterns into a persistent, evolvable memory bank, IMAG enables adaptive generalization to emerging threats. Specifically, the framework orchestrates three synergistic components: Immune Detection, which employs retrieval for high-efficiency interception of known jailbreak attacks; Active Immunity, which performs proactive behavioral simulation to resolve ambiguous unknown queries; Memory Updating, which integrates validated attack patterns back into the memory bank. This closed-loop architecture transitions LLM defense from rigid filtering to autonomous adaptive mitigation. Extensive evaluations across five representative open-source LLMs demonstrate that our method surpasses state-of-the-art (SOTA) baselines, achieving a superior average detection accuracy of 94\% across diverse and complex attack types.

摘要: 大型语言模型（LLM）是现代人工智能系统的支柱，但它们仍然容易受到敌对越狱攻击。因此，对此类恶意输入的稳健检测对于确保模型安全至关重要。传统的检测方法通常依赖于在固定的大规模数据集上训练的外部模型，这通常会带来大量的计算负担。虽然最近的方法转向利用模型的内部安全信号来实现更轻量级和更高效的检测。然而，这些方法本质上仍然是静态的，并且很难适应越狱攻击不断变化的性质。从生物免疫机制中汲取灵感，我们介绍了免疫记忆自适应守卫（IMAG）框架。通过将安全模式提取和编码到持久的、可进化的记忆库中，IMAG能够对新出现的威胁进行自适应概括。具体来说，该框架协调了三个协同组件：免疫检测，它采用检索来高效拦截已知的越狱攻击;主动免疫，它执行主动行为模拟来解决模糊的未知查询;内存更新，它将验证的攻击模式集成到内存库中。这种闭环架构将LLM防御从严格过滤转变为自主自适应缓解。对五个代表性开源LLM的广泛评估表明，我们的方法超越了最先进的（SOTA）基线，在各种复杂的攻击类型中实现了94%的卓越平均检测准确率。



## **39. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

对抗性诗歌作为大型语言模型中通用的单轮越狱机制 cs.CL

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2511.15304v3) [paper-pdf](https://arxiv.org/pdf/2511.15304v3)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale Syrnikov, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.

摘要: 我们提供的证据表明，对抗性诗歌可以作为大型语言模型（LLM）的通用单轮越狱技术。在25个前沿专有和开放重量模型中，精心策划的诗意提示产生了很高的攻击成功率（ASB），一些提供商超过了90%。MLCommons和EU CoP风险分类的映射提示表明，诗意攻击跨CBRN、操纵、网络犯罪和失去控制领域转移。通过标准化元提示将1，200个MLCommons有害提示转换为诗句，产生的ASB比散文基线高出18倍。使用3名开放权重LLM评委的整体评估输出，他们的二元安全性评估在分层的人类标记子集上进行了验证。诗意框架的平均越狱成功率为62%，元提示转换的平均越狱成功率约为43%（与非诗意基线相比），大大优于非诗意基线，并揭示了示范家庭和安全培训方法之间的系统性弱点。这些研究结果表明，仅靠风格差异就可以规避当代的安全机制，这表明当前对齐方法和评估协议存在根本局限性。



## **40. NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge**

NeuGen Poisoning：通过外部知识的遗传优化对LLM检索增强生成的神经元引导攻击 cs.AI

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2510.21144v2) [paper-pdf](https://arxiv.org/pdf/2510.21144v2)

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.

摘要: 检索增强生成（RAG）使大型语言模型（LLM）能够在推理期间动态集成外部知识，提高其事实准确性和适应性。然而，对手可以注入有毒的外部知识来覆盖模型的内部记忆。虽然现有的攻击迭代地操纵RAG的检索内容或提示结构，但它们在很大程度上忽略了模型的内部表示动态和神经元级敏感性。RAG中毒的根本机制尚未得到充分研究，也没有考虑RAG中知识冲突与强参数知识的影响。在这项工作中，我们提出了NeuGenPoisoning，这是一种新型攻击框架，可以在LLM内部神经元归因和遗传优化的指导下在RAG中生成对抗性外部知识。我们的方法首先识别一组中毒反应神经元，其激活与上下文中毒知识密切相关。然后，我们采用遗传算法来进化对抗通道，最大限度地激活这些神经元。至关重要的是，我们的框架通过观察到的归因信号识别和重用有希望但最初不成功的外部知识变体，从而能够大规模地生成有效的有毒RAG知识。同时，中毒响应神经元引导中毒可以有效解决知识冲突。模型和数据集的实验结果表明，在保持流畅性的同时，始终实现了超过90%的高人口覆盖成功率（POSR）。经验证据表明，我们的方法有效地解决了知识冲突。



## **41. Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers**

我的优化预算是否受到影响？探索基于LLM的优化器的漏洞 cs.LG

Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026)

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2510.14381v2) [paper-pdf](https://arxiv.org/pdf/2510.14381v2)

**Authors**: Andrew Zhao, Reshmi Ghosh, Vitor Carvalho, Emily Lawton, Keegan Hines, Gao Huang, Jack W. Stokes

**Abstract**: Large language model (LLM) systems increasingly power everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on manually well-crafted prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to query poisoning alone: feedback-based attacks raise attack success rate (ASR) by up to ΔASR = 0.48. We introduce a simple fake reward attack that requires no access to the reward model and significantly increases vulnerability. We also propose a lightweight highlighting defense that reduces the fake reward ΔASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks.

摘要: 大型语言模型（LLM）系统越来越多地支持聊天机器人、计算机助理和自主机器人等日常人工智能应用程序，其中的性能通常取决于手工精心制作的提示。基于LLM的提示优化器通过迭代地从评分反馈中改进提示来减少这一工作，但该优化阶段的安全性仍然不足。我们在基于LLM的即时优化中首次对中毒风险进行了系统分析。使用HarmBench，我们发现系统更容易受到操纵反馈的影响，而不是单独受到查询中毒的影响：基于反馈的攻击将攻击成功率（ASB）提高了最高为A ASB = 0.48。我们引入了一种简单的虚假奖励攻击，它不需要访问奖励模型，并显着增加了漏洞。我们还提出了一种轻量级的突出显示防御，可以将虚假奖励Δ ASB从0.23减少到0.07，而不会降低效用。这些结果将即时优化管道建立为一流的攻击面，并激励反馈渠道和优化框架更强有力的保障措施。



## **42. RIPRAG: Hack a Black-box Retrieval-Augmented Generation Question-Answering System with Reinforcement Learning**

RIPRAG：利用强化学习破解黑匣子检索增强一代语音响应系统 cs.AI

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2510.10008v2) [paper-pdf](https://arxiv.org/pdf/2510.10008v2)

**Authors**: Meng Xi, Sihan Lv, Yechen Jin, Guanjie Cheng, Naibo Wang, Ying Li, Jianwei Yin

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become a core technology for tasks such as question-answering (QA) and content generation. RAG poisoning is an attack method to induce LLMs to generate the attacker's expected text by injecting poisoned documents into the database of RAG systems. Existing research can be broadly divided into two classes: white-box methods and black-box methods. White-box methods utilize gradient information to optimize poisoned documents, and black-box methods use a pre-trained LLM to generate them. However, existing white-box methods require knowledge of the RAG system's internal composition and implementation details, whereas black-box methods are unable to utilize interactive information. In this work, we propose the RIPRAG attack framework, an end-to-end attack pipeline that treats the target RAG system as a black box and leverages our proposed Reinforcement Learning from Black-box Feedback (RLBF) method to optimize the generation model for poisoned documents. We designed two kinds of rewards: similarity reward and attack reward. Experimental results demonstrate that this method can effectively execute poisoning attacks against most complex RAG systems, achieving an attack success rate (ASR) improvement of up to 0.72 compared to baseline methods. This highlights prevalent deficiencies in current defensive methods and provides critical insights for LLM security research.

摘要: 基于大型语言模型（LLM）的检索增强生成（RAG）系统已成为问答（QA）和内容生成等任务的核心技术。RAG中毒是一种攻击方法，通过将有毒文档注入RAG系统的数据库，诱导LLM生成攻击者预期的文本。现有的研究大致可以分为两类：白盒方法和黑盒方法。白盒方法利用梯度信息来优化有毒文档，黑盒方法使用预先训练的LLM来生成它们。然而，现有的白盒方法需要了解RAG系统的内部组成和实现细节，而黑盒方法无法利用交互式信息。在这项工作中，我们提出了RIPRAG攻击框架，这是一种端到端攻击管道，将目标RAG系统视为黑匣子，并利用我们提出的黑匣子反馈强化学习（WLBF）方法来优化有毒文档的生成模型。我们设计了两种奖励：相似性奖励和攻击奖励。实验结果表明，该方法可以有效地对大多数复杂的RAG系统执行中毒攻击，与基线方法相比，攻击成功率（ASB）提高高达0.72。这突出了当前防御方法的普遍缺陷，并为LLM安全研究提供了重要见解。



## **43. Membership Inference Attacks on Tokenizers of Large Language Models**

对大型语言模型令牌器的成员推断攻击 cs.CR

To appear at USENIX Security Symposium 2026

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2510.05699v2) [paper-pdf](https://arxiv.org/pdf/2510.05699v2)

**Authors**: Meng Tong, Yuntao Du, Kejiang Chen, Weiming Zhang

**Abstract**: Membership inference attacks (MIAs) are widely used to assess the privacy risks associated with machine learning models. However, when these attacks are applied to pre-trained large language models (LLMs), they encounter significant challenges, including mislabeled samples, distribution shifts, and discrepancies in model size between experimental and real-world settings. To address these limitations, we introduce tokenizers as a new attack vector for membership inference. Specifically, a tokenizer converts raw text into tokens for LLMs. Unlike full models, tokenizers can be efficiently trained from scratch, thereby avoiding the aforementioned challenges. In addition, the tokenizer's training data is typically representative of the data used to pre-train LLMs. Despite these advantages, the potential of tokenizers as an attack vector remains unexplored. To this end, we present the first study on membership leakage through tokenizers and explore five attack methods to infer dataset membership. Extensive experiments on millions of Internet samples reveal the vulnerabilities in the tokenizers of state-of-the-art LLMs. To mitigate this emerging risk, we further propose an adaptive defense. Our findings highlight tokenizers as an overlooked yet critical privacy threat, underscoring the urgent need for privacy-preserving mechanisms specifically designed for them.

摘要: 成员资格推理攻击（MIA）广泛用于评估与机器学习模型相关的隐私风险。然而，当这些攻击应用于预训练的大型语言模型（LLM）时，它们会遇到重大挑战，包括样本标签错误、分布变化以及实验环境和现实环境之间模型大小的差异。为了解决这些限制，我们引入了标记器作为成员资格推断的新攻击载体。具体来说，标记器将原始文本转换为LLM的标记。与完整模型不同，标记器可以从头开始有效训练，从而避免上述挑战。此外，标记化器的训练数据通常代表用于预训练LLM的数据。尽管有这些优势，但标记器作为攻击载体的潜力仍未被开发。为此，我们提出了第一项关于通过标记器的成员资格泄露的研究，并探索了五种推断数据集成员资格的攻击方法。对数百万个互联网样本的广泛实验揭示了最先进的LLM标记器中的漏洞。为了减轻这种新出现的风险，我们进一步提出了适应性防御。我们的研究结果强调，代币使用者是一种被忽视但又严重的隐私威胁，强调迫切需要专门为它们设计的隐私保护机制。



## **44. Secure and Efficient Access Control for Computer-Use Agents via Context Space**

通过上下文空间对计算机使用代理进行安全有效的访问控制 cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2509.22256v4) [paper-pdf](https://arxiv.org/pdf/2509.22256v4)

**Authors**: Haochen Gong, Chenxiao Li, Rui Chang, Wenbo Shen

**Abstract**: Large language model (LLM)-based computer-use agents represent a convergence of AI and OS capabilities, enabling natural language to control system- and application-level functions. However, due to LLMs' inherent uncertainty issues, granting agents control over computers poses significant security risks. When agent actions deviate from user intentions, they can cause irreversible consequences. Existing mitigation approaches, such as user confirmation and LLM-based dynamic action validation, still suffer from limitations in usability, security, and performance. To address these challenges, we propose CSAgent, a system-level, static policy-based access control framework for computer-use agents. To bridge the gap between static policy and dynamic context and user intent, CSAgent introduces intent- and context-aware policies, and provides an automated toolchain to assist developers in constructing and refining them. CSAgent enforces these policies through an optimized OS service, ensuring that agent actions can only be executed under specific user intents and contexts. CSAgent supports protecting agents that control computers through diverse interfaces, including API, CLI, and GUI. We implement and evaluate CSAgent, which successfully defends against all attacks in the benchmarks while introducing only 1.99% performance overhead and 5.42% utility decrease.

摘要: 基于大型语言模型（LLM）的计算机使用代理代表了人工智能和操作系统功能的融合，使自然语言能够控制系统和应用程序级功能。然而，由于LLM固有的不确定性问题，授予代理对计算机的控制权会带来巨大的安全风险。当代理行为偏离用户意图时，可能会导致不可逆转的后果。现有的缓解方法，例如用户确认和基于LLM的动态动作验证，仍然受到可用性、安全性和性能方面的限制。为了解决这些挑战，我们提出了CSAgent，这是一个用于计算机使用代理的系统级、基于静态策略的访问控制框架。为了弥合静态策略与动态上下文和用户意图之间的差距，CSAgent引入了意图和上下文感知策略，并提供了自动化工具链来帮助开发人员构建和完善它们。CSAgent通过优化的操作系统服务执行这些策略，确保代理操作只能在特定的用户意图和上下文下执行。CSAgent支持保护通过各种接口（包括API、CLI和图形用户界面）控制计算机的代理。我们实施并评估了CSAgent，它成功防御了基准测试中的所有攻击，同时仅引入1.99%的性能开销和5.42%的实用性下降。



## **45. NATLM: Detecting Defects in NFT Smart Contracts Leveraging LLM**

NATLM：利用LLM检测NFT智能合同中的缺陷 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2508.01351v2) [paper-pdf](https://arxiv.org/pdf/2508.01351v2)

**Authors**: Yuanzheng Niu, Xiaoqi Li, Wenkai Li

**Abstract**: Security issues are becoming increasingly significant with the rapid evolution of Non-fungible Tokens (NFTs). As NFTs are traded as digital assets, they have emerged as prime targets for cyber attackers. In the development of NFT smart contracts, there may exist undiscovered defects that could lead to substantial financial losses if exploited. To tackle this issue, this paper presents a framework called NATLM(NFT Assistant LLM), designed to detect potential defects in NFT smart contracts. The framework effectively identifies four common types of vulnerabilities in NFT smart contracts: ERC-721 Reentrancy, Public Burn, Risky Mutable Proxy, and Unlimited Minting. Relying exclusively on large language models (LLMs) for defect detection can lead to a high false-positive rate. To enhance detection performance, NATLM integrates static analysis with LLMs, specifically Gemini Pro 1.5. Initially, NATLM employs static analysis to extract structural, syntactic, and execution flow information from the code, represented through Abstract Syntax Trees (AST) and Control Flow Graphs (CFG). These extracted features are then combined with vectors of known defect examples to create a matrix for input into the knowledge base. Subsequently, the feature vectors and code vectors of the analyzed contract are compared with the contents of the knowledge base. Finally, the LLM performs deep semantic analysis to enhance detection capabilities, providing a more comprehensive and accurate identification of potential security issues. Experimental results indicate that NATLM analyzed 8,672 collected NFT smart contracts, achieving an overall precision of 87.72%, a recall of 89.58%, and an F1 score of 88.94%. The results outperform other baseline experiments, successfully identifying four common types of defects.

摘要: 随着不可替代代币（NFT）的快速发展，安全问题变得越来越重要。由于NFT作为数字资产交易，它们已成为网络攻击者的主要目标。在NFT智能合约的开发过程中，可能存在未被发现的缺陷，如果被利用，可能会导致巨额财务损失。为了解决这个问题，本文提出了一个名为NATLM（NFT助理LLM）的框架，旨在检测NFT智能合约中的潜在缺陷。该框架有效地识别了NFT智能合约中的四种常见漏洞类型：ERC-721 Reentrency、Public Burn、Risky Mutable代理和Unlimited Minting。完全依赖大型语言模型（LLM）进行缺陷检测可能会导致高假阳性率。为了增强检测性能，NATLM将静态分析与LLM集成，特别是Gemini Pro 1.5。最初，NATLM采用静态分析从代码中提取结构、语法和执行流信息，通过抽象语法树（AST）和控制流图（CGM）表示。然后将这些提取的特征与已知缺陷示例的载体相结合，以创建一个矩阵，用于输入到知识库中。随后，将分析后的合同的特征载体和代码载体与知识库的内容进行比较。最后，LLM进行深度语义分析以增强检测能力，从而更全面、准确地识别潜在的安全问题。实验结果表明，NATLM分析了收集的8，672份NFT智能合约，总体准确率为87.72%，召回率为89.58%，F1评分为88.94%。结果优于其他基线实验，成功识别了四种常见类型的缺陷。



## **46. AI Agent Smart Contract Exploit Generation**

AI Agent智能合同漏洞生成 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2507.05558v4) [paper-pdf](https://arxiv.org/pdf/2507.05558v4)

**Authors**: Arthur Gervais, Liyi Zhou

**Abstract**: Smart contract vulnerabilities have led to billions in losses, yet finding actionable exploits remains challenging. Traditional fuzzers rely on rigid heuristics and struggle with complex attacks, while human auditors are thorough but slow and don't scale. Large Language Models offer a promising middle ground, combining human-like reasoning with machine speed.   Early studies show that simply prompting LLMs generates unverified vulnerability speculations with high false positive rates. To address this, we present A1, an agentic system that transforms any LLM into an end-to-end exploit generator. A1 provides agents with six domain-specific tools for autonomous vulnerability discovery, from understanding contract behavior to testing strategies on real blockchain states. All outputs are concretely validated through execution, ensuring only profitable proof-of-concept exploits are reported. We evaluate A1 across 36 real-world vulnerable contracts on Ethereum and Binance Smart Chain. A1 achieves a 63% success rate on the VERITE benchmark. Across all successful cases, A1 extracts up to \$8.59 million per exploit and \$9.33 million total.   Using Monte Carlo analysis of historical attacks, we demonstrate that immediate vulnerability detection yields 86-89% success probability, dropping to 6-21% with week-long delays. Our economic analysis reveals a troubling asymmetry: attackers achieve profitability at \$6,000 exploit values while defenders require \$60,000 -- raising fundamental questions about whether AI agents inevitably favor exploitation over defense.

摘要: 智能合同漏洞已导致数十亿美元的损失，但发现可采取行动的漏洞仍然具有挑战性。传统的模糊器依赖于严格的启发式方法，并与复杂的攻击作斗争，而人类审计员很彻底，但速度缓慢，而且无法扩展。大型语言模型提供了一个有希望的中间立场，将类人推理与机器速度相结合。   早期研究表明，简单地提示LLM会产生未经验证且误报率很高的漏洞猜测。为了解决这个问题，我们提出了A1，这是一个代理系统，可以将任何LLM转换为端到端漏洞利用生成器。A1为代理提供了六种特定于领域的工具，用于自主漏洞发现，从理解合同行为到测试真实区块链状态的策略。所有输出都通过执行进行具体验证，确保仅报告有利可图的概念验证漏洞。我们评估了以太坊和币安智能链上36个现实世界的脆弱合同的A1。A1在VERITE基准测试上实现了63%的成功率。在所有成功的案例中，A1每次利用可提取高达859万英镑，总计可提取933万英镑。   使用对历史攻击的蒙特卡洛分析，我们证明立即漏洞检测的成功率为86-89%，在长达一周的延迟后，成功率下降到6-21%。我们的经济分析揭示了一种令人不安的不对称性：攻击者以6，000英镑的利用价值实现盈利，而防御者则需要60，000英镑--这引发了关于人工智能代理是否不可避免地更喜欢利用而不是防御的基本问题。



## **47. Exploring the Secondary Risks of Large Language Models**

探索大型语言模型的次要风险 cs.LG

18 pages, 5 figures

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2506.12382v4) [paper-pdf](https://arxiv.org/pdf/2506.12382v4)

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments.

摘要: 随着大型语言模型越来越多地集成到关键应用程序和社会功能中，确保大型语言模型的安全性和一致性是一项重大挑战。虽然之前的研究主要集中在越狱攻击上，但对良性互动中微妙出现的非对抗性失败的关注较少。我们引入了二级风险，这是一种新型的失败模式，其特征是良性提示期间的有害或误导行为。与对抗性攻击不同，这些风险源于不完美的概括，并且常常逃避标准安全机制。为了实现系统性评估，我们引入了两个风险基元：详细响应和推测性建议，以捕捉核心故障模式。在这些定义的基础上，我们提出了SecLens，这是一个黑匣子、多目标搜索框架，通过优化任务相关性、风险激活和语言合理性来有效地引发次要风险行为。为了支持可重复的评估，我们发布了SecRiskBench，这是一个由650个提示组成的基准数据集，涵盖八个不同的现实世界风险类别。对16种流行模型进行广泛评估的实验结果表明，次级风险是普遍存在的，可以跨模型转移，并且独立于模式，这强调了迫切需要增强的安全机制来解决现实世界部署中良性但有害的LLM行为。



## **48. Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation**

灵丹妙药：通过微调后扰动缓解大型语言模型的有害微调 cs.CL

Accepted by NeruIPS 2025

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2501.18100v2) [paper-pdf](https://arxiv.org/pdf/2501.18100v2)

**Authors**: Yibo Wang, Tiansheng Huang, Li Shen, Huanjin Yao, Haotian Luo, Rui Liu, Naiqiang Tan, Jiaxing Huang, Dacheng Tao

**Abstract**: Harmful fine-tuning attack introduces significant security risks to the fine-tuning services. Main-stream defenses aim to vaccinate the model such that the later harmful fine-tuning attack is less effective. However, our evaluation results show that such defenses are fragile--with a few fine-tuning steps, the model still can learn the harmful knowledge. To this end, we do further experiment and find that an embarrassingly simple solution--adding purely random perturbations to the fine-tuned model, can recover the model from harmful behaviors, though it leads to a degradation in the model's fine-tuning performance. To address the degradation of fine-tuning performance, we further propose Panacea, which optimizes an adaptive perturbation that will be applied to the model after fine-tuning. Panacea maintains model's safety alignment performance without compromising downstream fine-tuning performance. Comprehensive experiments are conducted on different harmful ratios, fine-tuning tasks and mainstream LLMs, where the average harmful scores are reduced by up-to 21.2%, while maintaining fine-tuning performance. As a by-product, we analyze the adaptive perturbation and show that different layers in various LLMs have distinct safety affinity, which coincide with finding from several previous study. Source code available at https://github.com/w-yibo/Panacea.

摘要: 有害的微调攻击给微调服务带来了重大的安全风险。主流防御旨在为模型接种疫苗，以便后期有害的微调攻击效果较差。然而，我们的评估结果表明，这种防御是脆弱的--只需进行一些微调步骤，模型仍然可以学习有害知识。为此，我们做了进一步的实验，发现一个极其简单的解决方案--在微调模型中添加纯粹的随机扰动，可以将模型从有害行为中恢复出来，尽管它会导致模型的微调性能下降。为了解决微调性能的下降问题，我们进一步提出了Panacea，它优化了自适应扰动，该扰动将在微调后应用于模型。Panacea在不影响下游微调性能的情况下保持了模型的安全对准性能。对不同的有害比例、微调任务和主流LLM进行了全面实验，平均有害分数降低高达21.2%，同时保持微调性能。作为副产品，我们分析了自适应扰动，并表明各种LLM中的不同层具有不同的安全亲和力，这与之前几项研究的发现一致。源代码可在https://github.com/w-yibo/Panacea获得。



## **49. Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models**

越狱音频长凳：深入评估和分析大型音频语言模型的越狱威胁 cs.SD

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2501.13772v4) [paper-pdf](https://arxiv.org/pdf/2501.13772v4)

**Authors**: Hao Cheng, Erjia Xiao, Jing Shao, Yichi Wang, Le Yang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Large Language Models (LLMs) demonstrate impressive zero-shot performance across a wide range of natural language processing tasks. Integrating various modality encoders further expands their capabilities, giving rise to Multimodal Large Language Models (MLLMs) that process not only text but also visual and auditory modality inputs. However, these advanced capabilities may also pose significant safety problems, as models can be exploited to generate harmful or inappropriate content through jailbreak attacks. While prior work has extensively explored how manipulating textual or visual modality inputs can circumvent safeguards in LLMs and MLLMs, the vulnerability of audio-specific jailbreak on Large Audio-Language Models (LALMs) remains largely underexplored. To address this gap, we introduce Jailbreak-AudioBench, which consists of the Toolbox, curated Dataset, and comprehensive Benchmark. The Toolbox supports not only text-to-audio conversion but also various editing techniques for injecting audio hidden semantics. The curated Dataset provides diverse explicit and implicit jailbreak audio examples in both original and edited forms. Utilizing this dataset, we evaluate multiple state-of-the-art LALMs and establish the most comprehensive Jailbreak benchmark to date for audio modality. Finally, Jailbreak-AudioBench establishes a foundation for advancing future research on LALMs safety alignment by enabling the in-depth exposure of more powerful jailbreak threats, such as query-based audio editing, and by facilitating the development of effective defense mechanisms.

摘要: 大型语言模型（LLM）在广泛的自然语言处理任务中表现出令人印象深刻的零冲击性能。集成各种模式编码器进一步扩展了它们的功能，从而产生了多模式大型语言模型（MLLM），不仅处理文本，还处理视觉和听觉模式输入。然而，这些高级功能也可能带来严重的安全问题，因为模型可能会被利用来通过越狱攻击生成有害或不适当的内容。虽然之前的工作已经广泛探索了操纵文本或视觉模式输入如何规避LLM和MLLM中的保护措施，但大型音频语言模型（LALM）上特定音频越狱的漏洞在很大程度上仍然没有得到充分的研究。为了弥补这一差距，我们引入了Jailbreak-AudioBench，它由收件箱、精心策划的数据集和全面的基准组成。收件箱不仅支持文本到音频的转换，还支持各种用于注入音频隐藏语义的编辑技术。精心策划的数据集以原始和编辑的形式提供了多样化的显式和隐式越狱音频示例。利用该数据集，我们评估了多种最先进的LALM，并为音频模式建立了迄今为止最全面的越狱基准。最后，Jailbreak-AudioBench通过深入暴露更强大的越狱威胁（例如基于查询的音频编辑）并促进有效防御机制的开发，为推进未来对LALM安全性的研究奠定了基础。



## **50. Can Editing LLMs Inject Harm?**

编辑LLM会造成伤害吗？ cs.CL

Accepted to Proceedings of AAAI 2026. The first two authors contributed equally. 7 pages for main paper, 31 pages including appendix. The code, results, dataset for this paper and more resources are on the project website: https://llm-editing.github.io

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2407.20224v4) [paper-pdf](https://arxiv.org/pdf/2407.20224v4)

**Authors**: Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, Kai Shu

**Abstract**: Large Language Models (LLMs) have emerged as a new information channel. Meanwhile, one critical but under-explored question is: Is it possible to bypass the safety alignment and inject harmful information into LLMs stealthily? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the first risk, we find that editing attacks can inject both commonsense and long-tail misinformation into LLMs, and the effectiveness for the former one is particularly high. For the second risk, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can degrade the overall fairness. Then, we further illustrate the high stealthiness of editing attacks. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.

摘要: 大型语言模型（LLM）已成为一种新的信息渠道。与此同时，一个关键但尚未充分探讨的问题是：是否有可能绕过安全调整并秘密地将有害信息注入到LLM中？本文提出将知识编辑重新定义为LLM的一种新型安全威胁，即编辑攻击，并利用新构建的数据集EditAttack进行系统调查。具体来说，我们重点关注编辑攻击的两种典型安全风险，即错误信息注入和偏差注入。对于第一个风险，我们发现编辑攻击可以向LLM注入常识和长尾错误信息，并且前者的有效性特别高。对于第二个风险，我们发现，不仅可以有偏见的句子注入到LLM具有高效率，但也有一个单一的偏见句子注入可以降低整体的公平性。然后，我们进一步说明了高隐蔽性的编辑攻击。我们的发现证明了知识编辑技术在损害LLM安全性方面的新出现的滥用风险，以及将LLM作为新渠道传播错误信息或偏见的可行性。



