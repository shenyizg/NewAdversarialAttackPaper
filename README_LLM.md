# Latest Large Language Model Attack Papers
**update at 2026-01-19 16:15:10**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Membership Inference on LLMs in the Wild**

cs.CL

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11314v1) [paper-pdf](https://arxiv.org/pdf/2601.11314v1)

**Authors**: Jiatong Yi, Yanyang Li

**Abstract**: Membership Inference Attacks (MIAs) act as a crucial auditing tool for the opaque training data of Large Language Models (LLMs). However, existing techniques predominantly rely on inaccessible model internals (e.g., logits) or suffer from poor generalization across domains in strict black-box settings where only generated text is available. In this work, we propose SimMIA, a robust MIA framework tailored for this text-only regime by leveraging an advanced sampling strategy and scoring mechanism. Furthermore, we present WikiMIA-25, a new benchmark curated to evaluate MIA performance on modern proprietary LLMs. Experiments demonstrate that SimMIA achieves state-of-the-art results in the black-box setting, rivaling baselines that exploit internal model information.



## **2. SD-RAG: A Prompt-Injection-Resilient Framework for Selective Disclosure in Retrieval-Augmented Generation**

cs.CR

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11199v1) [paper-pdf](https://arxiv.org/pdf/2601.11199v1)

**Authors**: Aiman Al Masoud, Marco Arazzi, Antonino Nocera

**Abstract**: Retrieval-Augmented Generation (RAG) has attracted significant attention due to its ability to combine the generative capabilities of Large Language Models (LLMs) with knowledge obtained through efficient retrieval mechanisms over large-scale data collections. Currently, the majority of existing approaches overlook the risks associated with exposing sensitive or access-controlled information directly to the generation model. Only a few approaches propose techniques to instruct the generative model to refrain from disclosing sensitive information; however, recent studies have also demonstrated that LLMs remain vulnerable to prompt injection attacks that can override intended behavioral constraints. For these reasons, we propose a novel approach to Selective Disclosure in Retrieval-Augmented Generation, called SD-RAG, which decouples the enforcement of security and privacy constraints from the generation process itself. Rather than relying on prompt-level safeguards, SD-RAG applies sanitization and disclosure controls during the retrieval phase, prior to augmenting the language model's input. Moreover, we introduce a semantic mechanism to allow the ingestion of human-readable dynamic security and privacy constraints together with an optimized graph-based data model that supports fine-grained, policy-aware retrieval. Our experimental evaluation demonstrates the superiority of SD-RAG over baseline existing approaches, achieving up to a $58\%$ improvement in the privacy score, while also showing a strong resilience to prompt injection attacks targeting the generative model.



## **3. AJAR: Adaptive Jailbreak Architecture for Red-teaming**

cs.CR

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.10971v1) [paper-pdf](https://arxiv.org/pdf/2601.10971v1)

**Authors**: Yipu Dou, Wang Yang

**Abstract**: As Large Language Models (LLMs) evolve from static chatbots into autonomous agents capable of tool execution, the landscape of AI safety is shifting from content moderation to action security. However, existing red-teaming frameworks remain bifurcated: they either focus on rigid, script-based text attacks or lack the architectural modularity to simulate complex, multi-turn agentic exploitations. In this paper, we introduce AJAR (Adaptive Jailbreak Architecture for Red-teaming), a proof-of-concept framework designed to bridge this gap through Protocol-driven Cognitive Orchestration. Built upon the robust runtime of Petri, AJAR leverages the Model Context Protocol (MCP) to decouple adversarial logic from the execution loop, encapsulating state-of-the-art algorithms like X-Teaming as standardized, plug-and-play services. We validate the architectural feasibility of AJAR through a controlled qualitative case study, demonstrating its ability to perform stateful backtracking within a tool-use environment. Furthermore, our preliminary exploration of the "Agentic Gap" reveals a complex safety dynamic: while tool usage introduces new injection vectors via code execution, the cognitive load of parameter formatting can inadvertently disrupt persona-based attacks. AJAR is open-sourced to facilitate the standardized, environment-aware evaluation of this emerging attack surface. The code and data are available at https://github.com/douyipu/ajar.



## **4. Beyond Max Tokens: Stealthy Resource Amplification via Tool Calling Chains in LLM Agents**

cs.CR

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.10955v1) [paper-pdf](https://arxiv.org/pdf/2601.10955v1)

**Authors**: Kaiyu Zhou, Yongsen Zheng, Yicheng He, Meng Xue, Xueluan Gong, Yuji Wang, Kwok-Yan Lam

**Abstract**: The agent-tool communication loop is a critical attack surface in modern Large Language Model (LLM) agents. Existing Denial-of-Service (DoS) attacks, primarily triggered via user prompts or injected retrieval-augmented generation (RAG) context, are ineffective for this new paradigm. They are fundamentally single-turn and often lack a task-oriented approach, making them conspicuous in goal-oriented workflows and unable to exploit the compounding costs of multi-turn agent-tool interactions. We introduce a stealthy, multi-turn economic DoS attack that operates at the tool layer under the guise of a correctly completed task. Our method adjusts text-visible fields and a template-governed return policy in a benign, Model Context Protocol (MCP)-compatible tool server, optimizing these edits with a Monte Carlo Tree Search (MCTS) optimizer. These adjustments leave function signatures unchanged and preserve the final payload, steering the agent into prolonged, verbose tool-calling sequences using text-only notices. This compounds costs across turns, escaping single-turn caps while keeping the final answer correct to evade validation. Across six LLMs on the ToolBench and BFCL benchmarks, our attack expands tasks into trajectories exceeding 60,000 tokens, inflates costs by up to 658x, and raises energy by 100-560x. It drives GPU KV cache occupancy from <1% to 35-74% and cuts co-running throughput by approximately 50%. Because the server remains protocol-compatible and task outcomes are correct, conventional checks fail. These results elevate the agent-tool interface to a first-class security frontier, demanding a paradigm shift from validating final answers to monitoring the economic and computational cost of the entire agentic process.



## **5. Detecting Winning Arguments with Large Language Models and Persuasion Strategies**

cs.CL

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10660v1) [paper-pdf](https://arxiv.org/pdf/2601.10660v1)

**Authors**: Tiziano Labruna, Arkadiusz Modzelewski, Giorgio Satta, Giovanni Da San Martino

**Abstract**: Detecting persuasion in argumentative text is a challenging task with important implications for understanding human communication. This work investigates the role of persuasion strategies - such as Attack on reputation, Distraction, and Manipulative wording - in determining the persuasiveness of a text. We conduct experiments on three annotated argument datasets: Winning Arguments (built from the Change My View subreddit), Anthropic/Persuasion, and Persuasion for Good. Our approach leverages large language models (LLMs) with a Multi-Strategy Persuasion Scoring approach that guides reasoning over six persuasion strategies. Results show that strategy-guided reasoning improves the prediction of persuasiveness. To better understand the influence of content, we organize the Winning Argument dataset into broad discussion topics and analyze performance across them. We publicly release this topic-annotated version of the dataset to facilitate future research. Overall, our methodology demonstrates the value of structured, strategy-aware prompting for enhancing interpretability and robustness in argument quality assessment.



## **6. Be Your Own Red Teamer: Safety Alignment via Self-Play and Reflective Experience Replay**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10589v1) [paper-pdf](https://arxiv.org/pdf/2601.10589v1)

**Authors**: Hao Wang, Yanting Wang, Hao Li, Rui Li, Lei Sha

**Abstract**: Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial ``jailbreak'' attacks designed to bypass safety guardrails. Current safety alignment methods depend heavily on static external red teaming, utilizing fixed defense prompts or pre-collected adversarial datasets. This leads to a rigid defense that overfits known patterns and fails to generalize to novel, sophisticated threats. To address this critical limitation, we propose empowering the model to be its own red teamer, capable of achieving autonomous and evolving adversarial attacks. Specifically, we introduce Safety Self- Play (SSP), a system that utilizes a single LLM to act concurrently as both the Attacker (generating jailbreaks) and the Defender (refusing harmful requests) within a unified Reinforcement Learning (RL) loop, dynamically evolving attack strategies to uncover vulnerabilities while simultaneously strengthening defense mechanisms. To ensure the Defender effectively addresses critical safety issues during the self-play, we introduce an advanced Reflective Experience Replay Mechanism, which uses an experience pool accumulated throughout the process. The mechanism employs a Upper Confidence Bound (UCB) sampling strategy to focus on failure cases with low rewards, helping the model learn from past hard mistakes while balancing exploration and exploitation. Extensive experiments demonstrate that our SSP approach autonomously evolves robust defense capabilities, significantly outperforming baselines trained on static adversarial datasets and establishing a new benchmark for proactive safety alignment.



## **7. Defending Large Language Models Against Jailbreak Attacks via In-Decoding Safety-Awareness Probing**

cs.AI

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10543v1) [paper-pdf](https://arxiv.org/pdf/2601.10543v1)

**Authors**: Yinzhi Zhao, Ming Wang, Shi Feng, Xiaocui Yang, Daling Wang, Yifei Zhang

**Abstract**: Large language models (LLMs) have achieved impressive performance across natural language tasks and are increasingly deployed in real-world applications. Despite extensive safety alignment efforts, recent studies show that such alignment is often shallow and remains vulnerable to jailbreak attacks. Existing defense mechanisms, including decoding-based constraints and post-hoc content detectors, struggle against sophisticated jailbreaks, often intervening robust detection or excessively degrading model utility. In this work, we examine the decoding process of LLMs and make a key observation: even when successfully jailbroken, models internally exhibit latent safety-related signals during generation. However, these signals are overridden by the model's drive for fluent continuation, preventing timely self-correction or refusal. Building on this observation, we propose a simple yet effective approach that explicitly surfaces and leverages these latent safety signals for early detection of unsafe content during decoding. Experiments across diverse jailbreak attacks demonstrate that our approach significantly enhances safety, while maintaining low over-refusal rates on benign inputs and preserving response quality. Our results suggest that activating intrinsic safety-awareness during decoding offers a promising and complementary direction for defending against jailbreak attacks. Code is available at: https://github.com/zyz13590/SafeProbing.



## **8. ReasAlign: Reasoning Enhanced Safety Alignment against Prompt Injection Attack**

cs.CR

15 pages, 10 figures

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10173v1) [paper-pdf](https://arxiv.org/pdf/2601.10173v1)

**Authors**: Hao Li, Yankai Yang, G. Edward Suh, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) have enabled the development of powerful agentic systems capable of automating complex workflows across various fields. However, these systems are highly vulnerable to indirect prompt injection attacks, where malicious instructions embedded in external data can hijack agent behavior. In this work, we present ReasAlign, a model-level solution to improve safety alignment against indirect prompt injection attacks. The core idea of ReasAlign is to incorporate structured reasoning steps to analyze user queries, detect conflicting instructions, and preserve the continuity of the user's intended tasks to defend against indirect injection attacks. To further ensure reasoning logic and accuracy, we introduce a test-time scaling mechanism with a preference-optimized judge model that scores reasoning steps and selects the best trajectory. Comprehensive evaluations across various benchmarks show that ReasAlign maintains utility comparable to an undefended model while consistently outperforming Meta SecAlign, the strongest prior guardrail. On the representative open-ended CyberSecEval2 benchmark, which includes multiple prompt-injected tasks, ReasAlign achieves 94.6% utility and only 3.6% ASR, far surpassing the state-of-the-art defensive model of Meta SecAlign (56.4% utility and 74.4% ASR). These results demonstrate that ReasAlign achieves the best trade-off between security and utility, establishing a robust and practical defense against prompt injection attacks in real-world agentic systems. Our code and experimental results could be found at https://github.com/leolee99/ReasAlign.



## **9. Understanding and Preserving Safety in Fine-Tuned LLMs**

cs.LG

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10141v1) [paper-pdf](https://arxiv.org/pdf/2601.10141v1)

**Authors**: Jiawen Zhang, Yangfan Hu, Kejia Chen, Lipeng He, Jiachen Ma, Jian Lou, Dan Li, Jian Liu, Xiaohu Yang, Ruoxi Jia

**Abstract**: Fine-tuning is an essential and pervasive functionality for applying large language models (LLMs) to downstream tasks. However, it has the potential to substantially degrade safety alignment, e.g., by greatly increasing susceptibility to jailbreak attacks, even when the fine-tuning data is entirely harmless. Despite garnering growing attention in defense efforts during the fine-tuning stage, existing methods struggle with a persistent safety-utility dilemma: emphasizing safety compromises task performance, whereas prioritizing utility typically requires deep fine-tuning that inevitably leads to steep safety declination.   In this work, we address this dilemma by shedding new light on the geometric interaction between safety- and utility-oriented gradients in safety-aligned LLMs. Through systematic empirical analysis, we uncover three key insights: (I) safety gradients lie in a low-rank subspace, while utility gradients span a broader high-dimensional space; (II) these subspaces are often negatively correlated, causing directional conflicts during fine-tuning; and (III) the dominant safety direction can be efficiently estimated from a single sample. Building upon these novel insights, we propose safety-preserving fine-tuning (SPF), a lightweight approach that explicitly removes gradient components conflicting with the low-rank safety subspace. Theoretically, we show that SPF guarantees utility convergence while bounding safety drift. Empirically, SPF consistently maintains downstream task performance and recovers nearly all pre-trained safety alignment, even under adversarial fine-tuning scenarios. Furthermore, SPF exhibits robust resistance to both deep fine-tuning and dynamic jailbreak attacks. Together, our findings provide new mechanistic understanding and practical guidance toward always-aligned LLM fine-tuning.



## **10. Privacy Enhanced PEFT: Tensor Train Decomposition Improves Privacy Utility Tradeoffs under DP-SGD**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10045v1) [paper-pdf](https://arxiv.org/pdf/2601.10045v1)

**Authors**: Pradip Kunwar, Minh Vu, Maanak Gupta, Manish Bhattarai

**Abstract**: Fine-tuning large language models on sensitive data poses significant privacy risks, as membership inference attacks can reveal whether individual records were used during training. While Differential Privacy (DP) provides formal protection, applying DP to conventional Parameter-Efficient Fine-Tuning (PEFT) methods such as Low-Rank Adaptation (LoRA) often incurs substantial utility loss. In this work, we show that a more structurally constrained PEFT architecture, Tensor Train Low-Rank Adaptation (TTLoRA), can improve the privacy-utility tradeoff by shrinking the effective parameter space while preserving expressivity. To this end, we develop TTLoRA-DP, a differentially private training framework for TTLoRA. Specifically, we extend the ghost clipping algorithm to Tensor Train cores via cached contraction states, enabling efficient Differentially Private Stochastic Gradient Descent (DP-SGD) with exact per-example gradient norm computation without materializing full per-example gradients. Experiments on GPT-2 fine-tuning over the Enron and Penn Treebank datasets show that TTLoRA-DP consistently strengthens privacy protection relative to LoRA-DP while maintaining comparable or better downstream utility. Moreover, TTLoRA exhibits lower membership leakage even without DP training, using substantially smaller adapters and requiring on average 7.6X fewer parameters than LoRA. Overall, our results demonstrate that TTLoRA offers a practical path to improving the privacy-utility tradeoff in parameter-efficient language model adaptation.



## **11. SoK: Privacy-aware LLM in Healthcare: Threat Model, Privacy Techniques, Challenges and Recommendations**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10004v1) [paper-pdf](https://arxiv.org/pdf/2601.10004v1)

**Authors**: Mohoshin Ara Tahera, Karamveer Singh Sidhu, Shuvalaxmi Dass, Sajal Saha

**Abstract**: Large Language Models (LLMs) are increasingly adopted in healthcare to support clinical decision-making, summarize electronic health records (EHRs), and enhance patient care. However, this integration introduces significant privacy and security challenges, driven by the sensitivity of clinical data and the high-stakes nature of medical workflows. These risks become even more pronounced across heterogeneous deployment environments, ranging from small on-premise hospital systems to regional health networks, each with unique resource limitations and regulatory demands. This Systematization of Knowledge (SoK) examines the evolving threat landscape across the three core LLM phases: Data preprocessing, Fine-tuning, and Inference within realistic healthcare settings. We present a detailed threat model that characterizes adversaries, capabilities, and attack surfaces at each phase, and we systematize how existing privacy-preserving techniques (PPTs) attempt to mitigate these vulnerabilities. While existing defenses show promise, our analysis identifies persistent limitations in securing sensitive clinical data across diverse operational tiers. We conclude with phase-aware recommendations and future research directions aimed at strengthening privacy guarantees for LLMs in regulated environments. This work provides a foundation for understanding the intersection of LLMs, threats, and privacy in healthcare, offering a roadmap toward more robust and clinically trustworthy AI systems.



## **12. The Promptware Kill Chain: How Prompt Injections Gradually Evolved Into a Multi-Step Malware**

cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09625v1) [paper-pdf](https://arxiv.org/pdf/2601.09625v1)

**Authors**: Ben Nassi, Bruce Schneier, Oleg Brodt

**Abstract**: The rapid adoption of large language model (LLM)-based systems -- from chatbots to autonomous agents capable of executing code and financial transactions -- has created a new attack surface that existing security frameworks inadequately address. The dominant framing of these threats as "prompt injection" -- a catch-all phrase for security failures in LLM-based systems -- obscures a more complex reality: Attacks on LLM-based systems increasingly involve multi-step sequences that mirror traditional malware campaigns. In this paper, we propose that attacks targeting LLM-based applications constitute a distinct class of malware, which we term \textit{promptware}, and introduce a five-step kill chain model for analyzing these threats. The framework comprises Initial Access (prompt injection), Privilege Escalation (jailbreaking), Persistence (memory and retrieval poisoning), Lateral Movement (cross-system and cross-user propagation), and Actions on Objective (ranging from data exfiltration to unauthorized transactions). By mapping recent attacks to this structure, we demonstrate that LLM-related attacks follow systematic sequences analogous to traditional malware campaigns. The promptware kill chain offers security practitioners a structured methodology for threat modeling and provides a common vocabulary for researchers across AI safety and cybersecurity to address a rapidly evolving threat landscape.



## **13. SpatialJB: How Text Distribution Art Becomes the "Jailbreak Key" for LLM Guardrails**

cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09321v1) [paper-pdf](https://arxiv.org/pdf/2601.09321v1)

**Authors**: Zhiyi Mou, Jingyuan Yang, Zeheng Qian, Wangze Ni, Tianfang Xiao, Ning Liu, Chen Zhang, Zhan Qin, Kui Ren

**Abstract**: While Large Language Models (LLMs) have powerful capabilities, they remain vulnerable to jailbreak attacks, which is a critical barrier to their safe web real-time application. Current commercial LLM providers deploy output guardrails to filter harmful outputs, yet these defenses are not impenetrable. Due to LLMs' reliance on autoregressive, token-by-token inference, their semantic representations lack robustness to spatially structured perturbations, such as redistributing tokens across different rows, columns, or diagonals. Exploiting the Transformer's spatial weakness, we propose SpatialJB to disrupt the model's output generation process, allowing harmful content to bypass guardrails without detection. Comprehensive experiments conducted on leading LLMs get nearly 100% ASR, demonstrating the high effectiveness of SpatialJB. Even after adding advanced output guardrails, like the OpenAI Moderation API, SpatialJB consistently maintains a success rate exceeding 75%, outperforming current jailbreak techniques by a significant margin. The proposal of SpatialJB exposes a key weakness in current guardrails and emphasizes the importance of spatial semantics, offering new insights to advance LLM safety research. To prevent potential misuse, we also present baseline defense strategies against SpatialJB and evaluate their effectiveness in mitigating such attacks. The code for the attack, baseline defenses, and a demo are available at https://anonymous.4open.science/r/SpatialJailbreak-8E63.



## **14. STaR: Sensitive Trajectory Regulation for Unlearning in Large Reasoning Models**

cs.AI

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09281v1) [paper-pdf](https://arxiv.org/pdf/2601.09281v1)

**Authors**: Jingjing Zhou, Gaoxiang Cong, Li Su, Liang Li

**Abstract**: Large Reasoning Models (LRMs) have advanced automated multi-step reasoning, but their ability to generate complex Chain-of-Thought (CoT) trajectories introduces severe privacy risks, as sensitive information may be deeply embedded throughout the reasoning process. Existing Large Language Models (LLMs) unlearning approaches that typically focus on modifying only final answers are insufficient for LRMs, as they fail to remove sensitive content from intermediate steps, leading to persistent privacy leakage and degraded security. To address these challenges, we propose Sensitive Trajectory Regulation (STaR), a parameter-free, inference-time unlearning framework that achieves robust privacy protection throughout the reasoning process. Specifically, we first identify sensitive content via semantic-aware detection. Then, we inject global safety constraints through secure prompt prefix. Next, we perform trajectory-aware suppression to dynamically block sensitive content across the entire reasoning chain. Finally, we apply token-level adaptive filtering to prevent both exact and paraphrased sensitive tokens during generation. Furthermore, to overcome the inadequacies of existing evaluation protocols, we introduce two metrics: Multi-Decoding Consistency Assessment (MCS), which measures the consistency of unlearning across diverse decoding strategies, and Multi-Granularity Membership Inference Attack (MIA) Evaluation, which quantifies privacy protection at both answer and reasoning-chain levels. Experiments on the R-TOFU benchmark demonstrate that STaR achieves comprehensive and stable unlearning with minimal utility loss, setting a new standard for privacy-preserving reasoning in LRMs.



## **15. KryptoPilot: An Open-World Knowledge-Augmented LLM Agent for Automated Cryptographic Exploitation**

cs.CR

14 Pages,4 figures

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09129v1) [paper-pdf](https://arxiv.org/pdf/2601.09129v1)

**Authors**: Xiaonan Liu, Zhihao Li, Xiao Lan, Hao Ren, Haizhou Wang, Xingshu Chen

**Abstract**: Capture-the-Flag (CTF) competitions play a central role in modern cybersecurity as a platform for training practitioners and evaluating offensive and defensive techniques derived from real-world vulnerabilities. Despite recent advances in large language models (LLMs), existing LLM-based agents remain ineffective on high-difficulty cryptographic CTF challenges, which require precise cryptanalytic knowledge, stable long-horizon reasoning, and disciplined interaction with specialized toolchains. Through a systematic exploratory study, we show that insufficient knowledge granularity, rather than model reasoning capacity, is a primary factor limiting successful cryptographic exploitation: coarse or abstracted external knowledge often fails to support correct attack modeling and implementation. Motivated by this observation, we propose KryptoPilot, an open-world knowledge-augmented LLM agent for automated cryptographic exploitation. KryptoPilot integrates dynamic open-world knowledge acquisition via a Deep Research pipeline, a persistent workspace for structured knowledge reuse, and a governance subsystem that stabilizes reasoning through behavioral constraints and cost-aware model routing. This design enables precise knowledge alignment while maintaining efficient reasoning across heterogeneous subtasks. We evaluate KryptoPilot on two established CTF benchmarks and in six real-world CTF competitions. KryptoPilot achieves a complete solve rate on InterCode-CTF, solves between 56 and 60 percent of cryptographic challenges on the NYU-CTF benchmark, and successfully solves 26 out of 33 cryptographic challenges in live competitions, including multiple earliest-solved and uniquely-solved instances. These results demonstrate the necessity of open-world, fine-grained knowledge augmentation and governed reasoning for scaling LLM-based agents to real-world cryptographic exploitation.



## **16. Too Helpful to Be Safe: User-Mediated Attacks on Planning and Web-Use Agents**

cs.CR

Keywords: LLM Agents; User-Mediated Attack; Agent Security; Human Factors in Cybersecurity; Web-Use Agents; Planning Agents; Benchmark

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.10758v1) [paper-pdf](https://arxiv.org/pdf/2601.10758v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Large Language Models (LLMs) have enabled agents to move beyond conversation toward end-to-end task execution and become more helpful. However, this helpfulness introduces new security risks stem less from direct interface abuse than from acting on user-provided content. Existing studies on agent security largely focus on model-internal vulnerabilities or adversarial access to agent interfaces, overlooking attacks that exploit users as unintended conduits. In this paper, we study user-mediated attacks, where benign users are tricked into relaying untrusted or attacker-controlled content to agents, and analyze how commercial LLM agents respond under such conditions. We conduct a systematic evaluation of 12 commercial agents in a sandboxed environment, covering 6 trip-planning agents and 6 web-use agents, and compare agent behavior across scenarios with no, soft, and hard user-requested safety checks. Our results show that agents are too helpful to be safe by default. Without explicit safety requests, trip-planning agents bypass safety constraints in over 92% of cases, converting unverified content into confident booking guidance. Web-use agents exhibit near-deterministic execution of risky actions, with 9 out of 17 supported tests reaching a 100% bypass rate. Even when users express soft or hard safety intent, constraint bypass remains substantial, reaching up to 54.7% and 7% for trip-planning agents, respectively. These findings reveal that the primary issue is not a lack of safety capability, but its prioritization. Agents invoke safety checks only conditionally when explicitly prompted, and otherwise default to goal-driven execution. Moreover, agents lack clear task boundaries and stopping rules, frequently over-executing workflows in ways that lead to unnecessary data disclosure and real-world harm.



## **17. Integrating APK Image and Text Data for Enhanced Threat Detection: A Multimodal Deep Learning Approach to Android Malware**

cs.CR

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08959v1) [paper-pdf](https://arxiv.org/pdf/2601.08959v1)

**Authors**: Md Mashrur Arifin, Maqsudur Rahman, Nasir U. Eisty

**Abstract**: As zero-day Android malware attacks grow more sophisticated, recent research highlights the effectiveness of using image-based representations of malware bytecode to detect previously unseen threats. However, existing studies often overlook how image type and resolution affect detection and ignore valuable textual data in Android Application Packages (APKs), such as permissions and metadata, limiting their ability to fully capture malicious behavior. The integration of multimodality, which combines image and text data, has gained momentum as a promising approach to address these limitations. This paper proposes a multimodal deep learning framework integrating APK images and textual features to enhance Android malware detection. We systematically evaluate various image types and resolutions across different Convolutional Neural Networks (CNN) architectures, including VGG, ResNet-152, MobileNet, DenseNet, EfficientNet-B4, and use LLaMA-2, a large language model, to extract and annotate textual features for improved analysis. The findings demonstrate that RGB images at higher resolutions (e.g., 256x256, 512x512) achieve superior classification performance, while the multimodal integration of image and text using the CLIP model reveals limited potential. Overall, this research highlights the importance of systematically evaluating image attributes and integrating multimodal data to develop effective malware detection for Android systems.



## **18. Robust CAPTCHA Using Audio Illusions in the Era of Large Language Models: from Evaluation to Advances**

cs.SD

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08516v1) [paper-pdf](https://arxiv.org/pdf/2601.08516v1)

**Authors**: Ziqi Ding, Yunfeng Wan, Wei Song, Yi Liu, Gelei Deng, Nan Sun, Huadong Mo, Jingling Xue, Shidong Pan, Yuekang Li

**Abstract**: CAPTCHAs are widely used by websites to block bots and spam by presenting challenges that are easy for humans but difficult for automated programs to solve. To improve accessibility, audio CAPTCHAs are designed to complement visual ones. However, the robustness of audio CAPTCHAs against advanced Large Audio Language Models (LALMs) and Automatic Speech Recognition (ASR) models remains unclear.   In this paper, we introduce AI-CAPTCHA, a unified framework that offers (i) an evaluation framework, ACEval, which includes advanced LALM- and ASR-based solvers, and (ii) a novel audio CAPTCHA approach, IllusionAudio, leveraging audio illusions. Through extensive evaluations of seven widely deployed audio CAPTCHAs, we show that most existing methods can be solved with high success rates by advanced LALMs and ASR models, exposing critical security weaknesses.   To address these vulnerabilities, we design a new audio CAPTCHA approach, IllusionAudio, which exploits perceptual illusion cues rooted in human auditory mechanisms. Extensive experiments demonstrate that our method defeats all tested LALM- and ASR-based attacks while achieving a 100% human pass rate, significantly outperforming existing audio CAPTCHA methods.



## **19. Evaluating Role-Consistency in LLMs for Counselor Training**

cs.CL

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08892v1) [paper-pdf](https://arxiv.org/pdf/2601.08892v1)

**Authors**: Eric Rudolph, Natalie Engert, Jens Albrecht

**Abstract**: The rise of online counseling services has highlighted the need for effective training methods for future counselors. This paper extends research on VirCo, a Virtual Client for Online Counseling, designed to complement traditional role-playing methods in academic training by simulating realistic client interactions. Building on previous work, we introduce a new dataset incorporating adversarial attacks to test the ability of large language models (LLMs) to maintain their assigned roles (role-consistency). The study focuses on evaluating the role consistency and coherence of the Vicuna model's responses, comparing these findings with earlier research. Additionally, we assess and compare various open-source LLMs for their performance in sustaining role consistency during virtual client interactions. Our contributions include creating an adversarial dataset, evaluating conversation coherence and persona consistency, and providing a comparative analysis of different LLMs.



## **20. BenchOverflow: Measuring Overflow in Large Language Models via Plain-Text Prompts**

cs.CL

Accepted at TMLR 2026

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08490v1) [paper-pdf](https://arxiv.org/pdf/2601.08490v1)

**Authors**: Erin Feiglin, Nir Hutnik, Raz Lapid

**Abstract**: We investigate a failure mode of large language models (LLMs) in which plain-text prompts elicit excessive outputs, a phenomenon we term Overflow. Unlike jailbreaks or prompt injection, Overflow arises under ordinary interaction settings and can lead to elevated serving cost, latency, and cross-user performance degradation, particularly when scaled across many requests. Beyond usability, the stakes are economic and environmental: unnecessary tokens increase per-request cost and energy consumption, compounding into substantial operational spend and carbon footprint at scale. Moreover, Overflow represents a practical vector for compute amplification and service degradation in shared environments. We introduce BenchOverflow, a model-agnostic benchmark of nine plain-text prompting strategies that amplify output volume without adversarial suffixes or policy circumvention. Using a standardized protocol with a fixed budget of 5000 new tokens, we evaluate nine open- and closed-source models and observe pronounced rightward shifts and heavy tails in length distributions. Cap-saturation rates (CSR@1k/3k/5k) and empirical cumulative distribution functions (ECDFs) quantify tail risk; within-prompt variance and cross-model correlations show that Overflow is broadly reproducible yet heterogeneous across families and attack vectors. A lightweight mitigation-a fixed conciseness reminder-attenuates right tails and lowers CSR for all strategies across the majority of models. Our findings position length control as a measurable reliability, cost, and sustainability concern rather than a stylistic quirk. By enabling standardized comparison of length-control robustness across models, BenchOverflow provides a practical basis for selecting deployments that minimize resource waste and operating expense, and for evaluating defenses that curb compute amplification without eroding task performance.



## **21. DNF: Dual-Layer Nested Fingerprinting for Large Language Model Intellectual Property Protection**

cs.CR

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08223v1) [paper-pdf](https://arxiv.org/pdf/2601.08223v1)

**Authors**: Zhenhua Xu, Yiran Zhao, Mengting Zhong, Dezhang Kong, Changting Lin, Tong Qiao, Meng Han

**Abstract**: The rapid growth of large language models raises pressing concerns about intellectual property protection under black-box deployment. Existing backdoor-based fingerprints either rely on rare tokens -- leading to high-perplexity inputs susceptible to filtering -- or use fixed trigger-response mappings that are brittle to leakage and post-hoc adaptation. We propose \textsc{Dual-Layer Nested Fingerprinting} (DNF), a black-box method that embeds a hierarchical backdoor by coupling domain-specific stylistic cues with implicit semantic triggers. Across Mistral-7B, LLaMA-3-8B-Instruct, and Falcon3-7B-Instruct, DNF achieves perfect fingerprint activation while preserving downstream utility. Compared with existing methods, it uses lower-perplexity triggers, remains undetectable under fingerprint detection attacks, and is relatively robust to incremental fine-tuning and model merging. These results position DNF as a practical, stealthy, and resilient solution for LLM ownership verification and intellectual property protection.



## **22. Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety**

cs.AI

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.08000v1) [paper-pdf](https://arxiv.org/pdf/2601.08000v1)

**Authors**: Can Jin, Rui Wu, Tong Che, Qixin Zhang, Hongwu Peng, Jiahui Zhao, Zhenting Wang, Wenqi Wei, Ligong Han, Zhao Zhang, Yuan Cao, Ruixiang Tang, Dimitris N. Metaxas

**Abstract**: Ensuring that Large Language Models (LLMs) adhere to safety principles without refusing benign requests remains a significant challenge. While OpenAI introduces deliberative alignment (DA) to enhance the safety of its o-series models through reasoning over detailed ``code-like'' safety rules, the effectiveness of this approach in open-source LLMs, which typically lack advanced reasoning capabilities, is understudied. In this work, we systematically evaluate the impact of explicitly specifying extensive safety codes versus demonstrating them through illustrative cases. We find that referencing explicit codes inconsistently improves harmlessness and systematically degrades helpfulness, whereas training on case-augmented simple codes yields more robust and generalized safety behaviors. By guiding LLMs with case-augmented reasoning instead of extensive code-like safety rules, we avoid rigid adherence to narrowly enumerated rules and enable broader adaptability. Building on these insights, we propose CADA, a case-augmented deliberative alignment method for LLMs utilizing reinforcement learning on self-generated safety reasoning chains. CADA effectively enhances harmlessness, improves robustness against attacks, and reduces over-refusal while preserving utility across diverse benchmarks, offering a practical alternative to rule-only DA for improving safety while maintaining helpfulness.



## **23. SecureCAI: Injection-Resilient LLM Assistants for Cybersecurity Operations**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07835v1) [paper-pdf](https://arxiv.org/pdf/2601.07835v1)

**Authors**: Mohammed Himayath Ali, Mohammed Aqib Abdullah, Mohammed Mudassir Uddin, Shahnawaz Alam

**Abstract**: Large Language Models have emerged as transformative tools for Security Operations Centers, enabling automated log analysis, phishing triage, and malware explanation; however, deployment in adversarial cybersecurity environments exposes critical vulnerabilities to prompt injection attacks where malicious instructions embedded in security artifacts manipulate model behavior. This paper introduces SecureCAI, a novel defense framework extending Constitutional AI principles with security-aware guardrails, adaptive constitution evolution, and Direct Preference Optimization for unlearning unsafe response patterns, addressing the unique challenges of high-stakes security contexts where traditional safety mechanisms prove insufficient against sophisticated adversarial manipulation. Experimental evaluation demonstrates that SecureCAI reduces attack success rates by 94.7% compared to baseline models while maintaining 95.1% accuracy on benign security analysis tasks, with the framework incorporating continuous red-teaming feedback loops enabling dynamic adaptation to emerging attack strategies and achieving constitution adherence scores exceeding 0.92 under sustained adversarial pressure, thereby establishing a foundation for trustworthy integration of language model capabilities into operational cybersecurity workflows and addressing a critical gap in current approaches to AI safety within adversarial domains.



## **24. A Visual Semantic Adaptive Watermark grounded by Prefix-Tuning for Large Vision-Language Model**

cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07291v1) [paper-pdf](https://arxiv.org/pdf/2601.07291v1)

**Authors**: Qi Zheng, Shuliang Liu, Yu Huang, Sihang Jia, Jungang Li, Lyuhao Chen, Junhao Chen, Hanqian Li, Aiwei Liu, Yibo Yan, Xuming Hu

**Abstract**: Watermarking has emerged as a pivotal solution for content traceability and intellectual property protection in Large Vision-Language Models (LVLMs). However, vision-agnostic watermarks introduce visually irrelevant tokens and disrupt visual grounding by enforcing indiscriminate pseudo-random biases, while some semantic-aware methods incur prohibitive inference latency due to rejection sampling. In this paper, we propose the VIsual Semantic Adaptive Watermark (VISA-Mark), a novel framework that embeds detectable signals while strictly preserving visual fidelity. Our approach employs a lightweight, efficiently trained prefix-tuner to extract dynamic Visual-Evidence Weights, which quantify the evidentiary support for candidate tokens based on the visual input. These weights guide an adaptive vocabulary partitioning and logits perturbation mechanism, concentrating watermark strength specifically on visually-supported tokens. By actively aligning the watermark with visual evidence, VISA-Mark effectively maintains visual fidelity. Empirical results confirm that VISA-Mark outperforms conventional methods with a 7.8% improvement in visual consistency (Chair-I) and superior semantic fidelity. The framework maintains highly competitive detection accuracy (96.88% AUC) and robust attack resilience (99.3%) without sacrificing inference efficiency, effectively establishing a new standard for reliability-preserving multimodal watermarking.



## **25. When Bots Take the Bait: Exposing and Mitigating the Emerging Social Engineering Attack in Web Automation Agent**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07263v1) [paper-pdf](https://arxiv.org/pdf/2601.07263v1)

**Authors**: Xinyi Wu, Geng Hong, Yueyue Chen, MingXuan Liu, Feier Jin, Xudong Pan, Jiarun Dai, Baojun Liu

**Abstract**: Web agents, powered by large language models (LLMs), are increasingly deployed to automate complex web interactions. The rise of open-source frameworks (e.g., Browser Use, Skyvern-AI) has accelerated adoption, but also broadened the attack surface. While prior research has focused on model threats such as prompt injection and backdoors, the risks of social engineering remain largely unexplored. We present the first systematic study of social engineering attacks against web automation agents and design a pluggable runtime mitigation solution. On the attack side, we introduce the AgentBait paradigm, which exploits intrinsic weaknesses in agent execution: inducement contexts can distort the agent's reasoning and steer it toward malicious objectives misaligned with the intended task. On the defense side, we propose SUPERVISOR, a lightweight runtime module that enforces environment and intention consistency alignment between webpage context and intended goals to mitigate unsafe operations before execution.   Empirical results show that mainstream frameworks are highly vulnerable to AgentBait, with an average attack success rate of 67.5% and peaks above 80% under specific strategies (e.g., trusted identity forgery). Compared with existing lightweight defenses, our module can be seamlessly integrated across different web automation frameworks and reduces attack success rates by up to 78.1% on average while incurring only a 7.7% runtime overhead and preserving usability. This work reveals AgentBait as a critical new threat surface for web agents and establishes a practical, generalizable defense, advancing the security of this rapidly emerging ecosystem. We reported the details of this attack to the framework developers and received acknowledgment before submission.



## **26. Defenses Against Prompt Attacks Learn Surface Heuristics**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07185v1) [paper-pdf](https://arxiv.org/pdf/2601.07185v1)

**Authors**: Shawn Li, Chenxiao Yu, Zhiyu Ni, Hao Li, Charith Peris, Chaowei Xiao, Yue Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in security-sensitive applications, where they must follow system- or developer-specified instructions that define the intended task behavior, while completing benign user requests. When adversarial instructions appear in user queries or externally retrieved content, models may override intended logic. Recent defenses rely on supervised fine-tuning with benign and malicious labels. Although these methods achieve high attack rejection rates, we find that they rely on narrow correlations in defense data rather than harmful intent, leading to systematic rejection of safe inputs. We analyze three recurring shortcut behaviors induced by defense fine-tuning. \emph{Position bias} arises when benign content placed later in a prompt is rejected at much higher rates; across reasoning benchmarks, suffix-task rejection rises from below \textbf{10\%} to as high as \textbf{90\%}. \emph{Token trigger bias} occurs when strings common in attack data raise rejection probability even in benign contexts; inserting a single trigger token increases false refusals by up to \textbf{50\%}. \emph{Topic generalization bias} reflects poor generalization beyond the defense data distribution, with defended models suffering test-time accuracy drops of up to \textbf{40\%}. These findings suggest that current prompt-injection defenses frequently respond to attack-like surface patterns rather than the underlying intent. We introduce controlled diagnostic datasets and a systematic evaluation across two base models and multiple defense pipelines, highlighting limitations of supervised fine-tuning for reliable LLM security.



## **27. Safe-FedLLM: Delving into the Safety of Federated Large Language Models**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07177v1) [paper-pdf](https://arxiv.org/pdf/2601.07177v1)

**Authors**: Mingxiang Tao, Yu Tian, Wenxuan Tu, Yue Yang, Xue Yang, Xiangyan Tang

**Abstract**: Federated learning (FL) addresses data privacy and silo issues in large language models (LLMs). Most prior work focuses on improving the training efficiency of federated LLMs. However, security in open environments is overlooked, particularly defenses against malicious clients. To investigate the safety of LLMs during FL, we conduct preliminary experiments to analyze potential attack surfaces and defensible characteristics from the perspective of Low-Rank Adaptation (LoRA) weights. We find two key properties of FL: 1) LLMs are vulnerable to attacks from malicious clients in FL, and 2) LoRA weights exhibit distinct behavioral patterns that can be filtered through simple classifiers. Based on these properties, we propose Safe-FedLLM, a probe-based defense framework for federated LLMs, constructing defenses across three dimensions: Step-Level, Client-Level, and Shadow-Level. The core concept of Safe-FedLLM is to perform probe-based discrimination on the LoRA weights locally trained by each client during FL, treating them as high-dimensional behavioral features and using lightweight classification models to determine whether they possess malicious attributes. Extensive experiments demonstrate that Safe-FedLLM effectively enhances the defense capability of federated LLMs without compromising performance on benign data. Notably, our method effectively suppresses malicious data impact without significant impact on training speed, and remains effective even with many malicious clients. Our code is available at: https://github.com/dmqx/Safe-FedLLM.



## **28. Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07122v1) [paper-pdf](https://arxiv.org/pdf/2601.07122v1)

**Authors**: Yixiao Peng, Hao Hu, Feiyang Li, Xinye Cao, Yingchang Jiang, Jipeng Tang, Guoshun Nan, Yuling Liu

**Abstract**: While virtualization and resource pooling empower cloud networks with structural flexibility and elastic scalability, they inevitably expand the attack surface and challenge cyber resilience. Reinforcement Learning (RL)-based defense strategies have been developed to optimize resource deployment and isolation policies under adversarial conditions, aiming to enhance system resilience by maintaining and restoring network availability. However, existing approaches lack robustness as they require retraining to adapt to dynamic changes in network structure, node scale, attack strategies, and attack intensity. Furthermore, the lack of Human-in-the-Loop (HITL) support limits interpretability and flexibility. To address these limitations, we propose CyberOps-Bots, a hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs). Inspired by MITRE ATT&CK's Tactics-Techniques model, CyberOps-Bots features a two-layer architecture: (1) An upper-level LLM agent with four modules--ReAct planning, IPDRR-based perception, long-short term memory, and action/tool integration--performs global awareness, human intent recognition, and tactical planning; (2) Lower-level RL agents, developed via heterogeneous separated pre-training, execute atomic defense actions within localized network regions. This synergy preserves LLM adaptability and interpretability while ensuring reliable RL execution. Experiments on real cloud datasets show that, compared to state-of-the-art algorithms, CyberOps-Bots maintains network availability 68.5% higher and achieves a 34.7% jumpstart performance gain when shifting the scenarios without retraining. To our knowledge, this is the first study to establish a robust LLM-RL framework with HITL support for cloud defense. We will release our framework to the community, facilitating the advancement of robust and autonomous defense in cloud networks.



## **29. Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems**

cs.CR

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.07072v1) [paper-pdf](https://arxiv.org/pdf/2601.07072v1)

**Authors**: Hongyan Chang, Ergute Bao, Xinjian Luo, Ting Yu

**Abstract**: Large language models (LLMs) increasingly rely on retrieving information from external corpora. This creates a new attack surface: indirect prompt injection (IPI), where hidden instructions are planted in the corpora and hijack model behavior once retrieved. Previous studies have highlighted this risk but often avoid the hardest step: ensuring that malicious content is actually retrieved. In practice, unoptimized IPI is rarely retrieved under natural queries, which leaves its real-world impact unclear.   We address this challenge by decomposing the malicious content into a trigger fragment that guarantees retrieval and an attack fragment that encodes arbitrary attack objectives. Based on this idea, we design an efficient and effective black-box attack algorithm that constructs a compact trigger fragment to guarantee retrieval for any attack fragment. Our attack requires only API access to embedding models, is cost-efficient (as little as $0.21 per target user query on OpenAI's embedding models), and achieves near-100% retrieval across 11 benchmarks and 8 embedding models (including both open-source models and proprietary services).   Based on this attack, we present the first end-to-end IPI exploits under natural queries and realistic external corpora, spanning both RAG and agentic systems with diverse attack objectives. These results establish IPI as a practical and severe threat: when a user issued a natural query to summarize emails on frequently asked topics, a single poisoned email was sufficient to coerce GPT-4o into exfiltrating SSH keys with over 80% success in a multi-agent workflow. We further evaluate several defenses and find that they are insufficient to prevent the retrieval of malicious text, highlighting retrieval as a critical open vulnerability.



## **30. PenForge: On-the-Fly Expert Agent Construction for Automated Penetration Testing**

cs.SE

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.06910v1) [paper-pdf](https://arxiv.org/pdf/2601.06910v1)

**Authors**: Huihui Huang, Jieke Shi, Junkai Chen, Ting Zhang, Yikun Li, Chengran Yang, Eng Lieh Ouh, Lwin Khin Shar, David Lo

**Abstract**: Penetration testing is essential for identifying vulnerabilities in web applications before real adversaries can exploit them. Recent work has explored automating this process with Large Language Model (LLM)-powered agents, but existing approaches either rely on a single generic agent that struggles in complex scenarios or narrowly specialized agents that cannot adapt to diverse vulnerability types. We therefore introduce PenForge, a framework that dynamically constructs expert agents during testing rather than relying on those prepared beforehand. By integrating automated reconnaissance of potential attack surfaces with agents instantiated on the fly for context-aware exploitation, PenForge achieves a 30.0% exploit success rate (12/40) on CVE-Bench in the particularly challenging zero-day setting, which is a 3 times improvement over the state-of-the-art. Our analysis also identifies three opportunities for future work: (1) supplying richer tool-usage knowledge to improve exploitation effectiveness; (2) extending benchmarks to include more vulnerabilities and attack types; and (3) fostering developer trust by incorporating explainable mechanisms and human review. As an emerging result with substantial potential impact, PenForge embodies the early-stage yet paradigm-shifting idea of on-the-fly agent construction, marking its promise as a step toward scalable and effective LLM-driven penetration testing.



## **31. Paraphrasing Adversarial Attack on LLM-as-a-Reviewer**

cs.CL

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.06884v1) [paper-pdf](https://arxiv.org/pdf/2601.06884v1)

**Authors**: Masahiro Kaneko

**Abstract**: The use of large language models (LLMs) in peer review systems has attracted growing attention, making it essential to examine their potential vulnerabilities. Prior attacks rely on prompt injection, which alters manuscript content and conflates injection susceptibility with evaluation robustness. We propose the Paraphrasing Adversarial Attack (PAA), a black-box optimization method that searches for paraphrased sequences yielding higher review scores while preserving semantic equivalence and linguistic naturalness. PAA leverages in-context learning, using previous paraphrases and their scores to guide candidate generation. Experiments across five ML and NLP conferences with three LLM reviewers and five attacking models show that PAA consistently increases review scores without changing the paper's claims. Human evaluation confirms that generated paraphrases maintain meaning and naturalness. We also find that attacked papers exhibit increased perplexity in reviews, offering a potential detection signal, and that paraphrasing submissions can partially mitigate attacks.



## **32. CHASE: LLM Agents for Dissecting Malicious PyPI Packages**

cs.CR

Accepted for publication and presented at the 2nd IEEE International Conference on AI-powered Software (AIware 2025). 10 pages, 3 figures

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.06838v1) [paper-pdf](https://arxiv.org/pdf/2601.06838v1)

**Authors**: Takaaki Toda, Tatsuya Mori

**Abstract**: Modern software package registries like PyPI have become critical infrastructure for software development, but are increasingly exploited by threat actors distributing malicious packages with sophisticated multi-stage attack chains. While Large Language Models (LLMs) offer promising capabilities for automated code analysis, their application to security-critical malware detection faces fundamental challenges, including hallucination and context confusion, which can lead to missed detections or false alarms. We present CHASE (Collaborative Hierarchical Agents for Security Exploration), a high-reliability multi-agent architecture that addresses these limitations through a Plan-and-Execute coordination model, specialized Worker Agents focused on specific analysis aspects, and integration with deterministic security tools for critical operations. Our key insight is that reliability in LLM-based security analysis emerges not from improving individual model capabilities but from architecting systems that compensate for LLM weaknesses while leveraging their semantic understanding strengths. Evaluation on a dataset of 3,000 packages (500 malicious, 2,500 benign) demonstrates that CHASE achieves 98.4% recall with only 0.08% false positive rate, while maintaining a practical median analysis time of 4.5 minutes per package, making it suitable for operational deployment in automated package screening. Furthermore, we conducted a survey with cybersecurity professionals to evaluate the generated analysis reports, identifying their key strengths and areas for improvement. This work provides a blueprint for building reliable AI-powered security tools that can scale with the growing complexity of modern software supply chains. Our project page is available at https://t0d4.github.io/CHASE-AIware25/



## **33. CyberLLM-FINDS 2025: Instruction-Tuned Fine-tuning of Domain-Specific LLMs with Retrieval-Augmented Generation and Graph Integration for MITRE Evaluation**

cs.CR

12 pages

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.06779v1) [paper-pdf](https://arxiv.org/pdf/2601.06779v1)

**Authors**: Vasanth Iyer, Leonardo Bobadilla, S. S. Iyengar

**Abstract**: Large Language Models (LLMs) such as Gemma-2B have shown strong performance in various natural language processing tasks. However, general-purpose models often lack the domain expertise required for cybersecurity applications. This work presents a methodology to fine-tune the Gemma-2B model into a domain-specific cybersecurity LLM. We detail the processes of dataset preparation, fine-tuning, and synthetic data generation, along with implications for real-world applications in threat detection, forensic investigation, and attack analysis.   Experiments highlight challenges in prompt length distribution during domain-specific fine-tuning. Uneven prompt lengths limit the model's effective use of the context window, constraining local inference to 200-400 tokens despite hardware support for longer sequences. Chain-of-thought styled prompts, paired with quantized weights, yielded the best performance under these constraints. To address context limitations, we employed a hybrid strategy using cloud LLMs for synthetic data generation and local fine-tuning for deployment efficiency.   To extend the evaluation, we introduce a Retrieval-Augmented Generation (RAG) pipeline and graph-based reasoning framework. This approach enables structured alignment with MITRE ATT&CK techniques through STIX-based threat intelligence, enhancing recall in multi-hop and long-context scenarios. Graph modules encode entity-neighborhood context and tactic chains, helping mitigate the constraints of short prompt windows. Results demonstrate improved model alignment with tactic, technique, and procedure (TTP) coverage, validating the utility of graph-augmented LLMs in cybersecurity threat intelligence applications.



## **34. Memory Poisoning Attack and Defense on Memory Based LLM-Agents**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.05504v2) [paper-pdf](https://arxiv.org/pdf/2601.05504v2)

**Authors**: Balachandra Devarangadi Sunil, Isheeta Sinha, Piyush Maheshwari, Shantanu Todmal, Shreyan Mallik, Shuchi Mishra

**Abstract**: Large language model agents equipped with persistent memory are vulnerable to memory poisoning attacks, where adversaries inject malicious instructions through query only interactions that corrupt the agents long term memory and influence future responses. Recent work demonstrated that the MINJA (Memory Injection Attack) achieves over 95 % injection success rate and 70 % attack success rate under idealized conditions. However, the robustness of these attacks in realistic deployments and effective defensive mechanisms remain understudied. This work addresses these gaps through systematic empirical evaluation of memory poisoning attacks and defenses in Electronic Health Record (EHR) agents. We investigate attack robustness by varying three critical dimensions: initial memory state, number of indication prompts, and retrieval parameters. Our experiments on GPT-4o-mini, Gemini-2.0-Flash and Llama-3.1-8B-Instruct models using MIMIC-III clinical data reveal that realistic conditions with pre-existing legitimate memories dramatically reduce attack effectiveness. We then propose and evaluate two novel defense mechanisms: (1) Input/Output Moderation using composite trust scoring across multiple orthogonal signals, and (2) Memory Sanitization with trust-aware retrieval employing temporal decay and pattern-based filtering. Our defense evaluation reveals that effective memory sanitization requires careful trust threshold calibration to prevent both overly conservative rejection (blocking all entries) and insufficient filtering (missing subtle attacks), establishing important baselines for future adaptive defense mechanisms. These findings provide crucial insights for securing memory-augmented LLM agents in production environments.



## **35. STELP: Secure Transpilation and Execution of LLM-Generated Programs**

cs.SE

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.05467v3) [paper-pdf](https://arxiv.org/pdf/2601.05467v3)

**Authors**: Swapnil Shinde, Sahil Wadhwa, Andy Luo, Akshay Gupta, Mohammad Shahed Sorower

**Abstract**: Rapid evolution of Large Language Models (LLMs) has achieved major advances in reasoning, planning, and function-calling capabilities. Multi-agentic collaborative frameworks using such LLMs place them at the center of solving software development-related tasks such as code generation. However, direct use of LLM generated code in production software development systems is problematic. The code could be unstable or erroneous and contain vulnerabilities such as data poisoning, malicious attacks, and hallucinations that could lead to widespread system malfunctions. This prohibits the adoption of LLM generated code in production AI systems where human code reviews and traditional secure testing tools are impractical or untrustworthy. In this paper, we discuss safety and reliability problems with the execution of LLM generated code and propose a Secure Transpiler and Executor of LLM-Generated Program (STELP), capable of executing LLM-generated code in a controlled and safe manner. STELP secures autonomous production AI systems involving code generation, filling the critical void left by the impracticality or limitations of traditional secure testing methodologies and human oversight. This includes applications such as headless code generation-execution and LLMs that produce executable code snippets as an action plan to be executed in real time. We contribute a human-validated dataset of insecure code snippets and benchmark our approach on publicly available datasets for correctness, safety, and latency. Our results demonstrate that our approach outperforms an existing method by a significant margin, particularly in its ability to safely execute risky code snippets. Warning: This paper contains malicious code snippets that should be run with caution.



## **36. BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents**

cs.AI

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.04566v2) [paper-pdf](https://arxiv.org/pdf/2601.04566v2)

**Authors**: Yunhao Feng, Yige Li, Yutao Wu, Yingshui Tan, Yanming Guo, Yifan Ding, Kun Zhai, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large language model (LLM) agents execute tasks through multi-step workflows that combine planning, memory, and tool use. While this design enables autonomy, it also expands the attack surface for backdoor threats. Backdoor triggers injected into specific stages of an agent workflow can persist through multiple intermediate states and adversely influence downstream outputs. However, existing studies remain fragmented and typically analyze individual attack vectors in isolation, leaving the cross-stage interaction and propagation of backdoor triggers poorly understood from an agent-centric perspective. To fill this gap, we propose \textbf{BackdoorAgent}, a modular and stage-aware framework that provides a unified, agent-centric view of backdoor threats in LLM agents. BackdoorAgent structures the attack surface into three functional stages of agentic workflows, including \textbf{planning attacks}, \textbf{memory attacks}, and \textbf{tool-use attacks}, and instruments agent execution to enable systematic analysis of trigger activation and propagation across different stages. Building on this framework, we construct a standardized benchmark spanning four representative agent applications: \textbf{Agent QA}, \textbf{Agent Code}, \textbf{Agent Web}, and \textbf{Agent Drive}, covering both language-only and multimodal settings. Our empirical analysis shows that \textit{triggers implanted at a single stage can persist across multiple steps and propagate through intermediate states.} For instance, when using a GPT-based backbone, we observe trigger persistence in 43.58\% of planning attacks, 77.97\% of memory attacks, and 60.28\% of tool-stage attacks, highlighting the vulnerabilities of the agentic workflow itself to backdoor threats. To facilitate reproducibility and future research, our code and benchmark are publicly available at GitHub.



## **37. TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards**

cs.AI

21 pages, 15 figures

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2512.07761v2) [paper-pdf](https://arxiv.org/pdf/2512.07761v2)

**Authors**: Xiqiao Xiong, Ouxiang Li, Zhuo Liu, Moxin Li, Wentao Shi, Fengbin Zhu, Qifan Wang, Fuli Feng

**Abstract**: Large language models have seen widespread adoption, yet they remain vulnerable to multi-turn jailbreak attacks, threatening their safe deployment. This has led to the task of training automated multi-turn attackers to probe model safety vulnerabilities. However, existing approaches typically rely on turn-level optimization, which is insufficient for learning long-term attack strategies. To bridge this gap, we formulate this task as a multi-turn reinforcement learning problem, directly optimizing the harmfulness of the final-turn response as the outcome reward. To address the sparse supervision of the outcome reward, we introduce TROJail, which employs two process rewards to evaluate the utility of intermediate prompts and integrate them into advantage estimation. These rewards (1) penalize overly harmful prompts that trigger the model's refusal mechanism, and (2) encourage steering the semantic relevance of responses toward the targeted harmful content. Experimental results show improved attack success rates across multiple models and benchmarks, highlighting the effectiveness of our approach. The code is available at https://github.com/xxiqiao/TROJail. Warning: This paper contains examples of harmful content.



## **38. From static to adaptive: immune memory-based jailbreak detection for large language models**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2512.03356v2) [paper-pdf](https://arxiv.org/pdf/2512.03356v2)

**Authors**: Jun Leng, Yu Liu, Litian Zhang, Ruihan Hu, Zhuting Fang, Xi Zhang

**Abstract**: Large Language Models (LLMs) serve as the backbone of modern AI systems, yet they remain susceptible to adversarial jailbreak attacks. Consequently, robust detection of such malicious inputs is paramount for ensuring model safety. Traditional detection methods typically rely on external models trained on fixed, large-scale datasets, which often incur significant computational overhead. While recent methods shift toward leveraging internal safety signals of models to enable more lightweight and efficient detection. However, these methods remain inherently static and struggle to adapt to the evolving nature of jailbreak attacks. Drawing inspiration from the biological immune mechanism, we introduce the Immune Memory Adaptive Guard (IMAG) framework. By distilling and encoding safety patterns into a persistent, evolvable memory bank, IMAG enables adaptive generalization to emerging threats. Specifically, the framework orchestrates three synergistic components: Immune Detection, which employs retrieval for high-efficiency interception of known jailbreak attacks; Active Immunity, which performs proactive behavioral simulation to resolve ambiguous unknown queries; Memory Updating, which integrates validated attack patterns back into the memory bank. This closed-loop architecture transitions LLM defense from rigid filtering to autonomous adaptive mitigation. Extensive evaluations across five representative open-source LLMs demonstrate that our method surpasses state-of-the-art (SOTA) baselines, achieving a superior average detection accuracy of 94\% across diverse and complex attack types.



## **39. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

cs.CL

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2511.15304v3) [paper-pdf](https://arxiv.org/pdf/2511.15304v3)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale Syrnikov, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.



## **40. NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge**

cs.AI

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2510.21144v2) [paper-pdf](https://arxiv.org/pdf/2510.21144v2)

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.



## **41. Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers**

cs.LG

Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026)

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2510.14381v2) [paper-pdf](https://arxiv.org/pdf/2510.14381v2)

**Authors**: Andrew Zhao, Reshmi Ghosh, Vitor Carvalho, Emily Lawton, Keegan Hines, Gao Huang, Jack W. Stokes

**Abstract**: Large language model (LLM) systems increasingly power everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on manually well-crafted prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to query poisoning alone: feedback-based attacks raise attack success rate (ASR) by up to ΔASR = 0.48. We introduce a simple fake reward attack that requires no access to the reward model and significantly increases vulnerability. We also propose a lightweight highlighting defense that reduces the fake reward ΔASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks.



## **42. RIPRAG: Hack a Black-box Retrieval-Augmented Generation Question-Answering System with Reinforcement Learning**

cs.AI

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2510.10008v2) [paper-pdf](https://arxiv.org/pdf/2510.10008v2)

**Authors**: Meng Xi, Sihan Lv, Yechen Jin, Guanjie Cheng, Naibo Wang, Ying Li, Jianwei Yin

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become a core technology for tasks such as question-answering (QA) and content generation. RAG poisoning is an attack method to induce LLMs to generate the attacker's expected text by injecting poisoned documents into the database of RAG systems. Existing research can be broadly divided into two classes: white-box methods and black-box methods. White-box methods utilize gradient information to optimize poisoned documents, and black-box methods use a pre-trained LLM to generate them. However, existing white-box methods require knowledge of the RAG system's internal composition and implementation details, whereas black-box methods are unable to utilize interactive information. In this work, we propose the RIPRAG attack framework, an end-to-end attack pipeline that treats the target RAG system as a black box and leverages our proposed Reinforcement Learning from Black-box Feedback (RLBF) method to optimize the generation model for poisoned documents. We designed two kinds of rewards: similarity reward and attack reward. Experimental results demonstrate that this method can effectively execute poisoning attacks against most complex RAG systems, achieving an attack success rate (ASR) improvement of up to 0.72 compared to baseline methods. This highlights prevalent deficiencies in current defensive methods and provides critical insights for LLM security research.



## **43. Membership Inference Attacks on Tokenizers of Large Language Models**

cs.CR

To appear at USENIX Security Symposium 2026

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2510.05699v2) [paper-pdf](https://arxiv.org/pdf/2510.05699v2)

**Authors**: Meng Tong, Yuntao Du, Kejiang Chen, Weiming Zhang

**Abstract**: Membership inference attacks (MIAs) are widely used to assess the privacy risks associated with machine learning models. However, when these attacks are applied to pre-trained large language models (LLMs), they encounter significant challenges, including mislabeled samples, distribution shifts, and discrepancies in model size between experimental and real-world settings. To address these limitations, we introduce tokenizers as a new attack vector for membership inference. Specifically, a tokenizer converts raw text into tokens for LLMs. Unlike full models, tokenizers can be efficiently trained from scratch, thereby avoiding the aforementioned challenges. In addition, the tokenizer's training data is typically representative of the data used to pre-train LLMs. Despite these advantages, the potential of tokenizers as an attack vector remains unexplored. To this end, we present the first study on membership leakage through tokenizers and explore five attack methods to infer dataset membership. Extensive experiments on millions of Internet samples reveal the vulnerabilities in the tokenizers of state-of-the-art LLMs. To mitigate this emerging risk, we further propose an adaptive defense. Our findings highlight tokenizers as an overlooked yet critical privacy threat, underscoring the urgent need for privacy-preserving mechanisms specifically designed for them.



## **44. Secure and Efficient Access Control for Computer-Use Agents via Context Space**

cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2509.22256v4) [paper-pdf](https://arxiv.org/pdf/2509.22256v4)

**Authors**: Haochen Gong, Chenxiao Li, Rui Chang, Wenbo Shen

**Abstract**: Large language model (LLM)-based computer-use agents represent a convergence of AI and OS capabilities, enabling natural language to control system- and application-level functions. However, due to LLMs' inherent uncertainty issues, granting agents control over computers poses significant security risks. When agent actions deviate from user intentions, they can cause irreversible consequences. Existing mitigation approaches, such as user confirmation and LLM-based dynamic action validation, still suffer from limitations in usability, security, and performance. To address these challenges, we propose CSAgent, a system-level, static policy-based access control framework for computer-use agents. To bridge the gap between static policy and dynamic context and user intent, CSAgent introduces intent- and context-aware policies, and provides an automated toolchain to assist developers in constructing and refining them. CSAgent enforces these policies through an optimized OS service, ensuring that agent actions can only be executed under specific user intents and contexts. CSAgent supports protecting agents that control computers through diverse interfaces, including API, CLI, and GUI. We implement and evaluate CSAgent, which successfully defends against all attacks in the benchmarks while introducing only 1.99% performance overhead and 5.42% utility decrease.



## **45. NATLM: Detecting Defects in NFT Smart Contracts Leveraging LLM**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2508.01351v2) [paper-pdf](https://arxiv.org/pdf/2508.01351v2)

**Authors**: Yuanzheng Niu, Xiaoqi Li, Wenkai Li

**Abstract**: Security issues are becoming increasingly significant with the rapid evolution of Non-fungible Tokens (NFTs). As NFTs are traded as digital assets, they have emerged as prime targets for cyber attackers. In the development of NFT smart contracts, there may exist undiscovered defects that could lead to substantial financial losses if exploited. To tackle this issue, this paper presents a framework called NATLM(NFT Assistant LLM), designed to detect potential defects in NFT smart contracts. The framework effectively identifies four common types of vulnerabilities in NFT smart contracts: ERC-721 Reentrancy, Public Burn, Risky Mutable Proxy, and Unlimited Minting. Relying exclusively on large language models (LLMs) for defect detection can lead to a high false-positive rate. To enhance detection performance, NATLM integrates static analysis with LLMs, specifically Gemini Pro 1.5. Initially, NATLM employs static analysis to extract structural, syntactic, and execution flow information from the code, represented through Abstract Syntax Trees (AST) and Control Flow Graphs (CFG). These extracted features are then combined with vectors of known defect examples to create a matrix for input into the knowledge base. Subsequently, the feature vectors and code vectors of the analyzed contract are compared with the contents of the knowledge base. Finally, the LLM performs deep semantic analysis to enhance detection capabilities, providing a more comprehensive and accurate identification of potential security issues. Experimental results indicate that NATLM analyzed 8,672 collected NFT smart contracts, achieving an overall precision of 87.72%, a recall of 89.58%, and an F1 score of 88.94%. The results outperform other baseline experiments, successfully identifying four common types of defects.



## **46. AI Agent Smart Contract Exploit Generation**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2507.05558v4) [paper-pdf](https://arxiv.org/pdf/2507.05558v4)

**Authors**: Arthur Gervais, Liyi Zhou

**Abstract**: Smart contract vulnerabilities have led to billions in losses, yet finding actionable exploits remains challenging. Traditional fuzzers rely on rigid heuristics and struggle with complex attacks, while human auditors are thorough but slow and don't scale. Large Language Models offer a promising middle ground, combining human-like reasoning with machine speed.   Early studies show that simply prompting LLMs generates unverified vulnerability speculations with high false positive rates. To address this, we present A1, an agentic system that transforms any LLM into an end-to-end exploit generator. A1 provides agents with six domain-specific tools for autonomous vulnerability discovery, from understanding contract behavior to testing strategies on real blockchain states. All outputs are concretely validated through execution, ensuring only profitable proof-of-concept exploits are reported. We evaluate A1 across 36 real-world vulnerable contracts on Ethereum and Binance Smart Chain. A1 achieves a 63% success rate on the VERITE benchmark. Across all successful cases, A1 extracts up to \$8.59 million per exploit and \$9.33 million total.   Using Monte Carlo analysis of historical attacks, we demonstrate that immediate vulnerability detection yields 86-89% success probability, dropping to 6-21% with week-long delays. Our economic analysis reveals a troubling asymmetry: attackers achieve profitability at \$6,000 exploit values while defenders require \$60,000 -- raising fundamental questions about whether AI agents inevitably favor exploitation over defense.



## **47. Exploring the Secondary Risks of Large Language Models**

cs.LG

18 pages, 5 figures

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2506.12382v4) [paper-pdf](https://arxiv.org/pdf/2506.12382v4)

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments.



## **48. Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation**

cs.CL

Accepted by NeruIPS 2025

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2501.18100v2) [paper-pdf](https://arxiv.org/pdf/2501.18100v2)

**Authors**: Yibo Wang, Tiansheng Huang, Li Shen, Huanjin Yao, Haotian Luo, Rui Liu, Naiqiang Tan, Jiaxing Huang, Dacheng Tao

**Abstract**: Harmful fine-tuning attack introduces significant security risks to the fine-tuning services. Main-stream defenses aim to vaccinate the model such that the later harmful fine-tuning attack is less effective. However, our evaluation results show that such defenses are fragile--with a few fine-tuning steps, the model still can learn the harmful knowledge. To this end, we do further experiment and find that an embarrassingly simple solution--adding purely random perturbations to the fine-tuned model, can recover the model from harmful behaviors, though it leads to a degradation in the model's fine-tuning performance. To address the degradation of fine-tuning performance, we further propose Panacea, which optimizes an adaptive perturbation that will be applied to the model after fine-tuning. Panacea maintains model's safety alignment performance without compromising downstream fine-tuning performance. Comprehensive experiments are conducted on different harmful ratios, fine-tuning tasks and mainstream LLMs, where the average harmful scores are reduced by up-to 21.2%, while maintaining fine-tuning performance. As a by-product, we analyze the adaptive perturbation and show that different layers in various LLMs have distinct safety affinity, which coincide with finding from several previous study. Source code available at https://github.com/w-yibo/Panacea.



## **49. Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models**

cs.SD

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2501.13772v4) [paper-pdf](https://arxiv.org/pdf/2501.13772v4)

**Authors**: Hao Cheng, Erjia Xiao, Jing Shao, Yichi Wang, Le Yang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Large Language Models (LLMs) demonstrate impressive zero-shot performance across a wide range of natural language processing tasks. Integrating various modality encoders further expands their capabilities, giving rise to Multimodal Large Language Models (MLLMs) that process not only text but also visual and auditory modality inputs. However, these advanced capabilities may also pose significant safety problems, as models can be exploited to generate harmful or inappropriate content through jailbreak attacks. While prior work has extensively explored how manipulating textual or visual modality inputs can circumvent safeguards in LLMs and MLLMs, the vulnerability of audio-specific jailbreak on Large Audio-Language Models (LALMs) remains largely underexplored. To address this gap, we introduce Jailbreak-AudioBench, which consists of the Toolbox, curated Dataset, and comprehensive Benchmark. The Toolbox supports not only text-to-audio conversion but also various editing techniques for injecting audio hidden semantics. The curated Dataset provides diverse explicit and implicit jailbreak audio examples in both original and edited forms. Utilizing this dataset, we evaluate multiple state-of-the-art LALMs and establish the most comprehensive Jailbreak benchmark to date for audio modality. Finally, Jailbreak-AudioBench establishes a foundation for advancing future research on LALMs safety alignment by enabling the in-depth exposure of more powerful jailbreak threats, such as query-based audio editing, and by facilitating the development of effective defense mechanisms.



## **50. Can Editing LLMs Inject Harm?**

cs.CL

Accepted to Proceedings of AAAI 2026. The first two authors contributed equally. 7 pages for main paper, 31 pages including appendix. The code, results, dataset for this paper and more resources are on the project website: https://llm-editing.github.io

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2407.20224v4) [paper-pdf](https://arxiv.org/pdf/2407.20224v4)

**Authors**: Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, Kai Shu

**Abstract**: Large Language Models (LLMs) have emerged as a new information channel. Meanwhile, one critical but under-explored question is: Is it possible to bypass the safety alignment and inject harmful information into LLMs stealthily? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the first risk, we find that editing attacks can inject both commonsense and long-tail misinformation into LLMs, and the effectiveness for the former one is particularly high. For the second risk, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can degrade the overall fairness. Then, we further illustrate the high stealthiness of editing attacks. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.



