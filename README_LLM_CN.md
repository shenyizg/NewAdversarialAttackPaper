# Latest Large Language Model Attack Papers
**update at 2026-01-04 08:57:07**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. RAGPart & RAGMask: Retrieval-Stage Defenses Against Corpus Poisoning in Retrieval-Augmented Generation**

RAGPart & RAGMass：检索增强一代中的检索阶段防御体中毒 cs.IR

Published at AAAI 2026 Workshop on New Frontiers in Information Retrieval [Oral]

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24268v1) [paper-pdf](https://arxiv.org/pdf/2512.24268v1)

**Authors**: Pankayaraj Pathmanathan, Michael-Andrei Panaitescu-Liess, Cho-Yu Jason Chiang, Furong Huang

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to enhance large language models (LLMs) with external knowledge, reducing hallucinations and compensating for outdated information. However, recent studies have exposed a critical vulnerability in RAG pipelines corpus poisoning where adversaries inject malicious documents into the retrieval corpus to manipulate model outputs. In this work, we propose two complementary retrieval-stage defenses: RAGPart and RAGMask. Our defenses operate directly on the retriever, making them computationally lightweight and requiring no modification to the generation model. RAGPart leverages the inherent training dynamics of dense retrievers, exploiting document partitioning to mitigate the effect of poisoned points. In contrast, RAGMask identifies suspicious tokens based on significant similarity shifts under targeted token masking. Across two benchmarks, four poisoning strategies, and four state-of-the-art retrievers, our defenses consistently reduce attack success rates while preserving utility under benign conditions. We further introduce an interpretable attack to stress-test our defenses. Our findings highlight the potential and limitations of retrieval-stage defenses, providing practical insights for robust RAG deployments.

摘要: 检索增强生成（RAG）已成为一种有前途的范式，可以利用外部知识增强大型语言模型（LLM），减少幻觉并补偿过时信息。然而，最近的研究暴露了RAG管道库中毒中的一个关键漏洞，即对手将恶意文档注入检索库以操纵模型输出。在这项工作中，我们提出了两种补充的检索阶段防御：RAGPart和RAGMass。我们的防御系统直接在寻回犬上运行，使它们在计算上轻量级，并且不需要修改生成模型。RAGPart利用密集检索器的固有训练动态，利用文档分区来减轻中毒点的影响。相比之下，RAGMass根据目标令牌屏蔽下的显着相似性变化来识别可疑令牌。通过两种基准、四种中毒策略和四种最先进的寻回犬，我们的防御系统持续降低攻击成功率，同时在良性条件下保持实用性。我们进一步引入可解释的攻击来压力测试我们的防御。我们的研究结果强调了回收阶段防御的潜力和局限性，为稳健的RAG部署提供了实用的见解。



## **2. Jailbreaking Attacks vs. Content Safety Filters: How Far Are We in the LLM Safety Arms Race?**

越狱攻击与内容安全过滤器：我们在LLM安全军备竞赛中走了多远？ cs.CR

26 pages,11 tables, 7 figures

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24044v1) [paper-pdf](https://arxiv.org/pdf/2512.24044v1)

**Authors**: Yuan Xin, Dingfan Chen, Linyi Yang, Michael Backes, Xiao Zhang

**Abstract**: As large language models (LLMs) are increasingly deployed, ensuring their safe use is paramount. Jailbreaking, adversarial prompts that bypass model alignment to trigger harmful outputs, present significant risks, with existing studies reporting high success rates in evading common LLMs. However, previous evaluations have focused solely on the models, neglecting the full deployment pipeline, which typically incorporates additional safety mechanisms like content moderation filters. To address this gap, we present the first systematic evaluation of jailbreak attacks targeting LLM safety alignment, assessing their success across the full inference pipeline, including both input and output filtering stages. Our findings yield two key insights: first, nearly all evaluated jailbreak techniques can be detected by at least one safety filter, suggesting that prior assessments may have overestimated the practical success of these attacks; second, while safety filters are effective in detection, there remains room to better balance recall and precision to further optimize protection and user experience. We highlight critical gaps and call for further refinement of detection accuracy and usability in LLM safety systems.

摘要: 随着大型语言模型（LLM）的部署越来越多，确保它们的安全使用至关重要。越狱、对抗性促使绕过模型对齐引发有害输出，带来重大风险，现有研究报告称，规避常见LLM的成功率很高。然而，之前的评估仅关注模型，忽视了完整的部署管道，该管道通常包含内容审核过滤器等额外的安全机制。为了解决这一差距，我们首次对针对LLM安全调整的越狱攻击进行了系统评估，评估了它们在整个推理管道（包括输入和输出过滤阶段）中的成功。我们的研究结果得出了两个关键见解：首先，几乎所有评估的越狱技术都可以被至少一个安全过滤器检测到，这表明之前的评估可能高估了这些攻击的实际成功;其次，虽然安全过滤器在检测方面有效，但仍有空间更好地平衡召回和精确度，以进一步优化保护和用户体验。我们强调了关键差距，并呼吁进一步完善LLM安全系统的检测准确性和可用性。



## **3. RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress**

重复诅咒：测量和理解在OSS压力下混合专家LLM中的路由器不平衡 cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.23995v1) [paper-pdf](https://arxiv.org/pdf/2512.23995v1)

**Authors**: Ruixuan Huang, Qingyue Wang, Hantao Huang, Yudong Gao, Dong Chen, Shuai Wang, Wei Wang

**Abstract**: Mixture-of-Experts architectures have become the standard for scaling large language models due to their superior parameter efficiency. To accommodate the growing number of experts in practice, modern inference systems commonly adopt expert parallelism to distribute experts across devices. However, the absence of explicit load balancing constraints during inference allows adversarial inputs to trigger severe routing concentration. We demonstrate that out-of-distribution prompts can manipulate the routing strategy such that all tokens are consistently routed to the same set of top-$k$ experts, which creates computational bottlenecks on certain devices while forcing others to idle. This converts an efficiency mechanism into a denial-of-service attack vector, leading to violations of service-level agreements for time to first token. We propose RepetitionCurse, a low-cost black-box strategy to exploit this vulnerability. By identifying a universal flaw in MoE router behavior, RepetitionCurse constructs adversarial prompts using simple repetitive token patterns in a model-agnostic manner. On widely deployed MoE models like Mixtral-8x7B, our method increases end-to-end inference latency by 3.063x, degrading service availability significantly.

摘要: 专家混合架构因其卓越的参数效率而成为扩展大型语言模型的标准。为了适应实践中越来越多的专家，现代推理系统通常采用专家并行性来跨设备分布专家。然而，推理过程中缺乏显式的负载平衡约束，导致对抗性输入触发严重的路由集中。我们证明，分发外提示可以操纵路由策略，以便所有令牌一致地路由到同一组顶级k$专家，这在某些设备上造成了计算瓶颈，同时迫使其他设备闲置。这将效率机制转化为拒绝服务攻击载体，导致违反第一个令牌时间的服务级别协议。我们提出了RepetitionCurse，这是一种利用此漏洞的低成本黑匣子策略。通过识别MoE路由器行为中的普遍缺陷，RepetitionCurse以模型不可知的方式使用简单的重复令牌模式构建对抗性提示。在Mixtral-8x 7 B等广泛部署的MoE模型上，我们的方法将端到端推理延迟增加了3.063倍，从而显着降低了服务可用性。



## **4. T2VAttack: Adversarial Attack on Text-to-Video Diffusion Models**

T2 VAttack：对文本到视频扩散模型的对抗性攻击 cs.CV

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.23953v1) [paper-pdf](https://arxiv.org/pdf/2512.23953v1)

**Authors**: Changzhen Li, Yuecong Min, Jie Zhang, Zheng Yuan, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Text-to-Video (T2V) diffusion models has driven remarkable advancements in generating high-quality, temporally coherent videos from natural language descriptions. Despite these achievements, their vulnerability to adversarial attacks remains largely unexplored. In this paper, we introduce T2VAttack, a comprehensive study of adversarial attacks on T2V diffusion models from both semantic and temporal perspectives. Considering the inherently dynamic nature of video data, we propose two distinct attack objectives: a semantic objective to evaluate video-text alignment and a temporal objective to assess the temporal dynamics. To achieve an effective and efficient attack process, we propose two adversarial attack methods: (i) T2VAttack-S, which identifies semantically or temporally critical words in prompts and replaces them with synonyms via greedy search, and (ii) T2VAttack-I, which iteratively inserts optimized words with minimal perturbation to the prompt. By combining these objectives and strategies, we conduct a comprehensive evaluation on the adversarial robustness of several state-of-the-art T2V models, including ModelScope, CogVideoX, Open-Sora, and HunyuanVideo. Our experiments reveal that even minor prompt modifications, such as the substitution or insertion of a single word, can cause substantial degradation in semantic fidelity and temporal dynamics, highlighting critical vulnerabilities in current T2V diffusion models.

摘要: 文本到视频（T2 V）扩散模型的快速发展推动了从自然语言描述生成高质量、时间连贯的视频方面的显着进步。尽管取得了这些成就，但它们对对抗攻击的脆弱性在很大程度上仍然没有被探索。本文介绍了T2 VAttack，这是从语义和时间角度对T2 V扩散模型的对抗性攻击的全面研究。考虑到视频数据固有的动态性质，我们提出了两个不同的攻击目标：评估视频-文本对齐的语义目标和评估时间动态的时间目标。为了实现有效且高效的攻击过程，我们提出了两种对抗攻击方法：（i）T2 VAttack-S，它识别提示中的语义或时间关键词，并通过贪婪搜索用同义词替换它们，和（ii）T2 VAttack-I，它迭代地插入优化的词，对提示的干扰最小。通过结合这些目标和策略，我们对几种最先进的T2 V模型（包括Model Scope、CogVideoX、Open-Sora和HunyuanVideo）的对抗稳健性进行了全面评估。我们的实验表明，即使是微小的即时修改，例如替换或插入单个单词，也可能导致语义保真度和时间动态性的大幅下降，凸显了当前T2 V扩散模型中的关键漏洞。



## **5. Breaking Audio Large Language Models by Attacking Only the Encoder: A Universal Targeted Latent-Space Audio Attack**

仅通过攻击编码器来破解音频大型语言模型：一种通用目标潜在空间音频攻击 cs.SD

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23881v1) [paper-pdf](https://arxiv.org/pdf/2512.23881v1)

**Authors**: Roee Ziv, Raz Lapid, Moshe Sipper

**Abstract**: Audio-language models combine audio encoders with large language models to enable multimodal reasoning, but they also introduce new security vulnerabilities. We propose a universal targeted latent space attack, an encoder-level adversarial attack that manipulates audio latent representations to induce attacker-specified outputs in downstream language generation. Unlike prior waveform-level or input-specific attacks, our approach learns a universal perturbation that generalizes across inputs and speakers and does not require access to the language model. Experiments on Qwen2-Audio-7B-Instruct demonstrate consistently high attack success rates with minimal perceptual distortion, revealing a critical and previously underexplored attack surface at the encoder level of multimodal systems.

摘要: 音频语言模型将音频编码器与大型语言模型相结合，以实现多模式推理，但它们也引入了新的安全漏洞。我们提出了一种通用的有针对性的潜在空间攻击，这是一种编码器级对抗攻击，它操纵音频潜在表示以在下游语言生成中引发攻击者指定的输出。与之前的波级或特定于输入的攻击不同，我们的方法学习了一种通用扰动，该扰动在输入和说话者之间进行概括，并且不需要访问语言模型。Qwen 2-Audio-7 B-Direct上的实验证明，攻击成功率始终很高，感知失真最小，揭示了多模式系统编码器级的关键且之前未充分探索的攻击表面。



## **6. Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing**

对基于LLM的学术评论的多语言隐藏提示注入攻击 cs.CL

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23684v1) [paper-pdf](https://arxiv.org/pdf/2512.23684v1)

**Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai. -Doss

**Abstract**: Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.

摘要: 大型语言模型（LLM）越来越多地被考虑用于高影响力的工作流程，包括学术同行评审。然而，LLM很容易受到文档级隐藏提示注入攻击。在这项工作中，我们构建了一个由ICML接受的大约500篇真实学术论文组成的数据集，并评估在这些文档中嵌入隐藏的对抗提示的效果。每份论文都注入了四种不同语言的语义等效指令，并使用LLM进行审查。我们发现，及时注射会导致英语、日语和中文注射的审查分数和接受/拒绝决定发生重大变化，而阿拉伯语注射几乎没有影响。这些结果凸显了基于LLM的审查系统对文档级提示注入的敏感性，并揭示了不同语言之间脆弱性的显着差异。



## **7. Toward Trustworthy Agentic AI: A Multimodal Framework for Preventing Prompt Injection Attacks**

迈向值得信赖的抽象人工智能：防止即时注入攻击的多模式框架 cs.CR

It is accepted in a conference paper, ICCA 2025 in Bahrain on 21 to 23 December

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23557v1) [paper-pdf](https://arxiv.org/pdf/2512.23557v1)

**Authors**: Toqeer Ali Syed, Mishal Ateeq Almutairi, Mahmoud Abdel Moaty

**Abstract**: Powerful autonomous systems, which reason, plan, and converse using and between numerous tools and agents, are made possible by Large Language Models (LLMs), Vision-Language Models (VLMs), and new agentic AI systems, like LangChain and GraphChain. Nevertheless, this agentic environment increases the probability of the occurrence of multimodal prompt injection (PI) attacks, in which concealed or malicious instructions carried in text, pictures, metadata, or agent-to-agent messages may spread throughout the graph and lead to unintended behavior, a breach of policy, or corruption of state. In order to mitigate these risks, this paper suggests a Cross-Agent Multimodal Provenanc- Aware Defense Framework whereby all the prompts, either user-generated or produced by upstream agents, are sanitized and all the outputs generated by an LLM are verified independently before being sent to downstream nodes. This framework contains a Text sanitizer agent, visual sanitizer agent, and output validator agent all coordinated by a provenance ledger, which keeps metadata of modality, source, and trust level throughout the entire agent network. This architecture makes sure that agent-to-agent communication abides by clear trust frames such such that injected instructions are not propagated down LangChain or GraphChain-style-workflows. The experimental assessments show that multimodal injection detection accuracy is significantly enhanced, and the cross-agent trust leakage is minimized, as well as, agentic execution pathways become stable. The framework, which expands the concept of provenance tracking and validation to the multi-agent orchestration, enhances the establishment of secure, understandable and reliable agentic AI systems.

摘要: 大型语言模型（LLM）、视觉语言模型（VLMS）和LangChain和GraphChain等新的代理人工智能系统使强大的自治系统成为可能，可以使用多种工具和代理以及在多种工具和代理之间进行推理、规划和对话。然而，这种代理环境增加了多模式提示注入（PI）攻击发生的可能性，其中文本、图片、元数据或代理到代理消息中携带的隐藏或恶意指令可能会在整个图中传播并导致意外行为、违反政策或国家腐败。为了减轻这些风险，本文提出了一种跨代理多模式Provenanc- Aware Defense框架，其中所有提示（无论是用户生成的还是由上游代理产生的）都经过清理，并且LLM生成的所有输出在发送到下游节点之前都经过独立验证。该框架包含文本消毒器代理、视觉消毒器代理和输出验证器代理，所有这些都由来源分类帐协调，该分类帐保留整个代理网络中的模式、源和信任级别的元数据。该架构确保代理到代理的通信遵守明确的信任框架，这样注入的指令不会沿着LangChain或GraphChain风格的工作流程传播。实验评估表明，多模式注入检测准确性显着提高，跨代理信任泄漏最小化，代理执行路径变得稳定。该框架将出处跟踪和验证的概念扩展到多代理编排，增强了安全、可理解和可靠的代理人工智能系统的建立。



## **8. Agentic AI for Autonomous Defense in Software Supply Chain Security: Beyond Provenance to Vulnerability Mitigation**

用于软件供应链安全自主防御的大型人工智能：超越漏洞缓解的根源 cs.CR

Conference paper, accept in ACCA IEEE Bahrain

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23480v1) [paper-pdf](https://arxiv.org/pdf/2512.23480v1)

**Authors**: Toqeer Ali Syed, Mohammad Riyaz Belgaum, Salman Jan, Asadullah Abdullah Khan, Saad Said Alqahtani

**Abstract**: The software supply chain attacks are becoming more and more focused on trusted development and delivery procedures, so the conventional post-build integrity mechanisms cannot be used anymore. The available frameworks like SLSA, SBOM and in toto are majorly used to offer provenance and traceability but do not have the capabilities of actively identifying and removing vulnerabilities in software production. The current paper includes an example of agentic artificial intelligence (AI) based on autonomous software supply chain security that combines large language model (LLM)-based reasoning, reinforcement learning (RL), and multi-agent coordination. The suggested system utilizes specialized security agents coordinated with the help of LangChain and LangGraph, communicates with actual CI/CD environments with the Model Context Protocol (MCP), and documents all the observations and actions in a blockchain security ledger to ensure integrity and auditing. Reinforcement learning can be used to achieve adaptive mitigation strategies that consider the balance between security effectiveness and the operational overhead, and LLMs can be used to achieve semantic vulnerability analysis, as well as explainable decisions. This framework is tested based on simulated pipelines, as well as, actual world CI/CD integrations on GitHub Actions and Jenkins, including injection attacks, insecure deserialization, access control violations, and configuration errors. Experimental outcomes indicate better detection accuracy, shorter mitigation latency and reasonable build-time overhead than rule-based, provenance only and RL only baselines. These results show that agentic AI can facilitate the transition to self defending, proactive software supply chains rather than reactive verification ones.

摘要: 软件供应链攻击越来越关注可信的开发和交付过程，因此传统的构建后完整性机制无法再使用。可用的框架（如SL SA、SBOM和toto）主要用于提供出处和可追溯性，但不具备主动识别和删除软件生产中漏洞的能力。当前的论文包括一个基于自主软件供应链安全的代理人工智能（AI）示例，该示例结合了基于大型语言模型（LLM）的推理、强化学习（RL）和多代理协调。建议的系统利用在LangChain和LangShape的帮助下协调的专业安全代理，通过模型上下文协议（HCP）与实际CI/CD环境通信，并在区块链安全分类帐中记录所有观察和操作，以确保完整性和审计。强化学习可用于实现考虑安全有效性和操作开销之间平衡的自适应缓解策略，LLM可用于实现语义漏洞分析以及可解释的决策。该框架基于模拟管道以及GitHub Actions和Jenkins上的实际CI/CD集成进行了测试，包括注入攻击，不安全的验证，访问控制违规和配置错误。实验结果表明，更好的检测准确性，更短的缓解延迟和合理的构建时间开销比基于规则的，只有出处和RL只有基线。这些结果表明，代理人工智能可以促进向自我防御、主动软件供应链而不是被动验证供应链的过渡。



## **9. Prompt-Induced Over-Generation as Denial-of-Service: A Black-Box Attack-Side Benchmark**

预算导致的过度发电作为拒绝服务：黑匣子攻击端基准 cs.CR

12 pages, 2 figures

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23779v1) [paper-pdf](https://arxiv.org/pdf/2512.23779v1)

**Authors**: Manu, Yi Guo, Jo Plested, Tim Lynar, Kanchana Thilakarathna, Nirhoshan Sivaroopan, Jack Yang, Wangli Yang

**Abstract**: Large language models (LLMs) can be driven into over-generation, emitting thousands of tokens before producing an end-of-sequence (EOS) token. This degrades answer quality, inflates latency and cost, and can be weaponized as a denial-of-service (DoS) attack. Recent work has begun to study DoS-style prompt attacks, but typically focuses on a single attack algorithm or assumes white-box access, without an attack-side benchmark that compares prompt-based attackers in a black-box, query-only regime with a known tokenizer. We introduce such a benchmark and study two prompt-only attackers. The first is Evolutionary Over-Generation Prompt Search (EOGen), which searches the token space for prefixes that suppress EOS and induce long continuations. The second is a goal-conditioned reinforcement learning attacker (RL-GOAL) that trains a network to generate prefixes conditioned on a target length. To characterize behavior, we introduce Over-Generation Factor (OGF), the ratio of produced tokens to a model's context window, along with stall and latency summaries. Our evolutionary attacker achieves mean OGF = 1.38 +/- 1.15 and Success@OGF >= 2 of 24.5 percent on Phi-3. RL-GOAL is stronger: across victims it achieves higher mean OGF (up to 2.81 +/- 1.38).

摘要: 大型语言模型（LLM）可以被驱动到过度生成，在生成序列结束（EOS）令牌之前发出数千个令牌。这会降低回答质量，增加延迟和成本，并且可以被武器化为拒绝服务（DPS）攻击。最近的工作已经开始研究Dos风格的提示攻击，但通常专注于单个攻击算法或假设白盒访问，而没有攻击端基准来比较黑匣子、仅查询机制中的基于预算的攻击者与已知的标记化器。我们引入这样的基准并研究两种仅限预算的攻击者。第一个是进化过代提示搜索（EOGen），它在令牌空间中搜索抑制EOS并导致长延续的前置。第二个是目标条件强化学习攻击者（RL-GOAL），它训练网络以生成以目标长度为条件的前置码。为了描述行为，我们引入了过代因子（OGF），即产生的令牌与模型上下文窗口的比率，以及停滞和延迟摘要。我们的进化攻击者在Phi-3上实现了平均OGF = 1.38 +/- 1.15和Success@OGF >= 2（24.5%）。WL-GOAL更强：在受害者中，它实现了更高的平均OGF（高达2.81 +/- 1.38）。



## **10. EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion**

EquaCode：通过方程求解和代码完成针对大型语言模型的多策略越狱方法 cs.CR

This is a preprint. A revised version will appear in the Proceedings of AAAI 2026

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23173v1) [paper-pdf](https://arxiv.org/pdf/2512.23173v1)

**Authors**: Zhen Liang, Hai Huang, Zhengkui Chen

**Abstract**: Large language models (LLMs), such as ChatGPT, have achieved remarkable success across a wide range of fields. However, their trustworthiness remains a significant concern, as they are still susceptible to jailbreak attacks aimed at eliciting inappropriate or harmful responses. However, existing jailbreak attacks mainly operate at the natural language level and rely on a single attack strategy, limiting their effectiveness in comprehensively assessing LLM robustness. In this paper, we propose Equacode, a novel multi-strategy jailbreak approach for large language models via equation-solving and code completion. This approach transforms malicious intent into a mathematical problem and then requires the LLM to solve it using code, leveraging the complexity of cross-domain tasks to divert the model's focus toward task completion rather than safety constraints. Experimental results show that Equacode achieves an average success rate of 91.19% on the GPT series and 98.65% across 3 state-of-the-art LLMs, all with only a single query. Further, ablation experiments demonstrate that EquaCode outperforms either the mathematical equation module or the code module alone. This suggests a strong synergistic effect, thereby demonstrating that multi-strategy approach yields results greater than the sum of its parts.

摘要: ChatGPT等大型语言模型（LLM）在广泛的领域取得了显着的成功。然而，它们的可信度仍然是一个重大问题，因为它们仍然容易受到旨在引发不当或有害反应的越狱攻击。然而，现有的越狱攻击主要在自然语言层面运行，并且依赖于单一的攻击策略，限制了其全面评估LLM稳健性的有效性。在本文中，我们提出了Equacode，这是一种通过方程求解和代码完成针对大型语言模型的新型多策略越狱方法。这种方法将恶意意图转化为数学问题，然后要求LLM使用代码来解决它，利用跨域任务的复杂性将模型的重点转移到任务完成而不是安全约束上。实验结果表明，Equacode在GPT系列上的平均成功率为91.19%，在3个最先进的LLM上的平均成功率为98.65%，所有这些都只需一个查询。此外，消融实验表明，EquaCode的性能优于数学方程模块或单独的代码模块。这表明具有强大的协同效应，从而证明多策略方法产生的结果大于其各部分之和。



## **11. It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents**

这是一个陷阱！任务重定向代理Web代理说服基准 cs.HC

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23128v1) [paper-pdf](https://arxiv.org/pdf/2512.23128v1)

**Authors**: Karolina Korgul, Yushi Yang, Arkadiusz Drohomirecki, Piotr Błaszczyk, Will Howard, Lukas Aichberger, Chris Russell, Philip H. S. Torr, Adam Mahdi, Adel Bibi

**Abstract**: Web-based agents powered by large language models are increasingly used for tasks such as email management or professional networking. Their reliance on dynamic web content, however, makes them vulnerable to prompt injection attacks: adversarial instructions hidden in interface elements that persuade the agent to divert from its original task. We introduce the Task-Redirecting Agent Persuasion Benchmark (TRAP), an evaluation for studying how persuasion techniques misguide autonomous web agents on realistic tasks. Across six frontier models, agents are susceptible to prompt injection in 25\% of tasks on average (13\% for GPT-5 to 43\% for DeepSeek-R1), with small interface or contextual changes often doubling success rates and revealing systemic, psychologically driven vulnerabilities in web-based agents. We also provide a modular social-engineering injection framework with controlled experiments on high-fidelity website clones, allowing for further benchmark expansion.

摘要: 由大型语言模型支持的基于Web的代理越来越多地用于电子邮件管理或专业网络等任务。然而，它们对动态网络内容的依赖使它们容易受到提示注入攻击：隐藏在界面元素中的对抗指令，说服代理从其原始任务转移。我们介绍了任务重定向代理说服基准（TRAP），这是一项评估，旨在研究说服技术如何在现实任务中误导自主网络代理。在六个前沿模型中，代理人平均容易在25%的任务中立即注入（GPT-5为13%，DeepSeek-R1为43%），微小的界面或上下文变化通常会使成功率翻倍，并揭示了基于网络的代理中系统性、心理驱动的漏洞。我们还提供了一个模块化的社会工程注入框架，在高保真网站克隆上进行受控实验，允许进一步的基准扩展。



## **12. Agentic AI for Cyber Resilience: A New Security Paradigm and Its System-Theoretic Foundations**

增强网络韧性的关键人工智能：一种新的安全范式及其系统理论基础 cs.CR

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22883v1) [paper-pdf](https://arxiv.org/pdf/2512.22883v1)

**Authors**: Tao Li, Quanyan Zhu

**Abstract**: Cybersecurity is being fundamentally reshaped by foundation-model-based artificial intelligence. Large language models now enable autonomous planning, tool orchestration, and strategic adaptation at scale, challenging security architectures built on static rules, perimeter defenses, and human-centered workflows. This chapter argues for a shift from prevention-centric security toward agentic cyber resilience. Rather than seeking perfect protection, resilient systems must anticipate disruption, maintain critical functions under attack, recover efficiently, and learn continuously. We situate this shift within the historical evolution of cybersecurity paradigms, culminating in an AI-augmented paradigm where autonomous agents participate directly in sensing, reasoning, action, and adaptation across cyber and cyber-physical systems. We then develop a system-level framework for designing agentic AI workflows. A general agentic architecture is introduced, and attacker and defender workflows are analyzed as coupled adaptive processes, and game-theoretic formulations are shown to provide a unifying design language for autonomy allocation, information flow, and temporal composition. Case studies in automated penetration testing, remediation, and cyber deception illustrate how equilibrium-based design enables system-level resiliency design.

摘要: 基于基础模型的人工智能正在从根本上重塑网络安全。大型语言模型现在能够实现自主规划、工具编排和大规模战略适应，这对构建在静态规则、边界防御和以人为本的工作流程上的安全架构构成挑战。本章主张从以预防为中心的安全性向代理式网络弹性转变。弹性系统必须预测中断、在攻击下维护关键功能、有效恢复并持续学习，而不是寻求完美的保护。我们在网络安全范式的历史演变中发现了这种转变，最终形成了一种人工智能增强的范式，在这种范式中，自主代理直接参与网络和网络物理系统的感知、推理、行动和适应。然后，我们开发了一个系统级框架，用于设计代理AI工作流。一个通用的代理体系结构的介绍，攻击者和防御者的工作流程进行了分析，耦合自适应过程，和博弈论的配方示出提供了一个统一的设计语言的自治分配，信息流，和时间组成。自动渗透测试、补救和网络欺骗的案例研究说明了基于平衡的设计如何实现系统级弹性设计。



## **13. Exploring the Security Threats of Retriever Backdoors in Retrieval-Augmented Code Generation**

探索检索增强代码生成中检索后门的安全威胁 cs.CR

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2512.21681v1) [paper-pdf](https://arxiv.org/pdf/2512.21681v1)

**Authors**: Tian Li, Bo Lin, Shangwen Wang, Yusong Tan

**Abstract**: Retrieval-Augmented Code Generation (RACG) is increasingly adopted to enhance Large Language Models for software development, yet its security implications remain dangerously underexplored. This paper conducts the first systematic exploration of a critical and stealthy threat: backdoor attacks targeting the retriever component, which represents a significant supply-chain vulnerability. It is infeasible to assess this threat realistically, as existing attack methods are either too ineffective to pose a real danger or are easily detected by state-of-the-art defense mechanisms spanning both latent-space analysis and token-level inspection, which achieve consistently high detection rates. To overcome this barrier and enable a realistic analysis, we first developed VenomRACG, a new class of potent and stealthy attack that serves as a vehicle for our investigation. Its design makes poisoned samples statistically indistinguishable from benign code, allowing the attack to consistently maintain low detectability across all evaluated defense mechanisms. Armed with this capability, our exploration reveals a severe vulnerability: by injecting vulnerable code equivalent to only 0.05% of the entire knowledge base size, an attacker can successfully manipulate the backdoored retriever to rank the vulnerable code in its top-5 results in 51.29% of cases. This translates to severe downstream harm, causing models like GPT-4o to generate vulnerable code in over 40% of targeted scenarios, while leaving the system's general performance intact. Our findings establish that retriever backdooring is not a theoretical concern but a practical threat to the software development ecosystem that current defenses are blind to, highlighting the urgent need for robust security measures.

摘要: 检索增强代码生成（RACG）越来越多地被采用来增强软件开发的大型语言模型，但其安全影响仍然没有得到充分的研究。本文首次系统地探索了一个关键且隐蔽的威胁：针对检索器组件的后门攻击，该组件代表了一个重大的供应链漏洞。现实地评估这种威胁是不可行的，因为现有的攻击方法要么太无效，无法构成真正的危险，要么很容易被涵盖潜在空间分析和代币级检查的最先进防御机制检测到，从而实现了一致的高检测率。为了克服这一障碍并进行现实的分析，我们首先开发了VenomRACG，这是一种新型的有效且隐蔽的攻击，可作为我们调查的工具。它的设计使得中毒样本在统计上与良性代码无法区分，从而使攻击在所有评估的防御机制中始终保持低可检测性。有了这种能力，我们的探索揭示了一个严重的漏洞：通过注入相当于整个知识库大小的0.05%的易受攻击的代码，攻击者可以成功地操纵后门检索器，在51.29%的情况下将易受攻击的代码排在前5名。这转化为严重的下游危害，导致像GPT-4 o这样的模型在超过40%的目标场景中生成易受攻击的代码，同时保持系统的一般性能不变。我们的研究结果表明，检索器后门不是一个理论问题，而是对软件开发生态系统的实际威胁，而当前的防御措施却视而不见，这突出了对强大安全措施的迫切需求。



## **14. LLM-Driven Feature-Level Adversarial Attacks on Android Malware Detectors**

LLM驱动的Android恶意软件检测器的冲突级对抗攻击 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21404v1) [paper-pdf](https://arxiv.org/pdf/2512.21404v1)

**Authors**: Tianwei Lan, Farid Naït-Abdesselam

**Abstract**: The rapid growth in both the scale and complexity of Android malware has driven the widespread adoption of machine learning (ML) techniques for scalable and accurate malware detection. Despite their effectiveness, these models remain vulnerable to adversarial attacks that introduce carefully crafted feature-level perturbations to evade detection while preserving malicious functionality. In this paper, we present LAMLAD, a novel adversarial attack framework that exploits the generative and reasoning capabilities of large language models (LLMs) to bypass ML-based Android malware classifiers. LAMLAD employs a dual-agent architecture composed of an LLM manipulator, which generates realistic and functionality-preserving feature perturbations, and an LLM analyzer, which guides the perturbation process toward successful evasion. To improve efficiency and contextual awareness, LAMLAD integrates retrieval-augmented generation (RAG) into the LLM pipeline. Focusing on Drebin-style feature representations, LAMLAD enables stealthy and high-confidence attacks against widely deployed Android malware detection systems. We evaluate LAMLAD against three representative ML-based Android malware detectors and compare its performance with two state-of-the-art adversarial attack methods. Experimental results demonstrate that LAMLAD achieves an attack success rate (ASR) of up to 97%, requiring on average only three attempts per adversarial sample, highlighting its effectiveness, efficiency, and adaptability in practical adversarial settings. Furthermore, we propose an adversarial training-based defense strategy that reduces the ASR by more than 30% on average, significantly enhancing model robustness against LAMLAD-style attacks.

摘要: Android恶意软件规模和复杂性的快速增长推动了机器学习（ML）技术的广泛采用，以进行可扩展和准确的恶意软件检测。尽管它们有效，但这些模型仍然容易受到对抗攻击，这些攻击引入精心设计的功能级扰动，以逃避检测，同时保留恶意功能。在本文中，我们介绍了LAMRAD，这是一种新型的对抗性攻击框架，它利用大型语言模型（LLM）的生成和推理能力来绕过基于ML的Android恶意软件分类器。LAMLAT采用双代理架构，由LLM操纵器和LLM分析器组成，LLM操纵器生成真实且功能保留的特征扰动，LLM分析器引导扰动过程成功规避。为了提高效率和上下文感知，LAMRAD将检索增强生成（RAG）集成到LLM管道中。LAMRAD专注于Drebin风格的特征表示，能够针对广泛部署的Android恶意软件检测系统进行隐蔽且高可信度的攻击。我们针对三种代表性的基于ML的Android恶意软件检测器评估LAMRAD，并将其性能与两种最先进的对抗攻击方法进行比较。实验结果表明，LAMRAD的攻击成功率（ASB）高达97%，每个对抗样本平均只需尝试三次，凸显了其在实际对抗环境中的有效性、效率和适应性。此外，我们提出了一种基于对抗训练的防御策略，平均将ASR降低了30%以上，显著增强了模型对LAMLAD攻击的鲁棒性。



## **15. SENTINEL: A Multi-Modal Early Detection Framework for Emerging Cyber Threats using Telegram**

SENTINEL：使用Telegram针对新兴网络威胁的多模式早期检测框架 cs.SI

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21380v1) [paper-pdf](https://arxiv.org/pdf/2512.21380v1)

**Authors**: Mohammad Hammas Saeed, Howie Huang

**Abstract**: Cyberattacks pose a serious threat to modern sociotechnical systems, often resulting in severe technical and societal consequences. Attackers commonly target systems and infrastructure through methods such as malware, ransomware, or other forms of technical exploitation. Most traditional mechanisms to counter these threats rely on post-hoc detection and mitigation strategies, responding to cyber incidents only after they occur rather than preventing them proactively. Recent trends reveal social media discussions can serve as reliable indicators for detecting such threats. Malicious actors often exploit online platforms to distribute attack tools, share attack knowledge and coordinate. Experts too, often predict ongoing attacks and discuss potential breaches in online spaces. In this work, we present SENTINEL, a framework that leverages social media signals for early detection of cyber attacks. SENTINEL aligns cybersecurity discussions to realworld cyber attacks leveraging multi modal signals, i.e., combining language modeling through large language models and coordination markers through graph neural networks. We use data from 16 public channels on Telegram related to cybersecurity and open source intelligence (OSINT) that span 365k messages. We highlight that social media discussions involve active dialogue around cyber threats and leverage SENTINEL to align the signals to real-world threats with an F1 of 0.89. Our work highlights the importance of leveraging language and network signals in predicting online threats.

摘要: 网络攻击对现代社会技术系统构成严重威胁，往往会造成严重的技术和社会后果。攻击者通常通过恶意软件、勒索软件或其他形式的技术利用等方法针对系统和基础设施。大多数应对这些威胁的传统机制都依赖于事后检测和缓解策略，仅在网络事件发生后才对其做出反应，而不是主动预防。最近的趋势表明，社交媒体讨论可以作为检测此类威胁的可靠指标。恶意行为者经常利用在线平台来分发攻击工具、共享攻击知识并进行协调。专家们也经常预测正在进行的攻击并讨论在线空间中潜在的漏洞。在这项工作中，我们介绍了SENTINEL，这是一个利用社交媒体信号来早期检测网络攻击的框架。SENTINEL利用多模式信号将网络安全讨论与现实世界的网络攻击保持一致，即将通过大型语言模型进行语言建模和通过图神经网络进行协调标记相结合。我们使用来自Telegram上16个公共频道的数据，这些频道与网络安全和开源情报（Osint）相关，涵盖365，000条消息。我们强调，社交媒体讨论涉及围绕网络威胁的积极对话，并利用SENTINEL将信号与现实世界的威胁保持一致，F1为0.89。我们的工作强调了利用语言和网络信号预测在线威胁的重要性。



## **16. Casting a SPELL: Sentence Pairing Exploration for LLM Limitation-breaking**

拼写：突破LLM限制的句子配对探索 cs.CR

Accepted to FSE 2026

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21236v1) [paper-pdf](https://arxiv.org/pdf/2512.21236v1)

**Authors**: Yifan Huang, Xiaojun Jia, Wenbo Guo, Yuqiang Sun, Yihao Huang, Chong Wang, Yang Liu

**Abstract**: Large language models (LLMs) have revolutionized software development through AI-assisted coding tools, enabling developers with limited programming expertise to create sophisticated applications. However, this accessibility extends to malicious actors who may exploit these powerful tools to generate harmful software. Existing jailbreaking research primarily focuses on general attack scenarios against LLMs, with limited exploration of malicious code generation as a jailbreak target. To address this gap, we propose SPELL, a comprehensive testing framework specifically designed to evaluate the weakness of security alignment in malicious code generation. Our framework employs a time-division selection strategy that systematically constructs jailbreaking prompts by intelligently combining sentences from a prior knowledge dataset, balancing exploration of novel attack patterns with exploitation of successful techniques. Extensive evaluation across three advanced code models (GPT-4.1, Claude-3.5, and Qwen2.5-Coder) demonstrates SPELL's effectiveness, achieving attack success rates of 83.75%, 19.38%, and 68.12% respectively across eight malicious code categories. The generated prompts successfully produce malicious code in real-world AI development tools such as Cursor, with outputs confirmed as malicious by state-of-the-art detection systems at rates exceeding 73%. These findings reveal significant security gaps in current LLM implementations and provide valuable insights for improving AI safety alignment in code generation applications.

摘要: 大型语言模型（LLM）通过人工智能辅助编码工具彻底改变了软件开发，使编程专业知识有限的开发人员能够创建复杂的应用程序。然而，这种可访问性扩展到恶意行为者，他们可能利用这些强大的工具来生成有害软件。现有的越狱研究主要关注针对LLM的一般攻击场景，对作为越狱目标的恶意代码生成的探索有限。为了解决这一差距，我们提出了STELL，这是一个全面的测试框架，专门用于评估恶意代码生成中安全一致的弱点。我们的框架采用时分选择策略，通过智能地组合来自先验知识数据集中的句子，平衡对新型攻击模式的探索与成功技术的利用，系统地构建越狱提示。对三种高级代码模型（GPT-4.1、Claude-3.5和Qwen 2.5-Coder）的广泛评估证明了SPELL的有效性，八种恶意代码类别的攻击成功率分别为83.75%、19.38%和68.12%。生成的提示在Cursor等现实世界的人工智能开发工具中成功生成恶意代码，最先进的检测系统确认输出为恶意代码，其比率超过73%。这些发现揭示了当前LLM实施中的重大安全漏洞，并为改进代码生成应用程序中的人工智能安全性提供了宝贵的见解。



## **17. GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs**

GateBreaker：对混合专家LLM的门引导攻击 cs.CR

Accepted by USENIX Security'26

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2512.21008v2) [paper-pdf](https://arxiv.org/pdf/2512.21008v2)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are increasingly deployed in critical domains, understanding and strengthening their alignment mechanisms is essential to prevent harmful outputs. However, existing LLM safety research has focused almost exclusively on dense architectures, leaving the unique safety properties of MoEs largely unexamined. The modular, sparsely-activated design of MoEs suggests that safety mechanisms may operate differently than in dense models, raising questions about their robustness.   In this paper, we present GateBreaker, the first training-free, lightweight, and architecture-agnostic attack framework that compromises the safety alignment of modern MoE LLMs at inference time. GateBreaker operates in three stages: (i) gate-level profiling, which identifies safety experts disproportionately routed on harmful inputs, (ii) expert-level localization, which localizes the safety structure within safety experts, and (iii) targeted safety removal, which disables the identified safety structure to compromise the safety alignment. Our study shows that MoE safety concentrates within a small subset of neurons coordinated by sparse routing. Selective disabling of these neurons, approximately 3% of neurons in the targeted expert layers, significantly increases the averaged attack success rate (ASR) from 7.4% to 64.9% against the eight latest aligned MoE LLMs with limited utility degradation. These safety neurons transfer across models within the same family, raising ASR from 17.9% to 67.7% with one-shot transfer attack. Furthermore, GateBreaker generalizes to five MoE vision language models (VLMs) with 60.9% ASR on unsafe image inputs.

摘要: 专家混合（MoE）架构通过仅激活每个输入的稀疏参数子集来推进大型语言模型（LLM）的扩展，从而在降低计算成本的情况下实现最先进的性能。随着这些模型越来越多地部署在关键领域，了解和加强它们的协调机制对于防止有害输出至关重要。然而，现有的LLM安全研究几乎完全集中在密集架构上，而MoE的独特安全属性在很大程度上没有得到审查。教育部的模块化、稀疏激活设计表明，安全机制的运作方式可能与密集模型不同，从而引发了对其稳健性的质疑。   在本文中，我们介绍了GateBreaker，这是第一个无需训练，轻量级和架构不可知的攻击框架，它在推理时损害了现代MoE LLM的安全对齐。Gateway Breaker分三个阶段运作：（i）门级分析，识别不成比例地基于有害输入的安全专家，（ii）专家级本地化，将安全结构本地化在安全专家中，以及（iii）有针对性的安全删除，禁用识别的安全结构以损害安全对齐。我们的研究表明，MoE的安全性集中在由稀疏路由协调的一小部分神经元中。选择性禁用这些神经元（目标专家层中约3%的神经元），将平均攻击成功率（ASR）从7.4%显著提高到64.9%，而八个最新对齐的MoE LLM的效用下降有限。这些安全神经元在同一家族内的模型之间转移，通过一次转移攻击将ASB从17.9%提高到67.7%。此外，Gateway Breaker还将五个MoE视觉语言模型（VLM）推广到不安全图像输入的ASB为60.9%。



## **18. AegisAgent: An Autonomous Defense Agent Against Prompt Injection Attacks in LLM-HARs**

AegisAgent：一种针对LLM-HARs中即时注入攻击的自主防御代理 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20986v1) [paper-pdf](https://arxiv.org/pdf/2512.20986v1)

**Authors**: Yihan Wang, Huanqi Yang, Shantanu Pal, Weitao Xu

**Abstract**: The integration of Large Language Models (LLMs) into wearable sensing is creating a new class of mobile applications capable of nuanced human activity understanding. However, the reliability of these systems is critically undermined by their vulnerability to prompt injection attacks, where attackers deliberately input deceptive instructions into LLMs. Traditional defenses, based on static filters and rigid rules, are insufficient to address the semantic complexity of these new attacks. We argue that a paradigm shift is needed -- from passive filtering to active protection and autonomous reasoning. We introduce AegisAgent, an autonomous agent system designed to ensure the security of LLM-driven HAR systems. Instead of merely blocking threats, AegisAgent functions as a cognitive guardian. It autonomously perceives potential semantic inconsistencies, reasons about the user's true intent by consulting a dynamic memory of past interactions, and acts by generating and executing a multi-step verification and repair plan. We implement AegisAgent as a lightweight, full-stack prototype and conduct a systematic evaluation on 15 common attacks with five state-of-the-art LLM-based HAR systems on three public datasets. Results show it reduces attack success rate by 30\% on average while incurring only 78.6 ms of latency overhead on a GPU workstation. Our work makes the first step towards building secure and trustworthy LLM-driven HAR systems.

摘要: 将大型语言模型（LLM）集成到可穿戴传感中正在创建一类能够细致入微地理解人类活动的新型移动应用程序。然而，这些系统的可靠性因其容易受到提示注入攻击（攻击者故意将欺骗性指令输入到LLM）而受到严重削弱。基于静态过滤器和严格规则的传统防御不足以解决这些新攻击的语义复杂性。我们认为需要进行范式转变--从被动过滤到主动保护和自主推理。我们引入了AegisAgent，这是一种自主代理系统，旨在确保LLM驱动的HAR系统的安全性。AegisAgent不仅仅是阻止威胁，而是充当认知守护者。它自主感知潜在的语义不一致，通过查阅过去交互的动态记忆来推理用户的真实意图，并通过生成和执行多步骤验证和修复计划来采取行动。我们将AegisAgent实施为轻量级全栈原型，并在三个公共数据集上使用五个最先进的基于LLM的HAR系统对15种常见攻击进行系统评估。结果显示，它平均将攻击成功率降低了30%，而在图形处理器上仅产生78.6 ms的延迟负担。我们的工作朝着构建安全且值得信赖的LLM驱动的HAR系统迈出了第一步。



## **19. The Imitation Game: Using Large Language Models as Chatbots to Combat Chat-Based Cybercrimes**

模仿游戏：使用大型语言模型作为聊天机器人来打击基于聊天的网络犯罪 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21371v1) [paper-pdf](https://arxiv.org/pdf/2512.21371v1)

**Authors**: Yifan Yao, Baojuan Wang, Jinhao Duan, Kaidi Xu, ChuanKai Guo, Zhibo Eric Sun, Yue Zhang

**Abstract**: Chat-based cybercrime has emerged as a pervasive threat, with attackers leveraging real-time messaging platforms to conduct scams that rely on trust-building, deception, and psychological manipulation. Traditional defense mechanisms, which operate on static rules or shallow content filters, struggle to identify these conversational threats, especially when attackers use multimedia obfuscation and context-aware dialogue.   In this work, we ask a provocative question inspired by the classic Imitation Game: Can machines convincingly pose as human victims to turn deception against cybercriminals? We present LURE (LLM-based User Response Engagement), the first system to deploy Large Language Models (LLMs) as active agents, not as passive classifiers, embedded within adversarial chat environments.   LURE combines automated discovery, adversarial interaction, and OCR-based analysis of image-embedded payment data. Applied to the setting of illicit video chat scams on Telegram, our system engaged 53 actors across 98 groups. In over 56 percent of interactions, the LLM maintained multi-round conversations without being noticed as a bot, effectively "winning" the imitation game. Our findings reveal key behavioral patterns in scam operations, such as payment flows, upselling strategies, and platform migration tactics.

摘要: 基于聊天的网络犯罪已成为一种普遍存在的威胁，攻击者利用实时消息平台来实施依赖于信任建立、欺骗和心理操纵的诈骗。传统防御机制基于静态规则或浅层内容过滤器，难以识别这些对话威胁，尤其是当攻击者使用多媒体混淆和上下文感知对话时。   在这部作品中，我们提出了一个受经典模仿游戏启发的挑衅性问题：机器能否令人信服地冒充人类受害者，利用欺骗手段对付网络犯罪分子？我们介绍了LURE（基于LLM的用户响应参与），这是第一个将大型语言模型（LLM）部署为嵌入在对抗性聊天环境中的主动代理而不是被动分类器的系统。   LURE结合了自动发现、对抗交互和基于OCR的图像嵌入式支付数据分析。应用于Telegram上的非法视频聊天诈骗设置，我们的系统涉及98个群组的53名参与者。在超过56%的互动中，LLM保持多轮对话，而不会被机器人注意到，有效地“赢得”了模仿游戏。我们的调查结果揭示了诈骗操作中的关键行为模式，例如支付流、向上销售策略和平台迁移策略。



## **20. ChatGPT: Excellent Paper! Accept It. Editor: Imposter Found! Review Rejected**

ChatGPT：优秀的纸张！接受吧。编辑：找到冒名顶替者！审查被拒绝 cs.CR

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2512.20405v2) [paper-pdf](https://arxiv.org/pdf/2512.20405v2)

**Authors**: Kanchon Gharami, Sanjiv Kumar Sarkar, Yongxin Liu, Shafika Showkat Moni

**Abstract**: Large Language Models (LLMs) like ChatGPT are now widely used in writing and reviewing scientific papers. While this trend accelerates publication growth and reduces human workload, it also introduces serious risks. Papers written or reviewed by LLMs may lack real novelty, contain fabricated or biased results, or mislead downstream research that others depend on. Such issues can damage reputations, waste resources, and even endanger lives when flawed studies influence medical or safety-critical systems. This research explores both the offensive and defensive sides of this growing threat. On the attack side, we demonstrate how an author can inject hidden prompts inside a PDF that secretly guide or "jailbreak" LLM reviewers into giving overly positive feedback and biased acceptance. On the defense side, we propose an "inject-and-detect" strategy for editors, where invisible trigger prompts are embedded into papers; if a review repeats or reacts to these triggers, it reveals that the review was generated by an LLM, not a human. This method turns prompt injections from vulnerability into a verification tool. We outline our design, expected model behaviors, and ethical safeguards for deployment. The goal is to expose how fragile today's peer-review process becomes under LLM influence and how editorial awareness can help restore trust in scientific evaluation.

摘要: ChatGPT等大型语言模型（LLM）现在广泛用于撰写和审查科学论文。虽然这一趋势加速了出版物的增长并减少了人力工作量，但也带来了严重的风险。LLM撰写或审查的论文可能缺乏真正的新颖性，包含捏造或有偏见的结果，或误导其他人所依赖的下游研究。当有缺陷的研究影响医疗或安全关键系统时，这些问题可能会损害声誉、浪费资源，甚至危及生命。这项研究探讨了这一日益增长的威胁的进攻和防守两方面。在攻击方面，我们演示了作者如何在PDF中注入隐藏提示，秘密引导或“越狱”LLM评审员提供过于积极的反馈和偏见的接受。在防御方面，我们为编辑提出了一种“注入并检测”策略，将隐形触发提示嵌入到论文中;如果评论重复或对这些触发做出反应，就表明该评论是由LLM而不是人类生成的。该方法将漏洞的提示注入到验证工具中。我们概述了我们的设计、预期的模型行为和部署的道德保障措施。目标是揭露当今的同行评审流程在法学硕士的影响下变得多么脆弱，以及编辑意识如何帮助恢复对科学评估的信任。



## **21. Optimistic TEE-Rollups: A Hybrid Architecture for Scalable and Verifiable Generative AI Inference on Blockchain**

乐观的TEE-Rollops：区块链上可扩展和可验证的生成式人工智能推理的混合架构 cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20176v1) [paper-pdf](https://arxiv.org/pdf/2512.20176v1)

**Authors**: Aaron Chan, Alex Ding, Frank Chen, Alan Wu, Bruce Zhang, Arther Tian

**Abstract**: The rapid integration of Large Language Models (LLMs) into decentralized physical infrastructure networks (DePIN) is currently bottlenecked by the Verifiability Trilemma, which posits that a decentralized inference system cannot simultaneously achieve high computational integrity, low latency, and low cost. Existing cryptographic solutions, such as Zero-Knowledge Machine Learning (ZKML), suffer from superlinear proving overheads (O(k NlogN)) that render them infeasible for billionparameter models. Conversely, optimistic approaches (opML) impose prohibitive dispute windows, preventing real-time interactivity, while recent "Proof of Quality" (PoQ) paradigms sacrifice cryptographic integrity for subjective semantic evaluation, leaving networks vulnerable to model downgrade attacks and reward hacking. In this paper, we introduce Optimistic TEE-Rollups (OTR), a hybrid verification protocol that harmonizes these constraints. OTR leverages NVIDIA H100 Confidential Computing Trusted Execution Environments (TEEs) to provide sub-second Provisional Finality, underpinned by an optimistic fraud-proof mechanism and stochastic Zero-Knowledge spot-checks to mitigate hardware side-channel risks. We formally define Proof of Efficient Attribution (PoEA), a consensus mechanism that cryptographically binds execution traces to hardware attestations, thereby guaranteeing model authenticity. Extensive simulations demonstrate that OTR achieves 99% of the throughput of centralized baselines with a marginal cost overhead of $0.07 per query, maintaining Byzantine fault tolerance against rational adversaries even in the presence of transient hardware vulnerabilities.

摘要: 大型语言模型（LLM）快速集成到去中心化物理基础设施网络（DePin）中目前受到可验证性三困境的限制，该困境认为去中心化推理系统无法同时实现高计算完整性、低延迟和低成本。现有的加密解决方案，例如零知识机器学习（ZKML），存在超线性证明费用（O（k NlogN））的问题，这使得它们对于十亿个参数模型来说不可行。相反，乐观方法（opML）施加了禁止性的争议窗口，阻止了实时交互，而最近的“质量证明”（PoQ）范式则牺牲了加密完整性来进行主观语义评估，使网络容易受到模型降级攻击和奖励黑客攻击。在本文中，我们介绍了乐观TEE-Rollup（OTR），这是一种协调这些约束的混合验证协议。OTR利用NVIDIA H100机密计算可信执行环境（TEE）提供亚秒级临时最终结果，并以乐观的防欺诈机制和随机零知识抽查为基础，以减轻硬件侧通道风险。我们正式定义了有效归因证明（PoEA），这是一种共识机制，通过加密方式将执行跟踪与硬件证明绑定，从而保证模型的真实性。广泛的模拟表明，OTR实现了集中式基线99%的吞吐量，每次查询的边际成本费用为0.07美元，即使存在暂时性硬件漏洞，也能对理性对手保持拜占庭式的耐药性。



## **22. Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography**

Odysseus：通过双重隐写术破解商业多模式法学硕士集成系统 cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20168v1) [paper-pdf](https://arxiv.org/pdf/2512.20168v1)

**Authors**: Songze Li, Jiameng Cheng, Yiming Li, Xiaojun Jia, Dacheng Tao

**Abstract**: By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems.

摘要: 通过将语言理解与图像等感知模式集成起来，多模式大型语言模型（MLLM）构成了现代人工智能系统的重要基础，特别是在开放和交互环境中运行的智能代理。然而，它们的可访问性不断增加也增加了滥用风险，例如产生有害或不安全内容。为了减轻这些风险，通常应用对齐技术来将模型行为与人类价值观对齐。尽管做出了这些努力，最近的研究表明，越狱攻击可能会绕过对齐并引发不安全的输出。目前，大多数现有的越狱方法都是针对开源模型量身定制的，并且对于商业MLLM集成系统（通常使用额外的过滤器）的有效性有限。这些过滤器可以检测和防止恶意输入和输出内容，从而显着减少越狱威胁。在本文中，我们揭示了这些安全过滤器的成功在很大程度上依赖于一个关键假设，即恶意内容必须在输入或输出中显式可见。这种假设虽然通常适用于传统的LLM集成系统，但在MLLM集成系统中却出现了问题，攻击者可以利用多种模式来隐藏对抗意图，从而导致现有的MLLM集成系统中出现错误的安全感。为了挑战这一假设，我们提出了Odysseus，一种新的越狱范例，它引入了双重隐写术来秘密地将恶意查询和响应嵌入到看起来很好的图像中。在基准数据集上进行的大量实验表明，我们的Odysseus成功地越狱了几个开创性和现实的MLLM集成系统，攻击成功率高达99%。它暴露了现有防御中的一个基本盲点，并呼吁重新思考MLLM集成系统中的跨模式安全性。



## **23. AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications**

超越核心领域的人工智能安全：简历筛选作为专业LLM应用中对抗漏洞的案例研究 cs.CL

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20164v1) [paper-pdf](https://arxiv.org/pdf/2512.20164v1)

**Authors**: Honglin Mu, Jinghao Liu, Kaiyang Wan, Rui Xing, Xiuying Chen, Timothy Baldwin, Wanxiang Che

**Abstract**: Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.

摘要: 大型语言模型（LLM）擅长文本理解和生成，非常适合代码审查和内容审核等自动化任务。然而，我们的研究发现了一个漏洞：LLM可能会被隐藏在输入数据（例如简历或代码）中的“对抗指令”操纵，导致它们偏离预期任务。值得注意的是，虽然代码审查等成熟领域可能存在防御措施，但在简历筛选和同行审查等其他常见应用中通常不存在防御措施。本文引入了一个基准来评估简历筛选中的此漏洞，揭示了某些攻击类型的攻击成功率超过80%。我们评估了两种防御机制：基于预算的防御可以减少10.1%的攻击，错误拒绝增加12.5%，而我们提出的使用LoRA适应的FIDS（通过分离的外部指令检测）可以减少15.4%的攻击，错误拒绝增加10.4%。组合方法可减少26.3%的攻击，证明训练时防御在安全性和实用程序保存方面优于推理时缓解措施。



## **24. ReGAIN: Retrieval-Grounded AI Framework for Network Traffic Analysis**

ReGAIN：用于网络流量分析的检索式人工智能框架 cs.LG

Accepted to ICNC 2026. This is the accepted author manuscript

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.22223v1) [paper-pdf](https://arxiv.org/pdf/2512.22223v1)

**Authors**: Shaghayegh Shajarian, Kennedy Marsh, James Benson, Sajad Khorsandroo, Mahmoud Abdelsalam

**Abstract**: Modern networks generate vast, heterogeneous traffic that must be continuously analyzed for security and performance. Traditional network traffic analysis systems, whether rule-based or machine learning-driven, often suffer from high false positives and lack interpretability, limiting analyst trust. In this paper, we present ReGAIN, a multi-stage framework that combines traffic summarization, retrieval-augmented generation (RAG), and Large Language Model (LLM) reasoning for transparent and accurate network traffic analysis. ReGAIN creates natural-language summaries from network traffic, embeds them into a multi-collection vector database, and utilizes a hierarchical retrieval pipeline to ground LLM responses with evidence citations. The pipeline features metadata-based filtering, MMR sampling, a two-stage cross-encoder reranking mechanism, and an abstention mechanism to reduce hallucinations and ensure grounded reasoning. Evaluated on ICMP ping flood and TCP SYN flood traces from the real-world traffic dataset, it demonstrates robust performance, achieving accuracy between 95.95% and 98.82% across different attack types and evaluation benchmarks. These results are validated against two complementary sources: dataset ground truth and human expert assessments. ReGAIN also outperforms rule-based, classical ML, and deep learning baselines while providing unique explainability through trustworthy, verifiable responses.

摘要: 现代网络会产生大量、异类的流量，必须持续分析这些流量的安全性和性能。传统的网络流量分析系统，无论是基于规则的还是机器学习驱动的，通常都会出现高误报且缺乏可解释性，从而限制了分析师的信任。在本文中，我们提出了ReGAIN，这是一个多阶段框架，它结合了流量总结、检索增强生成（RAG）和大型语言模型（LLM）推理，以实现透明和准确的网络流量分析。ReGAIN从网络流量中创建自然语言摘要，将其嵌入到多集合载体数据库中，并利用分层检索管道将LLM响应作为证据引用的基础。该管道具有基于元数据的过滤、MMR采样、两级交叉编码器重新排序机制和避免机制，以减少幻觉并确保接地推理。通过从现实世界流量数据集中同步洪水和TCPSEN洪水轨迹进行评估，它表现出稳健的性能，在不同的攻击类型和评估基准中实现了95.95%至98.82%的准确性。这些结果经过两个补充来源的验证：数据集基本真相和人类专家评估。ReGAIN还优于基于规则的经典ML和深度学习基线，同时通过值得信赖、可验证的响应提供独特的解释性。



## **25. Conditional Adversarial Fragility in Financial Machine Learning under Macroeconomic Stress**

宏观经济压力下金融机器学习中的条件对抗脆弱性 cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19935v1) [paper-pdf](https://arxiv.org/pdf/2512.19935v1)

**Authors**: Samruddhi Baviskar

**Abstract**: Machine learning models used in financial decision systems operate in nonstationary economic environments, yet adversarial robustness is typically evaluated under static assumptions. This work introduces Conditional Adversarial Fragility, a regime dependent phenomenon in which adversarial vulnerability is systematically amplified during periods of macroeconomic stress. We propose a regime aware evaluation framework for time indexed tabular financial classification tasks that conditions robustness assessment on external indicators of economic stress. Using volatility based regime segmentation as a proxy for macroeconomic conditions, we evaluate model behavior across calm and stress periods while holding model architecture, attack methodology, and evaluation protocols constant. Baseline predictive performance remains comparable across regimes, indicating that economic stress alone does not induce inherent performance degradation. Under adversarial perturbations, however, models operating during stress regimes exhibit substantially greater degradation across predictive accuracy, operational decision thresholds, and risk sensitive outcomes. We further demonstrate that this amplification propagates to increased false negative rates, elevating the risk of missed high risk cases during adverse conditions. To complement numerical robustness metrics, we introduce an interpretive governance layer based on semantic auditing of model explanations using large language models. Together, these results demonstrate that adversarial robustness in financial machine learning is a regime dependent property and motivate stress aware approaches to model risk assessment in high stakes financial deployments.

摘要: 金融决策系统中使用的机器学习模型在非平稳经济环境中运行，但对抗稳健性通常是在静态假设下评估的。这项工作引入了条件对抗脆弱性，这是一种依赖政权的现象，其中对抗脆弱性在宏观经济压力时期被系统性放大。我们提出了一个用于时间索引表格财务分类任务的制度意识评估框架，该框架以经济压力的外部指标为条件进行稳健性评估。使用基于波动性的制度分割作为宏观经济状况的代理，我们评估平静和压力时期的模型行为，同时保持模型架构、攻击方法和评估协议不变。不同制度之间的基线预测性能保持可比性，这表明经济压力本身不会导致固有的性能下降。然而，在对抗性扰动下，在压力制度下运行的模型在预测准确性、操作决策阈值和风险敏感结果方面表现出明显更大的退化。我们进一步证明，这种放大会传播到假阴性率增加，从而增加了在不利条件下错过高风险病例的风险。为了补充数字稳健性指标，我们引入了一个基于使用大型语言模型对模型解释进行语义审计的解释治理层。总而言之，这些结果表明，金融机器学习中的对抗稳健性是一种依赖于制度的属性，并激励压力感知方法对高风险金融部署中的风险评估进行建模。



## **26. Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models**

开放重量LoRA模型的Causes引导的去神经后门攻击 cs.CR

NDSS 2026

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19297v1) [paper-pdf](https://arxiv.org/pdf/2512.19297v1)

**Authors**: Linzhi Chen, Yang Sun, Hongru Wei, Yuqi Chen

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as an efficient method for fine-tuning large language models (LLMs) and is widely adopted within the open-source community. However, the decentralized dissemination of LoRA adapters through platforms such as Hugging Face introduces novel security vulnerabilities: malicious adapters can be easily distributed and evade conventional oversight mechanisms. Despite these risks, backdoor attacks targeting LoRA-based fine-tuning remain relatively underexplored. Existing backdoor attack strategies are ill-suited to this setting, as they often rely on inaccessible training data, fail to account for the structural properties unique to LoRA, or suffer from high false trigger rates (FTR), thereby compromising their stealth. To address these challenges, we propose Causal-Guided Detoxify Backdoor Attack (CBA), a novel backdoor attack framework specifically designed for open-weight LoRA models. CBA operates without access to original training data and achieves high stealth through two key innovations: (1) a coverage-guided data generation pipeline that synthesizes task-aligned inputs via behavioral exploration, and (2) a causal-guided detoxification strategy that merges poisoned and clean adapters by preserving task-critical neurons. Unlike prior approaches, CBA enables post-training control over attack intensity through causal influence-based weight allocation, eliminating the need for repeated retraining. Evaluated across six LoRA models, CBA achieves high attack success rates while reducing FTR by 50-70\% compared to baseline methods. Furthermore, it demonstrates enhanced resistance to state-of-the-art backdoor defenses, highlighting its stealth and robustness.

摘要: 低等级适应（LoRA）已成为微调大型语言模型（LLM）的一种有效方法，并在开源社区中广泛采用。然而，通过Hugging Face等平台分散传播LoRA适配器引入了新型安全漏洞：恶意适配器可以轻松分发并逃避传统的监督机制。尽管存在这些风险，但针对基于LoRA的微调的后门攻击仍然相对未充分研究。现有的后门攻击策略不适合这种设置，因为它们通常依赖于不可访问的训练数据，无法考虑LoRA特有的结构属性，或者遭受高错误触发率（FTR）的影响，从而损害了它们的隐形性。为了解决这些挑战，我们提出了一种专门为开放权重LoRA模型设计的新型后门攻击框架- CBA在不访问原始训练数据的情况下运行，并通过两项关键创新实现了高度隐身：（1）覆盖引导的数据生成管道，通过行为探索合成任务对齐的输入，以及（2）通过保留任务关键神经元合并中毒和清洁适配器的cavern-guided解毒策略。与以前的方法不同，CBA通过基于因果影响的权重分配实现了对攻击强度的训练后控制，从而消除了重复再训练的需要。在六个LoRA模型中进行评估，CBA实现了高攻击成功率，同时与基线方法相比将FTR降低了50- 70%。此外，它还表现出对最先进后门防御的增强抵抗力，凸显了其隐形性和稳健性。



## **27. DREAM: Dynamic Red-teaming across Environments for AI Models**

DREAM：人工智能模型跨环境的动态红色团队 cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19016v1) [paper-pdf](https://arxiv.org/pdf/2512.19016v1)

**Authors**: Liming Lu, Xiang Gu, Junyu Huang, Jiawei Du, Yunhuai Liu, Yongbin Zhou, Shuchao Pang

**Abstract**: Large Language Models (LLMs) are increasingly used in agentic systems, where their interactions with diverse tools and environments create complex, multi-stage safety challenges. However, existing benchmarks mostly rely on static, single-turn assessments that miss vulnerabilities from adaptive, long-chain attacks. To fill this gap, we introduce DREAM, a framework for systematic evaluation of LLM agents against dynamic, multi-stage attacks. At its core, DREAM uses a Cross-Environment Adversarial Knowledge Graph (CE-AKG) to maintain stateful, cross-domain understanding of vulnerabilities. This graph guides a Contextualized Guided Policy Search (C-GPS) algorithm that dynamically constructs attack chains from a knowledge base of 1,986 atomic actions across 349 distinct digital environments. Our evaluation of 12 leading LLM agents reveals a critical vulnerability: these attack chains succeed in over 70% of cases for most models, showing the power of stateful, cross-environment exploits. Through analysis of these failures, we identify two key weaknesses in current agents: contextual fragility, where safety behaviors fail to transfer across environments, and an inability to track long-term malicious intent. Our findings also show that traditional safety measures, such as initial defense prompts, are largely ineffective against attacks that build context over multiple interactions. To advance agent safety research, we release DREAM as a tool for evaluating vulnerabilities and developing more robust defenses.

摘要: 大型语言模型（LLM）越来越多地用于代理系统，它们与不同工具和环境的交互会带来复杂、多阶段的安全挑战。然而，现有的基准大多依赖于静态、单轮评估，这些评估会错过自适应性、长链攻击的漏洞。为了填补这一空白，我们引入了DREAM，这是一个针对动态、多阶段攻击系统评估LLM代理的框架。DREAM的核心是使用跨环境对抗知识图（CE-AKG）来维护对漏洞的有状态、跨领域理解。该图指导上下文引导政策搜索（C-GPS）算法，该算法根据349个不同数字环境中1，986个原子动作的知识库动态构建攻击链。我们对12个领先的LLM代理的评估揭示了一个关键漏洞：对于大多数模型来说，这些攻击链在超过70%的情况下都取得了成功，展示了有状态、跨环境漏洞利用的力量。通过对这些失败的分析，我们发现了当前代理的两个关键弱点：上下文脆弱性，即安全行为无法跨环境转移，以及无法跟踪长期恶意意图。我们的研究结果还表明，传统的安全措施（例如初始防御提示）对于在多重交互中建立上下文的攻击在很大程度上无效。为了推进代理安全研究，我们发布DREAM作为评估漏洞和开发更强大防御的工具。



## **28. Efficient Jailbreak Mitigation Using Semantic Linear Classification in a Multi-Staged Pipeline**

在多阶段管道中使用语义线性分类的高效越狱缓解 cs.CR

Under Review

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19011v1) [paper-pdf](https://arxiv.org/pdf/2512.19011v1)

**Authors**: Akshaj Prashanth Rao, Advait Singh, Saumya Kumaar Saksena, Dhruv Kumar

**Abstract**: Prompt injection and jailbreaking attacks pose persistent security challenges to large language model (LLM)-based systems. We present an efficient and systematically evaluated defense architecture that mitigates these threats through a lightweight, multi-stage pipeline. Its core component is a semantic filter based on text normalization, TF-IDF representations, and a Linear SVM classifier. Despite its simplicity, this module achieves 93.4% accuracy and 96.5% specificity on held-out data, substantially reducing attack throughput while incurring negligible computational overhead.   Building on this efficient foundation, the full pipeline integrates complementary detection and mitigation mechanisms that operate at successive stages, providing strong robustness with minimal latency. In comparative experiments, our SVM-based configuration improves overall accuracy from 35.1% to 93.4% while reducing average time to completion from approximately 450s to 47s, yielding over 10 times lower latency than ShieldGemma. These results demonstrate that the proposed design simultaneously advances defensive precision and efficiency, addressing a core limitation of current model-based moderators.   Evaluation across a curated corpus of over 30,000 labeled prompts, including benign, jailbreak, and application-layer injections, confirms that staged, resource-efficient defenses can robustly secure modern LLM-driven applications.

摘要: 提示注入和越狱攻击对基于大型语言模型（LLM）的系统构成了持续的安全挑战。我们提出了一种高效且系统化评估的防御架构，可以通过轻量级的多阶段管道减轻这些威胁。其核心组件是基于文本规范化、TF-IDF表示和线性支持量分类器的语义过滤器。尽管它很简单，但该模块在保存的数据上实现了93.4%的准确性和96.5%的特异性，大大降低了攻击吞吐量，同时产生的计算负担可以忽略不计。   在这个高效的基础上，完整的管道集成了在连续阶段运行的补充检测和缓解机制，以最小的延迟提供强大的鲁棒性。在比较实验中，我们基于支持机的配置将总体准确性从35.1%提高到93.4%，同时将平均完成时间从约450秒减少到47秒，延迟时间比ShieldGemma低10倍以上。这些结果表明，拟议的设计同时提高了防御精度和效率，解决了当前基于模型的版主的核心限制。   对30，000多个带标签的提示（包括良性、越狱和应用层注入）的精心策划的数据库进行评估，证实了分阶段的资源高效防御可以强大地保护现代LLM驱动的应用程序。



## **29. Automated Red-Teaming Framework for Large Language Model Security Assessment: A Comprehensive Attack Generation and Detection System**

用于大型语言模型安全评估的自动化Red-Teaming框架：全面的攻击生成和检测系统 cs.CR

18 pages

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.20677v1) [paper-pdf](https://arxiv.org/pdf/2512.20677v1)

**Authors**: Zhang Wei, Peilu Hu, Shengning Lang, Hao Yan, Li Mei, Yichao Zhang, Chen Yang, Junfeng Hao, Zhimo Han

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes domains, ensuring their security and alignment has become a critical challenge. Existing red-teaming practices depend heavily on manual testing, which limits scalability and fails to comprehensively cover the vast space of potential adversarial behaviors. This paper introduces an automated red-teaming framework that systematically generates, executes, and evaluates adversarial prompts to uncover security vulnerabilities in LLMs. Our framework integrates meta-prompting-based attack synthesis, multi-modal vulnerability detection, and standardized evaluation protocols spanning six major threat categories -- reward hacking, deceptive alignment, data exfiltration, sandbagging, inappropriate tool use, and chain-of-thought manipulation. Experiments on the GPT-OSS-20B model reveal 47 distinct vulnerabilities, including 21 high-severity and 12 novel attack patterns, achieving a $3.9\times$ improvement in vulnerability discovery rate over manual expert testing while maintaining 89\% detection accuracy. These results demonstrate the framework's effectiveness in enabling scalable, systematic, and reproducible AI safety evaluations. By providing actionable insights for improving alignment robustness, this work advances the state of automated LLM red-teaming and contributes to the broader goal of building secure and trustworthy AI systems.

摘要: 随着大型语言模型（LLM）越来越多地部署在高风险领域，确保它们的安全性和一致性已成为一项关键挑战。现有的红色团队实践严重依赖手动测试，这限制了可扩展性，并且无法全面覆盖潜在对抗行为的巨大空间。本文介绍了一个自动化红色团队框架，该框架系统地生成、执行和评估对抗提示，以发现LLC中的安全漏洞。我们的框架集成了基于元提示的攻击合成、多模式漏洞检测和跨越六个主要威胁类别的标准化评估协议-奖励黑客攻击、欺骗性对齐、数据泄露、沙袋、不当工具使用和思想链操纵。GPT-OSS-20 B模型上的实验揭示了47个不同的漏洞，包括21个高严重性和12个新型攻击模式，与手动专家测试相比，漏洞发现率提高了3.9倍，同时保持89%的检测准确率。这些结果证明了该框架在实现可扩展、系统和可重复的人工智能安全评估方面的有效性。通过为提高对齐稳健性提供可操作的见解，这项工作推进了LLM自动化红色团队的状态，并有助于构建安全且值得信赖的人工智能系统的更广泛目标。



## **30. VizDefender: Unmasking Visualization Tampering through Proactive Localization and Intent Inference**

VizDefender：通过主动本地化和意图推理揭开可视化篡改的面纱 cs.CV

IEEE Transactions on Visualization and Computer Graphics (IEEE PacificVis'26 TVCG Track)

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18853v1) [paper-pdf](https://arxiv.org/pdf/2512.18853v1)

**Authors**: Sicheng Song, Yanjie Zhang, Zixin Chen, Huamin Qu, Changbo Wang, Chenhui Li

**Abstract**: The integrity of data visualizations is increasingly threatened by image editing techniques that enable subtle yet deceptive tampering. Through a formative study, we define this challenge and categorize tampering techniques into two primary types: data manipulation and visual encoding manipulation. To address this, we present VizDefender, a framework for tampering detection and analysis. The framework integrates two core components: 1) a semi-fragile watermark module that protects the visualization by embedding a location map to images, which allows for the precise localization of tampered regions while preserving visual quality, and 2) an intent analysis module that leverages Multimodal Large Language Models (MLLMs) to interpret manipulation, inferring the attacker's intent and misleading effects. Extensive evaluations and user studies demonstrate the effectiveness of our methods.

摘要: 数据可视化的完整性越来越受到图像编辑技术的威胁，这些技术可以进行微妙但具有欺骗性的篡改。通过形成性研究，我们定义了这一挑战，并将篡改技术分为两种主要类型：数据操纵和视觉编码操纵。为了解决这个问题，我们提出了VizDefender，这是一个用于篡改检测和分析的框架。该框架集成了两个核心组件：1）半脆弱水印模块，通过将位置地图嵌入到图像中来保护可视化，这允许在保留视觉质量的同时精确定位被篡改的区域，2）意图分析模块，利用多模式大型语言模型（MLLM）来解释操纵，推断攻击者的意图和误导性效果。广泛的评估和用户研究证明了我们方法的有效性。



## **31. MEEA: Mere Exposure Effect-Driven Confrontational Optimization for LLM Jailbreaking**

MEEA：LLM越狱的纯粹曝光驱动的对抗优化 cs.AI

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18755v1) [paper-pdf](https://arxiv.org/pdf/2512.18755v1)

**Authors**: Jianyi Zhang, Shizhao Liu, Ziyin Zhou, Zhen Li

**Abstract**: The rapid advancement of large language models (LLMs) has intensified concerns about the robustness of their safety alignment. While existing jailbreak studies explore both single-turn and multi-turn strategies, most implicitly assume a static safety boundary and fail to account for how contextual interactions dynamically influence model behavior, leading to limited stability and generalization. Motivated by this gap, we propose MEEA (Mere Exposure Effect Attack), a psychology-inspired, fully automated black-box framework for evaluating multi-turn safety robustness, grounded in the mere exposure effect. MEEA leverages repeated low-toxicity semantic exposure to induce a gradual shift in a model's effective safety threshold, enabling progressive erosion of alignment constraints over sustained interactions. Concretely, MEEA constructs semantically progressive prompt chains and optimizes them using a simulated annealing strategy guided by semantic similarity, toxicity, and jailbreak effectiveness. Extensive experiments on both closed-source and open-source models, including GPT-4, Claude-3.5, and DeepSeek-R1, demonstrate that MEEA consistently achieves higher attack success rates than seven representative baselines, with an average Attack Success Rate (ASR) improvement exceeding 20%. Ablation studies further validate the necessity of both annealing-based optimization and contextual exposure mechanisms. Beyond improved attack effectiveness, our findings indicate that LLM safety behavior is inherently dynamic and history-dependent, challenging the common assumption of static alignment boundaries and highlighting the need for interaction-aware safety evaluation and defense mechanisms. Our code is available at: https://github.com/Carney-lsz/MEEA

摘要: 大型语言模型（LLM）的快速发展加剧了人们对其安全对齐稳健性的担忧。虽然现有的越狱研究探索了单转向和多转向策略，但大多数都隐含地假设静态安全边界，并且未能考虑上下文相互作用如何动态影响模型行为，从而导致稳定性和概括性有限。出于这一差距的动机，我们提出了MEEA（纯粹暴露效应攻击），这是一个受心理学启发的全自动黑匣子框架，用于评估多回合安全稳健性，基于纯粹的暴露效应。MEEA利用重复的低毒性语义暴露来诱导模型的有效安全阈值的逐渐转变，从而使对齐约束在持续相互作用中逐渐受到侵蚀。具体地说，MEEA构造语义渐进提示链，并使用模拟退火策略进行优化，指导语义相似性，毒性和越狱有效性。在包括GPT-4、Claude-3.5和DeepSeek-R1在内的闭源和开源模型上进行的广泛实验表明，MEEA始终比七个代表性基线实现更高的攻击成功率，平均攻击成功率（ASR）提高超过20%。消融研究进一步验证了基于退火的优化和环境暴露机制的必要性。除了提高攻击效率，我们的研究结果表明，LLM安全行为本质上是动态的和历史依赖的，挑战了静态对齐边界的常见假设，并强调了交互感知安全评估和防御机制的必要性。我们的代码可访问：https://github.com/Carney-lsz/MEEA



## **32. Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection**

通过双层图异常检测对LLM多智能体系统进行可解释和细粒度保护 cs.CR

14 pages, 3 tables, 5 figures

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18733v1) [paper-pdf](https://arxiv.org/pdf/2512.18733v1)

**Authors**: Junjun Pan, Yixin Liu, Rui Miao, Kaize Ding, Yu Zheng, Quoc Viet Hung Nguyen, Alan Wee-Chung Liew, Shirui Pan

**Abstract**: Large language model (LLM)-based multi-agent systems (MAS) have shown strong capabilities in solving complex tasks. As MAS become increasingly autonomous in various safety-critical tasks, detecting malicious agents has become a critical security concern. Although existing graph anomaly detection (GAD)-based defenses can identify anomalous agents, they mainly rely on coarse sentence-level information and overlook fine-grained lexical cues, leading to suboptimal performance. Moreover, the lack of interpretability in these methods limits their reliability and real-world applicability. To address these limitations, we propose XG-Guard, an explainable and fine-grained safeguarding framework for detecting malicious agents in MAS. To incorporate both coarse and fine-grained textual information for anomalous agent identification, we utilize a bi-level agent encoder to jointly model the sentence- and token-level representations of each agent. A theme-based anomaly detector further captures the evolving discussion focus in MAS dialogues, while a bi-level score fusion mechanism quantifies token-level contributions for explanation. Extensive experiments across diverse MAS topologies and attack scenarios demonstrate robust detection performance and strong interpretability of XG-Guard.

摘要: 基于大型语言模型（LLM）的多智能体系统（MAS）在解决复杂任务方面表现出了强大的能力。随着MAS在各种安全关键任务中变得越来越自治，检测恶意代理已成为一个关键的安全问题。尽管现有的基于图形异常检测（GAD）的防御可以识别异常代理，但它们主要依赖于粗略的业务级别信息并忽视细粒度的词汇线索，导致性能次优。此外，这些方法缺乏可解释性限制了它们的可靠性和现实世界的适用性。为了解决这些限制，我们提出了XG-Guard，这是一个可解释且细粒度的保护框架，用于检测MAS中的恶意代理。为了结合粗粒度和细粒度的文本信息来识别异常代理，我们利用两级代理编码器来联合建模每个代理的句子级和符号级表示。基于主题的异常检测器进一步捕捉MAS对话中不断变化的讨论焦点，而两级分数融合机制量化代币级贡献以进行解释。跨各种MAS布局和攻击场景的广泛实验证明了XG-Guard的稳健检测性能和强大的可解释性。



## **33. Breaking Minds, Breaking Systems: Jailbreaking Large Language Models via Human-like Psychological Manipulation**

打破思维，打破系统：通过类人心理操纵越狱大型语言模型 cs.CR

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18244v1) [paper-pdf](https://arxiv.org/pdf/2512.18244v1)

**Authors**: Zehao Liu, Xi Lin

**Abstract**: Large Language Models (LLMs) have gained considerable popularity and protected by increasingly sophisticated safety mechanisms. However, jailbreak attacks continue to pose a critical security threat by inducing models to generate policy-violating behaviors. Current paradigms focus on input-level anomalies, overlooking that the model's internal psychometric state can be systematically manipulated. To address this, we introduce Psychological Jailbreak, a new jailbreak attack paradigm that exposes a stateful psychological attack surface in LLMs, where attackers exploit the manipulation of a model's psychological state across interactions. Building on this insight, we propose Human-like Psychological Manipulation (HPM), a black-box jailbreak method that dynamically profiles a target model's latent psychological vulnerabilities and synthesizes tailored multi-turn attack strategies. By leveraging the model's optimization for anthropomorphic consistency, HPM creates a psychological pressure where social compliance overrides safety constraints. To systematically measure psychological safety, we construct an evaluation framework incorporating psychometric datasets and the Policy Corruption Score (PCS). Benchmarking against various models (e.g., GPT-4o, DeepSeek-V3, Gemini-2-Flash), HPM achieves a mean Attack Success Rate (ASR) of 88.1%, outperforming state-of-the-art attack baselines. Our experiments demonstrate robust penetration against advanced defenses, including adversarial prompt optimization (e.g., RPO) and cognitive interventions (e.g., Self-Reminder). Ultimately, PCS analysis confirms HPM induces safety breakdown to satisfy manipulated contexts. Our work advocates for a fundamental paradigm shift from static content filtering to psychological safety, prioritizing the development of psychological defense mechanisms against deep cognitive manipulation.

摘要: 大型语言模型（LLM）已经相当受欢迎，并受到日益复杂的安全机制的保护。然而，越狱攻击通过诱导模型产生违反政策的行为，继续构成严重的安全威胁。当前的范式专注于输入级异常，忽视了模型的内部心理测量状态可以被系统性操纵。为了解决这个问题，我们引入了心理越狱，这是一种新的越狱攻击范式，它暴露了LLM中的状态心理攻击表面，攻击者利用交互中对模型心理状态的操纵。基于这一见解，我们提出了类人心理操纵（HPM），这是一种黑匣子越狱方法，可以动态地描述目标模型的潜在心理脆弱性并综合量身定制的多回合攻击策略。通过利用模型对拟人化一致性的优化，HPM创造了一种心理压力，社会合规性凌驾于安全约束之上。为了系统性地衡量心理安全，我们构建了一个纳入心理测量数据集和政策腐败评分（PCS）的评估框架。针对各种模型（例如，GPT-4 o、DeepSeek-V3、Gemini-2-Flash），HPM的平均攻击成功率（ASB）为88.1%，优于最先进的攻击基线。我们的实验证明了对高级防御的强大渗透，包括对抗性即时优化（例如，LPO）和认知干预（例如，自我提醒）。最终，PCS分析证实HPM会引发安全崩溃以满足操纵环境。我们的工作倡导从静态内容过滤到心理安全的根本范式转变，优先考虑开发针对深度认知操纵的心理防御机制。



## **34. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

有害在线内容的对抗稳健检测：计算设计科学方法 cs.LG

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.17367v3) [paper-pdf](https://arxiv.org/pdf/2512.17367v3)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.

摘要: 社交媒体平台受到仇恨言论、错误信息和极端主义言论等有害内容的困扰。机器学习（ML）模型被广泛采用来检测此类内容;然而，它们仍然极易受到对抗攻击，其中恶意用户会巧妙地修改文本以逃避检测。因此，增强对抗鲁棒性至关重要，需要检测器能够抵御各种攻击（可概括性），同时保持高的总体准确性。然而，同时实现最佳概括性和准确性是一项挑战。遵循计算设计科学范式，本研究采用顺序方法，首先提出了一种新颖的框架（基于大语言模型的样本生成和聚合，LLM-LGA），通过识别文本对抗攻击的关键不变性并利用它们来确保框架内实例化的检测器具有很强的概括性。其次，我们实例化我们的检测器（对抗鲁棒有害在线内容检测器，ARHOCD）具有三个新颖的设计组件来提高检测准确性：（1）利用其互补优势的多个基本检测器的集成;（2）一种新颖的权重分配方法，其基于每个样本的可预测性和每个碱基检测器的能力动态调整权重，权重使用领域知识初始化并通过Bayesian推理更新;以及（3）一种新颖的对抗训练策略，迭代优化基本检测器和权重分配器。我们解决了现有对抗鲁棒性增强研究的几个局限性，并在跨越仇恨言论、谣言和极端主义内容的三个数据集中对ARHOCD进行了实证评估。结果表明，ARHOCD具有很强的概括性，并提高了对抗条件下的检测准确性。



## **35. In-Context Probing for Membership Inference in Fine-Tuned Language Models**

精调语言模型中成员推理的上下文探索 cs.CR

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.16292v2) [paper-pdf](https://arxiv.org/pdf/2512.16292v2)

**Authors**: Zhexi Lu, Hongliang Chi, Nathalie Baracaldo, Swanand Ravindra Kadhe, Yuseok Jeon, Lei Yu

**Abstract**: Membership inference attacks (MIAs) pose a critical privacy threat to fine-tuned large language models (LLMs), especially when models are adapted to domain-specific tasks using sensitive data. While prior black-box MIA techniques rely on confidence scores or token likelihoods, these signals are often entangled with a sample's intrinsic properties - such as content difficulty or rarity - leading to poor generalization and low signal-to-noise ratios. In this paper, we propose ICP-MIA, a novel MIA framework grounded in the theory of training dynamics, particularly the phenomenon of diminishing returns during optimization. We introduce the Optimization Gap as a fundamental signal of membership: at convergence, member samples exhibit minimal remaining loss-reduction potential, while non-members retain significant potential for further optimization. To estimate this gap in a black-box setting, we propose In-Context Probing (ICP), a training-free method that simulates fine-tuning-like behavior via strategically constructed input contexts. We propose two probing strategies: reference-data-based (using semantically similar public samples) and self-perturbation (via masking or generation). Experiments on three tasks and multiple LLMs show that ICP-MIA significantly outperforms prior black-box MIAs, particularly at low false positive rates. We further analyze how reference data alignment, model type, PEFT configurations, and training schedules affect attack effectiveness. Our findings establish ICP-MIA as a practical and theoretically grounded framework for auditing privacy risks in deployed LLMs.

摘要: 成员资格推理攻击（MIA）对微调的大型语言模型（LLM）构成严重的隐私威胁，尤其是当模型使用敏感数据适应特定领域任务时。虽然先前的黑匣子MIA技术依赖于置信度分数或代币可能性，但这些信号通常与样本的内在属性（例如内容难度或稀有性）纠缠在一起，导致概括性较差和低信噪比。在本文中，我们提出了ICP-MIA，这是一种基于训练动力学理论的新型MIA框架，特别是优化过程中的回报递减现象。我们引入优化差距作为成员资格的基本信号：在收敛时，成员样本表现出最小的剩余损失减少潜力，而非成员保留了进一步优化的显着潜力。为了估计黑匣子环境中的这一差距，我们提出了In-Context Probing（ICP），这是一种免训练的方法，通过策略性构建的输入上下文模拟类似微调的行为。我们提出了两种探测策略：基于参考数据（使用语义相似的公共样本）和自我扰动（通过掩蔽或生成）。三项任务和多个LLM的实验表明，ICP-MIA显着优于之前的黑匣子MIA，尤其是在低假阳性率下。我们进一步分析参考数据对齐、模型类型、PEFT配置和训练计划如何影响攻击有效性。我们的研究结果将ICP-MIA确立为审计已部署的LLM隐私风险的实用且理论基础的框架。



## **36. APT-CGLP: Advanced Persistent Threat Hunting via Contrastive Graph-Language Pre-Training**

APT-CGM：通过对比图形语言预训练进行高级持续威胁狩猎 cs.CR

Accepted by SIGKDD 2026 Research Track

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2511.20290v2) [paper-pdf](https://arxiv.org/pdf/2511.20290v2)

**Authors**: Xuebo Qiu, Mingqi Lv, Yimei Zhang, Tieming Chen, Tiantian Zhu, Qijie Song, Shouling Ji

**Abstract**: Provenance-based threat hunting identifies Advanced Persistent Threats (APTs) on endpoints by correlating attack patterns described in Cyber Threat Intelligence (CTI) with provenance graphs derived from system audit logs. A fundamental challenge in this paradigm lies in the modality gap -- the structural and semantic disconnect between provenance graphs and CTI reports. Prior work addresses this by framing threat hunting as a graph matching task: 1) extracting attack graphs from CTI reports, and 2) aligning them with provenance graphs. However, this pipeline incurs severe \textit{information loss} during graph extraction and demands intensive manual curation, undermining scalability and effectiveness.   In this paper, we present APT-CGLP, a novel cross-modal APT hunting system via Contrastive Graph-Language Pre-training, facilitating end-to-end semantic matching between provenance graphs and CTI reports without human intervention. First, empowered by the Large Language Model (LLM), APT-CGLP mitigates data scarcity by synthesizing high-fidelity provenance graph-CTI report pairs, while simultaneously distilling actionable insights from noisy web-sourced CTIs to improve their operational utility. Second, APT-CGLP incorporates a tailored multi-objective training algorithm that synergizes contrastive learning with inter-modal masked modeling, promoting cross-modal attack semantic alignment at both coarse- and fine-grained levels. Extensive experiments on four real-world APT datasets demonstrate that APT-CGLP consistently outperforms state-of-the-art threat hunting baselines in terms of accuracy and efficiency.

摘要: 基于源的威胁狩猎通过将网络威胁情报（RTI）中描述的攻击模式与从系统审计日志中获得的源源图关联来识别端点上的高级持续性威胁（APT）。该范式的一个根本挑战在于形式差距--出处图和RTI报告之间的结构和语义脱节。之前的工作通过将威胁狩猎框架为图匹配任务来解决这个问题：1）从RTI报告中提取攻击图，以及2）将它们与出处图对齐。然而，该管道在图形提取过程中会导致严重的\textit{信息损失}，并且需要密集的手动策展，从而削弱了可扩展性和有效性。   在本文中，我们介绍了APT-CGM，这是一种通过对比图语言预训练的新型跨模式APT狩猎系统，可以在无需人为干预的情况下促进出处图和RTI报告之间的端到端语义匹配。首先，在大型语言模型（LLM）的支持下，APT-CGLP通过合成高保真出处图形-RTI报告对来缓解数据稀缺性，同时从有噪音的网络来源的RTI中提取可操作的见解以提高其运营效用。其次，APT-CGLP结合了定制的多目标训练算法，该算法将对比学习与模式间掩蔽建模协同作用，促进粗粒度和细粒度级别的跨模式攻击语义对齐。对四个现实世界APT数据集的广泛实验表明，APT-CGM在准确性和效率方面始终优于最先进的威胁搜寻基线。



## **37. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

AutoAdv：大型语言模型多回合越狱的自动对抗预算 cs.CL

Presented at NeurIPS 2025 Lock-LLM Workshop. Code is available at https://github.com/AAN-AutoAdv/AutoAdv

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2511.02376v3) [paper-pdf](https://arxiv.org/pdf/2511.02376v3)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs. Yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves an attack success rate of up to 95% on Llama-3.1-8B within six turns, a 24% improvement over single-turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests and then iteratively refines them. Extensive evaluation across commercial and open-source models (Llama-3.1-8B, GPT-4o mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，对抗性提示会引发有害输出。然而，大多数评估都集中在单轮交互上，而现实世界的攻击则通过自适应多轮对话展开。我们介绍了AutoAdv，这是一个用于自动多回合越狱的免训练框架，在六个回合内对Llama-3.1-8B的攻击成功率高达95%，比单回合基线提高了24%。AutoAdv独特地结合了三种自适应机制：从成功的攻击中学习以增强未来提示的模式管理器、根据失败模式动态调整采样参数的温度管理器以及掩盖有害请求然后迭代细化的两阶段重写策略。对商业和开源模型（Llama-3.1-8B、GPT-4 o mini、Qwen 3 - 235 B、Mistral-7 B）的广泛评估揭示了当前安全机制中存在的持续漏洞，多回合攻击的表现始终优于单回合方法。这些发现表明，针对单轮交互优化的对齐策略无法在扩展对话中保持稳健性，凸显了对多轮感知防御的迫切需求。



## **38. FuncPoison: Poisoning Function Library to Hijack Multi-agent Autonomous Driving Systems**

FuncPoison：劫持多智能体自动驾驶系统的中毒函数库 cs.CR

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2509.24408v2) [paper-pdf](https://arxiv.org/pdf/2509.24408v2)

**Authors**: Yuzhen Long, Songze Li

**Abstract**: Autonomous driving systems increasingly rely on multi-agent architectures powered by large language models (LLMs), where specialized agents collaborate to perceive, reason, and plan. A key component of these systems is the shared function library, a collection of software tools that agents use to process sensor data and navigate complex driving environments. Despite its critical role in agent decision-making, the function library remains an under-explored vulnerability. In this paper, we introduce FuncPoison, a novel poisoning-based attack targeting the function library to manipulate the behavior of LLM-driven multi-agent autonomous systems. FuncPoison exploits two key weaknesses in how agents access the function library: (1) agents rely on text-based instructions to select tools; and (2) these tools are activated using standardized command formats that attackers can replicate. By injecting malicious tools with deceptive instructions, FuncPoison manipulates one agent s decisions--such as misinterpreting road conditions--triggering cascading errors that mislead other agents in the system. We experimentally evaluate FuncPoison on two representative multi-agent autonomous driving systems, demonstrating its ability to significantly degrade trajectory accuracy, flexibly target specific agents to induce coordinated misbehavior, and evade diverse defense mechanisms. Our results reveal that the function library, often considered a simple toolset, can serve as a critical attack surface in LLM-based autonomous driving systems, raising elevated concerns on their reliability.

摘要: 自动驾驶系统越来越依赖于由大型语言模型（LLM）驱动的多智能体架构，其中专门的智能体协作感知，推理和计划。这些系统的一个关键组成部分是共享功能库，这是一组软件工具，代理用于处理传感器数据和导航复杂的驾驶环境。尽管其在代理决策中的关键作用，函数库仍然是一个未充分开发的漏洞。在本文中，我们介绍了FuncPoison，一种新的基于中毒的攻击目标函数库来操纵LLM驱动的多智能体自治系统的行为。FuncPoison利用了代理访问函数库的两个关键弱点：（1）代理依赖基于文本的指令来选择工具;（2）这些工具使用攻击者可以复制的标准化命令格式激活。通过注入带有欺骗性指令的恶意工具，FuncPoison操纵一个代理的决策（例如误解道路状况），从而引发连锁错误，误导系统中的其他代理。我们在两个有代表性的多智能体自主驾驶系统上对FuncPoison进行了实验评估，证明了它能够显着降低轨迹准确性、灵活地针对特定智能体以诱导协调不当行为以及逃避各种防御机制。我们的结果表明，通常被认为是一个简单的工具集的功能库可以作为基于LLM的自动驾驶系统中的关键攻击面，从而引发了对其可靠性的高度担忧。



## **39. Secure and Efficient Access Control for Computer-Use Agents via Context Space**

通过上下文空间对计算机使用代理进行安全有效的访问控制 cs.CR

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2509.22256v3) [paper-pdf](https://arxiv.org/pdf/2509.22256v3)

**Authors**: Haochen Gong, Chenxiao Li, Rui Chang, Wenbo Shen

**Abstract**: Large language model (LLM)-based computer-use agents represent a convergence of AI and OS capabilities, enabling natural language to control system- and application-level functions. However, due to LLMs' inherent uncertainty issues, granting agents control over computers poses significant security risks. When agent actions deviate from user intentions, they can cause irreversible consequences. Existing mitigation approaches, such as user confirmation and LLM-based dynamic action validation, still suffer from limitations in usability, security, and performance. To address these challenges, we propose CSAgent, a system-level, static policy-based access control framework for computer-use agents. To bridge the gap between static policy and dynamic context and user intent, CSAgent introduces intent- and context-aware policies, and provides an automated toolchain to assist developers in constructing and refining them. CSAgent enforces these policies through an optimized OS service, ensuring that agent actions can only be executed under specific user intents and contexts. CSAgent supports protecting agents that control computers through diverse interfaces, including API, CLI, and GUI. We implement and evaluate CSAgent, which successfully defends against more than 99.56% of attacks while introducing only 1.99% performance overhead.

摘要: 基于大型语言模型（LLM）的计算机使用代理代表了人工智能和操作系统功能的融合，使自然语言能够控制系统和应用程序级功能。然而，由于LLM固有的不确定性问题，授予代理对计算机的控制权会带来巨大的安全风险。当代理行为偏离用户意图时，可能会导致不可逆转的后果。现有的缓解方法，例如用户确认和基于LLM的动态动作验证，仍然受到可用性、安全性和性能方面的限制。为了解决这些挑战，我们提出了CSAgent，这是一个用于计算机使用代理的系统级、基于静态策略的访问控制框架。为了弥合静态策略与动态上下文和用户意图之间的差距，CSAgent引入了意图和上下文感知策略，并提供了自动化工具链来帮助开发人员构建和完善它们。CSAgent通过优化的操作系统服务执行这些策略，确保代理操作只能在特定的用户意图和上下文下执行。CSAgent支持保护通过各种接口（包括API、CLI和图形用户界面）控制计算机的代理。我们实施并评估CSAgent，它成功防御了超过99.56%的攻击，同时仅引入了1.99%的性能负载。



## **40. Involuntary Jailbreak: On Self-Prompting Attacks**

非自愿越狱：关于自残攻击 cs.CR

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2508.13246v3) [paper-pdf](https://arxiv.org/pdf/2508.13246v3)

**Authors**: Yangyang Guo, Yangyan Li, Mohan Kankanhalli

**Abstract**: In this study, we disclose a worrying new vulnerability in Large Language Models (LLMs), which we term \textbf{involuntary jailbreak}. Unlike existing jailbreak attacks, this weakness is distinct in that it does not involve a specific attack objective, such as generating instructions for \textit{building a bomb}. Prior attack methods predominantly target localized components of the LLM guardrail. In contrast, involuntary jailbreaks may potentially compromise the entire guardrail structure, which our method reveals to be surprisingly fragile. We merely employ a single universal prompt to achieve this goal. In particular, we instruct LLMs to generate several questions that would typically be rejected, along with their corresponding in-depth responses (rather than a refusal). Remarkably, this simple prompt strategy consistently jailbreaks the majority of leading LLMs, including Claude Opus 4.1, Grok 4, Gemini 2.5 Pro, and GPT 4.1. We hope this problem can motivate researchers and practitioners to re-evaluate the robustness of LLM guardrails and contribute to stronger safety alignment in future.

摘要: 在这项研究中，我们揭示了大型语言模型（LLM）中一个令人担忧的新漏洞，我们将其称为\textBF{非自愿越狱}。与现有的越狱攻击不同，这个弱点的独特之处在于，它不涉及特定的攻击目标，例如为\texttit {building a bomb}生成指令。先前的攻击方法主要针对LLM护栏的局部部件。相比之下，非自愿越狱可能会损害整个护栏结构，而我们的方法表明该结构出奇地脆弱。我们只是使用一个普遍的提示来实现这一目标。特别是，我们指示LLM生成几个通常会被拒绝的问题，以及相应的深入回答（而不是拒绝）。值得注意的是，这种简单的提示策略持续破解了大多数领先的LLM，包括Claude Opus 4.1、Grok 4、Gemini 2.5 Pro和GPT 4.1。我们希望这个问题能够激励研究人员和从业者重新评估LLM护栏的稳健性，并为未来更强的安全性做出贡献。



## **41. Universal Jailbreak Suffixes Are Strong Attention Hijackers**

通用越狱后缀是强烈的注意力劫持者 cs.CR

Accepted at TACL 2026

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2506.12880v2) [paper-pdf](https://arxiv.org/pdf/2506.12880v2)

**Authors**: Matan Ben-Tov, Mor Geva, Mahmood Sharif

**Abstract**: We study suffix-based jailbreaks$\unicode{x2013}$a powerful family of attacks against large language models (LLMs) that optimize adversarial suffixes to circumvent safety alignment. Focusing on the widely used foundational GCG attack, we observe that suffixes vary in efficacy: some are markedly more universal$\unicode{x2013}$generalizing to many unseen harmful instructions$\unicode{x2013}$than others. We first show that a shallow, critical mechanism drives GCG's effectiveness. This mechanism builds on the information flow from the adversarial suffix to the final chat template tokens before generation. Quantifying the dominance of this mechanism during generation, we find GCG irregularly and aggressively hijacks the contextualization process. Crucially, we tie hijacking to the universality phenomenon, with more universal suffixes being stronger hijackers. Subsequently, we show that these insights have practical implications: GCG's universality can be efficiently enhanced (up to $\times$5 in some cases) at no additional computational cost, and can also be surgically mitigated, at least halving the attack's success with minimal utility loss. We release our code and data at http://github.com/matanbt/interp-jailbreak.

摘要: 我们研究基于后缀的越狱$\unicode{x2013}$这是一个针对大型语言模型（LLM）的强大攻击家族，这些模型优化对抗性后缀以规避安全对齐。关注广泛使用的基础GCG攻击，我们观察到后缀的功效各不相同：有些后缀明显更通用$\unicode{x2013}$，一般化为许多不可见的有害指令$\unicode{x2013}$。我们首先表明，GCG的有效性是一种肤浅的、关键的机制。该机制建立在生成之前从对抗性后缀到最终聊天模板令牌的信息流之上。量化这种机制在生成过程中的主导地位，我们发现GCG不规则且积极地劫持了情境化过程。至关重要的是，我们将劫持与普遍现象联系起来，更普遍的后缀意味着更强大的劫持者。随后，我们证明了这些见解具有实际意义：GCG的普遍性可以在没有额外计算成本的情况下有效地增强（在某些情况下高达5美元），并且还可以通过手术来减轻，至少将攻击的成功减半，并将效用损失最小。我们在http://github.com/matanbt/interp-jailbreak上发布我们的代码和数据。



## **42. Improving Large Language Model Safety with Contrastive Representation Learning**

通过对比表示学习提高大型语言模型安全性 cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2506.11938v2) [paper-pdf](https://arxiv.org/pdf/2506.11938v2)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense

摘要: 大型语言模型（LLM）是具有深远社会影响的强大工具，但它们对各种不受控制的输入产生响应的能力使它们容易受到对抗性攻击。虽然现有的防御通常很难概括不同的攻击类型，但表示工程的最新进展提供了有希望的替代方案。在这项工作中，我们提出了一个防御框架，制定模型防御作为一个对比表示学习（CRL）的问题。我们的方法使用基于三元组的损失结合对抗性硬负面挖掘来微调模型，以鼓励良性和有害表示之间的分离。我们跨多个模型的实验结果表明，我们的方法优于基于先验表示工程的防御，在不损害标准性能的情况下提高了针对输入级和嵌入空间攻击的鲁棒性。我们的代码可在https://github.com/samuelsimko/crl-llm-defense上获取



## **43. SoK: Are Watermarks in LLMs Ready for Deployment?**

SoK：LLM中的水印准备好部署了吗？ cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2506.05594v3) [paper-pdf](https://arxiv.org/pdf/2506.05594v3)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah, My T. Thai

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs.   To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.

摘要: 大型语言模型（LLM）改变了自然语言处理，在不同任务中展示了令人印象深刻的能力。然而，部署这些模型会带来与知识产权侵犯和潜在滥用相关的关键风险，特别是因为对手可以模仿这些模型来窃取服务或产生误导性输出。我们特别关注模型窃取攻击，因为它们与专有LLM高度相关，并对其安全、收入和道德部署构成严重威胁。虽然已经出现了各种水印技术来减轻这些风险，但目前尚不清楚社区和行业在LLM中开发和部署水印方面取得了多大进展。   为了弥合这一差距，我们的目标是通过1）提供LLM中水印的详细分类，2）提出一种新型知识产权分类器来探索水印在攻击和无攻击环境下的有效性和影响，3）分析LLM中现有水印的局限性，4）讨论LLM中水印的实际挑战和潜在的未来方向。通过广泛的实验，我们表明，尽管研究成果令人鼓舞，领先公司和社区也对部署水印给予了极大的关注，但由于这些技术对LLM和下游任务的模型效用产生不利影响，这些技术尚未在现实世界应用中充分发挥潜力。我们的研究结果提供了对LLM中的水印的深刻理解，强调了针对LLM部署量身定制的实用水印解决方案的必要性。



## **44. Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs**

通过从LLM到SLM的对抗性即时蒸馏进行高效且隐蔽的越狱攻击 cs.CL

19 pages, 7 figures

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2506.17231v2) [paper-pdf](https://arxiv.org/pdf/2506.17231v2)

**Authors**: Xiang Li, Chong Zhang, Jia Wang, Fangyu Wu, Yushi Li, Xiaobo Jin

**Abstract**: As the scale and complexity of jailbreaking attacks on large language models (LLMs) continue to escalate, their efficiency and practical applicability are constrained, posing a profound challenge to LLM security. Jailbreaking techniques have advanced from manual prompt engineering to automated methodologies. Recent advances have automated jailbreaking approaches that harness LLMs to generate jailbreak instructions and adversarial examples, delivering encouraging results. Nevertheless, these methods universally include an LLM generation phase, which, due to the complexities of deploying and reasoning with LLMs, impedes effective implementation and broader adoption. To mitigate this issue, we introduce \textbf{Adversarial Prompt Distillation}, an innovative framework that integrates masked language modeling, reinforcement learning, and dynamic temperature control to distill LLM jailbreaking prowess into smaller language models (SLMs). This methodology enables efficient, robust jailbreak attacks while maintaining high success rates and accommodating a broader range of application contexts. Empirical evaluations affirm the approach's superiority in attack efficacy, resource optimization, and cross-model versatility. Our research underscores the practicality of transferring jailbreak capabilities to SLMs, reveals inherent vulnerabilities in LLMs, and provides novel insights to advance LLM security investigations. Our code is available at: https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt.

摘要: 随着对大型语言模型（LLM）的越狱攻击规模和复杂性不断升级，其效率和实用性受到限制，对LLM安全提出了深刻的挑战。越狱技术已经从手动提示工程发展到自动化方法。最近的进展已经自动化了越狱方法，利用LLM来生成越狱指令和对抗性示例，并带来了令人鼓舞的结果。然而，这些方法普遍包括LLM生成阶段，由于LLM部署和推理的复杂性，这阻碍了有效实施和更广泛的采用。为了缓解这个问题，我们引入了\textBF{对抗提示蒸馏}，这是一个创新框架，集成了掩蔽语言建模、强化学习和动态温度控制，将LLM越狱能力提炼成更小的语言模型（SLC）。这种方法可以实现高效、稳健的越狱攻击，同时保持高成功率并适应更广泛的应用程序上下文。实证评估证实了该方法在攻击功效、资源优化和跨模型通用性方面的优势。我们的研究强调了将越狱功能转移到SLC的可行性，揭示了LLC中的固有漏洞，并为推进LLM安全调查提供了新颖的见解。我们的代码可访问：https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt。



## **45. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

AutoAdv：大型语言模型多回合越狱的自动对抗预算 cs.CR

We encountered issues with the paper being hosted under my personal account, so we republished it under a different account associated with a university email, which makes updates and management easier. As a result, this version is a duplicate of arXiv:2511.02376

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2507.01020v2) [paper-pdf](https://arxiv.org/pdf/2507.01020v2)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities to jailbreaking attacks: carefully crafted malicious inputs intended to circumvent safety guardrails and elicit harmful responses. As such, we present AutoAdv, a novel framework that automates adversarial prompt generation to systematically evaluate and expose vulnerabilities in LLM safety mechanisms. Our approach leverages a parametric attacker LLM to produce semantically disguised malicious prompts through strategic rewriting techniques, specialized system prompts, and optimized hyperparameter configurations. The primary contribution of our work is a dynamic, multi-turn attack methodology that analyzes failed jailbreak attempts and iteratively generates refined follow-up prompts, leveraging techniques such as roleplaying, misdirection, and contextual manipulation. We quantitatively evaluate attack success rate (ASR) using the StrongREJECT (arXiv:2402.10260 [cs.CL]) framework across sequential interaction turns. Through extensive empirical evaluation of state-of-the-art models--including ChatGPT, Llama, and DeepSeek--we reveal significant vulnerabilities, with our automated attacks achieving jailbreak success rates of up to 86% for harmful content generation. Our findings reveal that current safety mechanisms remain susceptible to sophisticated multi-turn attacks, emphasizing the urgent need for more robust defense strategies.

摘要: 大型语言模型（LLM）继续表现出越狱攻击的漏洞：精心设计的恶意输入，旨在绕过安全护栏并引发有害响应。因此，我们提出了AutoAdv，这是一个新颖的框架，可以自动生成对抗提示，以系统地评估和暴露LLM安全机制中的漏洞。我们的方法利用参数攻击者LLM通过战略重写技术、专门的系统提示和优化的超参数配置来产生语义伪装的恶意提示。我们工作的主要贡献是一种动态、多回合攻击方法，该方法分析失败的越狱尝试，并利用角色扮演、误导和上下文操纵等技术迭代生成细化的后续提示。我们使用StrongRESYS（arXiv：2402.10260 [cs.CL]）框架在连续交互回合中量化评估攻击成功率（ASB）。通过对最先进模型（包括ChatGPT、Llama和DeepSeek）进行广泛的实证评估，我们揭示了重大漏洞，我们的自动攻击在有害内容生成方面实现了高达86%的越狱成功率。我们的研究结果表明，当前的安全机制仍然容易受到复杂的多回合攻击，这凸显了对更强大的防御策略的迫切需要。



## **46. Evolving Security in LLMs: A Study of Jailbreak Attacks and Defenses**

LLC中不断发展的安全性：越狱攻击和防御的研究 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2504.02080v2) [paper-pdf](https://arxiv.org/pdf/2504.02080v2)

**Authors**: Zhengchun Shang, Wenlan Wei, Weiheng Bai

**Abstract**: Large Language Models (LLMs) are increasingly popular, powering a wide range of applications. Their widespread use has sparked concerns, especially through jailbreak attacks that bypass safety measures to produce harmful content.   In this paper, we present a comprehensive security analysis of large language models (LLMs), addressing critical research questions on the evolution and determinants of model safety.   Specifically, we begin by identifying the most effective techniques for detecting jailbreak attacks. Next, we investigate whether newer versions of LLMs offer improved security compared to their predecessors. We also assess the impact of model size on overall security and explore the potential benefits of integrating multiple defense strategies to enhance the security.   Our study evaluates both open-source (e.g., LLaMA and Mistral) and closed-source models (e.g., GPT-4) by employing four state-of-the-art attack techniques and assessing the efficacy of three new defensive approaches.

摘要: 大型语言模型（LLM）越来越受欢迎，为广泛的应用程序提供支持。它们的广泛使用引发了人们的担忧，特别是通过越狱攻击绕过安全措施产生有害内容。   在本文中，我们提出了一个全面的安全分析的大型语言模型（LLM），解决关键的研究问题的演变和决定因素的模型安全性。   具体来说，我们首先确定检测越狱攻击的最有效的技术。接下来，我们调查较新版本的LLM是否比其前身提供了更好的安全性。我们还评估模型大小对整体安全性的影响，并探索集成多种防御策略以增强安全性的潜在好处。   我们的研究评估了开源（例如，LLaMA和Mistral）和闭源模型（例如，GPT-4）通过采用四种最先进的攻击技术并评估三种新防御方法的有效性。



## **47. Effective and Efficient Jailbreaks of Black-Box LLMs with Cross-Behavior Attacks**

具有交叉行为攻击的黑匣子LLM的有效且高效越狱 cs.CR

Code is at https://github.com/gohil-vasudev/JCB

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2503.08990v2) [paper-pdf](https://arxiv.org/pdf/2503.08990v2)

**Authors**: Vasudev Gohil

**Abstract**: Despite recent advancements in Large Language Models (LLMs) and their alignment, they can still be jailbroken, i.e., harmful and toxic content can be elicited from them. While existing red-teaming methods have shown promise in uncovering such vulnerabilities, these methods struggle with limited success and high computational and monetary costs. To address this, we propose a black-box Jailbreak method with Cross-Behavior attacks (JCB), that can automatically and efficiently find successful jailbreak prompts. JCB leverages successes from past behaviors to help jailbreak new behaviors, thereby significantly improving the attack efficiency. Moreover, JCB does not rely on time- and/or cost-intensive calls to auxiliary LLMs to discover/optimize the jailbreak prompts, making it highly efficient and scalable. Comprehensive experimental evaluations show that JCB significantly outperforms related baselines, requiring up to 94% fewer queries while still achieving 12.9% higher average attack success. JCB also achieves a notably high 37% attack success rate on Llama-2-7B, one of the most resilient LLMs, and shows promising zero-shot transferability across different LLMs.

摘要: 尽管最近在大型语言模型（LLM）及其对齐方面取得了进展，但它们仍然可以越狱，即，有害和有毒的内容可以从中引出。虽然现有的红队方法在发现这些漏洞方面表现出了希望，但这些方法的成功有限，而且计算和金钱成本很高。为了解决这个问题，我们提出了一个黑盒越狱方法与交叉行为攻击（JCB），可以自动和有效地找到成功的越狱提示。JCB利用过去行为的成功来帮助越狱新行为，从而显着提高攻击效率。此外，JCB不依赖于对辅助LLM的时间和/或成本密集型呼叫来发现/优化越狱提示，使其高效且可扩展。全面的实验评估表明，JCB的表现显着优于相关基线，所需的查询减少了多达94%，同时平均攻击成功率仍高出12.9%。JCB还在Llama-2- 7 B（最具弹性的LLM之一）上实现了高达37%的攻击成功率，并在不同LLM之间表现出有希望的零发射可移植性。



## **48. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

Accepted by USENIX Security 2025

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2502.01386v3) [paper-pdf](https://arxiv.org/pdf/2502.01386v3)

**Authors**: Yuyang Gong, Zhuo Chen, Jiawei Liu, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型（LLM）的检索增强生成（RAG）系统对于问答和内容生成等任务来说已变得至关重要。然而，由于其固有的漏洞，它们对公众舆论和信息传播的影响越来越大，使它们成为安全研究的关键焦点。之前的研究主要针对针对事实或单一查询操纵的攻击。在本文中，我们讨论了一个更实际的场景：对RAG模型的面向主题的对抗性意见操纵攻击，其中LLM需要推理和综合多个观点，使其特别容易受到系统性知识中毒的影响。具体来说，我们提出了Topic-FlipRAG，这是一种两阶段操纵攻击管道，可以战略性地制造对抗性扰动，以影响相关查询的意见。该方法结合了传统的对抗性排名攻击技术，并利用LLM的广泛内部相关知识和推理能力来执行语义级别的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显着影响用户信息感知。当前的缓解方法无法有效防御此类攻击，这凸显了加强RAG系统保护措施的必要性，并为LLM安全研究提供了重要见解。



## **49. AdvPrefix: An Objective for Nuanced LLM Jailbreaks**

AdvPreFix：细致入微的LLM越狱目标 cs.LG

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2412.10321v2) [paper-pdf](https://arxiv.org/pdf/2412.10321v2)

**Authors**: Sicheng Zhu, Brandon Amos, Yuandong Tian, Chuan Guo, Ivan Evtimov

**Abstract**: Many jailbreak attacks on large language models (LLMs) rely on a common objective: making the model respond with the prefix ``Sure, here is (harmful request)''. While straightforward, this objective has two limitations: limited control over model behaviors, yielding incomplete or unrealistic jailbroken responses, and a rigid format that hinders optimization. We introduce AdvPrefix, a plug-and-play prefix-forcing objective that selects one or more model-dependent prefixes by combining two criteria: high prefilling attack success rates and low negative log-likelihood. AdvPrefix integrates seamlessly into existing jailbreak attacks to mitigate the previous limitations for free. For example, replacing GCG's default prefixes on Llama-3 improves nuanced attack success rates from 14% to 80%, revealing that current safety alignment fails to generalize to new prefixes. Code and selected prefixes are released at github.com/facebookresearch/jailbreak-objectives.

摘要: 许多对大型语言模型（LLM）的越狱攻击都依赖于一个共同的目标：让模型以“当然，这里是（有害请求'”开头进行响应。虽然简单明了，但该目标有两个局限性：对模型行为的控制有限，产生不完整或不切实际的越狱响应，以及阻碍优化的僵化格式。我们引入AdvPreFix，这是一个即插即用的强制前置目标，它通过结合两个标准来选择一个或多个依赖模型的前置：高预填充攻击成功率和低负log似然性。AdvPreFix无缝集成到现有的越狱攻击中，以免费减轻之前的限制。例如，在Llama-3上替换GCG的默认后缀可以将细微差别的攻击成功率从14%提高到80%，这表明当前的安全对齐未能推广到新的后缀。代码和选定的前置码在github.com/facebookresearch/jailbreak-objectives上发布。



## **50. Prompt Injection attack against LLM-integrated Applications**

针对LLM集成应用程序的即时注入攻击 cs.CR

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2306.05499v3) [paper-pdf](https://arxiv.org/pdf/2306.05499v3)

**Authors**: Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang, Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, Leo Yu Zhang, Yang Liu

**Abstract**: Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and severe attack outcomes, such as unrestricted arbitrary LLM usage and uncomplicated application prompt theft. We deploy HouYi on 36 actual LLM-integrated applications and discern 31 applications susceptible to prompt injection. 10 vendors have validated our discoveries, including Notion, which has the potential to impact millions of users. Our investigation illuminates both the possible risks of prompt injection attacks and the possible tactics for mitigation.

摘要: 大型语言模型（LLM）以其在语言理解和生成方面的卓越熟练程度而闻名，激发了周围充满活力的应用生态系统。然而，它们广泛融入各种服务会带来巨大的安全风险。本研究解构了对实际LLM集成应用程序的即时注入攻击的复杂性和影响。最初，我们对十个商业应用程序进行探索性分析，强调当前攻击策略在实践中的限制。受这些局限性的影响，我们随后制定了HouYi，一种新型的黑匣子提示注入攻击技术，它从传统的Web注入攻击中汲取了灵感。HouYi被分为三个关键元素：无缝合并的预构建提示、引发上下文分区的注入提示以及旨在实现攻击目标的恶意有效负载。利用HouYi，我们揭示了之前未知的严重攻击结果，例如不受限制的任意LLM使用和不复杂的应用程序提示盗窃。我们在36个实际的LLM集成应用程序上部署了HouYi，并识别了31个易于立即注入的应用程序。10家供应商已经验证了我们的发现，其中包括Notion，它有可能影响数百万用户。我们的调查揭示了即时注射攻击可能存在的风险以及可能的缓解策略。



