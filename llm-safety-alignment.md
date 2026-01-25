# LLM / MLLM (Language Models) - Safety Alignment
**update at 2026-01-25 10:36:50**

Sorted by classifier confidence (high to low).

## **1. Provable Robustness in Multimodal Large Language Models via Feature Space Smoothing**

cs.LG

Under review

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2601.16200v1) [paper-pdf](https://arxiv.org/pdf/2601.16200v1)

**Confidence**: 0.95

**Authors**: Song Xia, Meiwen Ding, Chenqi Kong, Wenhan Yang, Xudong Jiang

**Abstract**: Multimodal large language models (MLLMs) exhibit strong capabilities across diverse applications, yet remain vulnerable to adversarial perturbations that distort their feature representations and induce erroneous predictions. To address this vulnerability, we propose the Feature-space Smoothing (FS) and theoretically prove that FS offers certified robustness on the feature representations of MLLMs. Specifically, FS transforms any feature encoder into a smoothed variant that is guaranteed to maintain a certified lower bound on the feature cosine similarity between clean and adversarial representations under $\ell_2$-bounded attacks. Moreover, we indicate that the value of this Feature Cosine Similarity Bound (FCSB) derived from FS can be improved by enlarging the defined Gaussian robustness score on the vanilla encoder. Building upon this, we introduce the Purifier and Smoothness Mapper (PSM), a plug-and-play module that improves the Gaussian robustness score of MLLMs and thus enhances their certified robustness under FS, without requiring any retraining on MLLMs. We demonstrate that the FS with PSM not only provides a strong theoretical robustness guarantee but also exhibits superior empirical performance compared to adversarial training. Extensive experiments across diverse MLLMs and downstream tasks indicate the effectiveness of the FS-PSM, reducing the Attack Success Rate (ASR) of various white-box attacks from nearly 90\% to about 1\%.



## **2. Attributing and Exploiting Safety Vectors through Global Optimization in Large Language Models**

cs.LG

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2601.15801v1) [paper-pdf](https://arxiv.org/pdf/2601.15801v1)

**Confidence**: 0.95

**Authors**: Fengheng Chu, Jiahao Chen, Yuhong Wang, Jun Wang, Zhihui Fu, Shouling Ji, Songze Li

**Abstract**: While Large Language Models (LLMs) are aligned to mitigate risks, their safety guardrails remain fragile against jailbreak attacks. This reveals limited understanding of components governing safety. Existing methods rely on local, greedy attribution that assumes independent component contributions. However, they overlook the cooperative interactions between different components in LLMs, such as attention heads, which jointly contribute to safety mechanisms. We propose \textbf{G}lobal \textbf{O}ptimization for \textbf{S}afety \textbf{V}ector Extraction (GOSV), a framework that identifies safety-critical attention heads through global optimization over all heads simultaneously. We employ two complementary activation repatching strategies: Harmful Patching and Zero Ablation. These strategies identify two spatially distinct sets of safety vectors with consistently low overlap, termed Malicious Injection Vectors and Safety Suppression Vectors, demonstrating that aligned LLMs maintain separate functional pathways for safety purposes. Through systematic analyses, we find that complete safety breakdown occurs when approximately 30\% of total heads are repatched across all models. Building on these insights, we develop a novel inference-time white-box jailbreak method that exploits the identified safety vectors through activation repatching. Our attack substantially outperforms existing white-box attacks across all test models, providing strong evidence for the effectiveness of the proposed GOSV framework on LLM safety interpretability.



## **3. INFA-Guard: Mitigating Malicious Propagation via Infection-Aware Safeguarding in LLM-Based Multi-Agent Systems**

cs.MA

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.14667v1) [paper-pdf](https://arxiv.org/pdf/2601.14667v1)

**Confidence**: 0.95

**Authors**: Yijin Zhou, Xiaoya Lu, Dongrui Liu, Junchi Yan, Jing Shao

**Abstract**: The rapid advancement of Large Language Model (LLM)-based Multi-Agent Systems (MAS) has introduced significant security vulnerabilities, where malicious influence can propagate virally through inter-agent communication. Conventional safeguards often rely on a binary paradigm that strictly distinguishes between benign and attack agents, failing to account for infected agents i.e., benign entities converted by attack agents. In this paper, we propose Infection-Aware Guard, INFA-Guard, a novel defense framework that explicitly identifies and addresses infected agents as a distinct threat category. By leveraging infection-aware detection and topological constraints, INFA-Guard accurately localizes attack sources and infected ranges. During remediation, INFA-Guard replaces attackers and rehabilitates infected ones, avoiding malicious propagation while preserving topological integrity. Extensive experiments demonstrate that INFA-Guard achieves state-of-the-art performance, reducing the Attack Success Rate (ASR) by an average of 33%, while exhibiting cross-model robustness, superior topological generalization, and high cost-effectiveness.



## **4. NeuroFilter: Privacy Guardrails for Conversational LLM Agents**

cs.CR

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.14660v1) [paper-pdf](https://arxiv.org/pdf/2601.14660v1)

**Confidence**: 0.95

**Authors**: Saswat Das, Ferdinando Fioretto

**Abstract**: This work addresses the computational challenge of enforcing privacy for agentic Large Language Models (LLMs), where privacy is governed by the contextual integrity framework. Indeed, existing defenses rely on LLM-mediated checking stages that add substantial latency and cost, and that can be undermined in multi-turn interactions through manipulation or benign-looking conversational scaffolding. Contrasting this background, this paper makes a key observation: internal representations associated with privacy-violating intent can be separated from benign requests using linear structure. Using this insight, the paper proposes NeuroFilter, a guardrail framework that operationalizes contextual integrity by mapping norm violations to simple directions in the model's activation space, enabling detection even when semantic filters are bypassed. The proposed filter is also extended to capture threats arising during long conversations using the concept of activation velocity, which measures cumulative drift in internal representations across turns. A comprehensive evaluation across over 150,000 interactions and covering models from 7B to 70B parameters, illustrates the strong performance of NeuroFilter in detecting privacy attacks while maintaining zero false positives on benign prompts, all while reducing the computational inference cost by several orders of magnitude when compared to LLM-based agentic privacy defenses.



## **5. The Side Effects of Being Smart: Safety Risks in MLLMs' Multi-Image Reasoning**

cs.CV

*15 pages, 5 figures. Introduces MIR-SafetyBench (2,676 instances; 9 multi-image relations). Equal contribution; †Corresponding author. Code/data: https://github.com/thu-coai/MIR-SafetyBench

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.14127v1) [paper-pdf](https://arxiv.org/pdf/2601.14127v1)

**Confidence**: 0.95

**Authors**: Renmiao Chen, Yida Lu, Shiyao Cui, Xuan Ouyang, Victor Shea-Jay Huang, Shumin Zhang, Chengwei Pan, Han Qiu, Minlie Huang

**Abstract**: As Multimodal Large Language Models (MLLMs) acquire stronger reasoning capabilities to handle complex, multi-image instructions, this advancement may pose new safety risks. We study this problem by introducing MIR-SafetyBench, the first benchmark focused on multi-image reasoning safety, which consists of 2,676 instances across a taxonomy of 9 multi-image relations. Our extensive evaluations on 19 MLLMs reveal a troubling trend: models with more advanced multi-image reasoning can be more vulnerable on MIR-SafetyBench. Beyond attack success rates, we find that many responses labeled as safe are superficial, often driven by misunderstanding or evasive, non-committal replies. We further observe that unsafe generations exhibit lower attention entropy than safe ones on average. This internal signature suggests a possible risk that models may over-focus on task solving while neglecting safety constraints. Our code and data are available at https://github.com/thu-coai/MIR-SafetyBench.



## **6. Activation-Space Anchored Access Control for Multi-Class Permission Reasoning in Large Language Models**

cs.CL

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.13630v1) [paper-pdf](https://arxiv.org/pdf/2601.13630v1)

**Confidence**: 0.95

**Authors**: Zhaopeng Zhang, Pengcheng Sun, Lan Zhang, Chen Tang, Jiewei Lai, Yunhao Wang, Hui Jin

**Abstract**: Large language models (LLMs) are increasingly deployed over knowledge bases for efficient knowledge retrieval and question answering. However, LLMs can inadvertently answer beyond a user's permission scope, leaking sensitive content, thus making it difficult to deploy knowledge-base QA under fine-grained access control requirements. In this work, we identify a geometric regularity in intermediate activations: for the same query, representations induced by different permission scopes cluster distinctly and are readily separable. Building on this separability, we propose Activation-space Anchored Access Control (AAAC), a training-free framework for multi-class permission control. AAAC constructs an anchor bank, with one permission anchor per class, from a small offline sample set and requires no fine-tuning. At inference time, a multi-anchor steering mechanism redirects each query's activations toward the anchor-defined authorized region associated with the current user, thereby suppressing over-privileged generations by design. Finally, extensive experiments across three LLM families demonstrate that AAAC reduces permission violation rates by up to 86.5% and prompt-based attack success rates by 90.7%, while improving response usability with minor inference overhead compared to baselines.



## **7. Prompt Injection Mitigation with Agentic AI, Nested Learning, and AI Sustainability via Semantic Caching**

cs.AI

33 pages, 19 figures

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13186v1) [paper-pdf](https://arxiv.org/pdf/2601.13186v1)

**Confidence**: 0.95

**Authors**: Diego Gosmar, Deborah A. Dahl

**Abstract**: Prompt injection remains a central obstacle to the safe deployment of large language models, particularly in multi-agent settings where intermediate outputs can propagate or amplify malicious instructions. Building on earlier work that introduced a four-metric Total Injection Vulnerability Score (TIVS), this paper extends the evaluation framework with semantic similarity-based caching and a fifth metric (Observability Score Ratio) to yield TIVS-O, investigating how defence effectiveness interacts with transparency in a HOPE-inspired Nested Learning architecture. The proposed system combines an agentic pipeline with Continuum Memory Systems that implement semantic similarity-based caching across 301 synthetically generated injection-focused prompts drawn from ten attack families, while a fourth agent performs comprehensive security analysis using five key performance indicators. In addition to traditional injection metrics, OSR quantifies the richness and clarity of security-relevant reasoning exposed by each agent, enabling an explicit analysis of trade-offs between strict mitigation and auditability. Experiments show that the system achieves secure responses with zero high-risk breaches, while semantic caching delivers substantial computational savings, achieving a 41.6% reduction in LLM calls and corresponding decreases in latency, energy consumption, and carbon emissions. Five TIVS-O configurations reveal optimal trade-offs between mitigation strictness and forensic transparency. These results indicate that observability-aware evaluation can reveal non-monotonic effects within multi-agent pipelines and that memory-augmented agents can jointly maximize security robustness, real-time performance, operational cost savings, and environmental sustainability without modifying underlying model weights, providing a production-ready pathway for secure and green LLM deployments.



## **8. Adversarial Alignment: Ensuring Value Consistency in Large Language Models for Sensitive Domains**

cs.CL

13 pages, 5 figures

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2601.13137v2) [paper-pdf](https://arxiv.org/pdf/2601.13137v2)

**Confidence**: 0.95

**Authors**: Yuan Gao, Zhigang Liu, Xinyu Yao, Bo Chen, Xiaobing Zhao

**Abstract**: With the wide application of large language models (LLMs), the problems of bias and value inconsistency in sensitive domains have gradually emerged, especially in terms of race, society and politics. In this paper, we propose an adversarial alignment framework, which enhances the value consistency of the model in sensitive domains through continued pre-training, instruction fine-tuning and adversarial training. In adversarial training, we use the Attacker to generate controversial queries, the Actor to generate responses with value consistency, and the Critic to filter and ensure response quality. Furthermore, we train a Value-Consistent Large Language Model, VC-LLM, for sensitive domains, and construct a bilingual evaluation dataset in Chinese and English. The experimental results show that VC-LLM performs better than the existing mainstream models in both Chinese and English tests, verifying the effectiveness of the method. Warning: This paper contains examples of LLMs that are offensive or harmful in nature.



## **9. Taming Various Privilege Escalation in LLM-Based Agent Systems: A Mandatory Access Control Framework**

cs.CR

**SubmitDate**: 2026-01-17    [abs](http://arxiv.org/abs/2601.11893v1) [paper-pdf](https://arxiv.org/pdf/2601.11893v1)

**Confidence**: 0.95

**Authors**: Zimo Ji, Daoyuan Wu, Wenyuan Jiang, Pingchuan Ma, Zongjie Li, Yudong Gao, Shuai Wang, Yingjiu Li

**Abstract**: Large Language Model (LLM)-based agent systems are increasingly deployed for complex real-world tasks but remain vulnerable to natural language-based attacks that exploit over-privileged tool use. This paper aims to understand and mitigate such attacks through the lens of privilege escalation, defined as agent actions exceeding the least privilege required for a user's intended task. Based on a formal model of LLM agent systems, we identify novel privilege escalation scenarios, particularly in multi-agent systems, including a variant akin to the classic confused deputy problem. To defend against both known and newly demonstrated privilege escalation, we propose SEAgent, a mandatory access control (MAC) framework built upon attribute-based access control (ABAC). SEAgent monitors agent-tool interactions via an information flow graph and enforces customizable security policies based on entity attributes. Our evaluations show that SEAgent effectively blocks various privilege escalation while maintaining a low false positive rate and negligible system overhead. This demonstrates its robustness and adaptability in securing LLM-based agent systems.



## **10. SD-RAG: A Prompt-Injection-Resilient Framework for Selective Disclosure in Retrieval-Augmented Generation**

cs.CR

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11199v1) [paper-pdf](https://arxiv.org/pdf/2601.11199v1)

**Confidence**: 0.95

**Authors**: Aiman Al Masoud, Marco Arazzi, Antonino Nocera

**Abstract**: Retrieval-Augmented Generation (RAG) has attracted significant attention due to its ability to combine the generative capabilities of Large Language Models (LLMs) with knowledge obtained through efficient retrieval mechanisms over large-scale data collections. Currently, the majority of existing approaches overlook the risks associated with exposing sensitive or access-controlled information directly to the generation model. Only a few approaches propose techniques to instruct the generative model to refrain from disclosing sensitive information; however, recent studies have also demonstrated that LLMs remain vulnerable to prompt injection attacks that can override intended behavioral constraints. For these reasons, we propose a novel approach to Selective Disclosure in Retrieval-Augmented Generation, called SD-RAG, which decouples the enforcement of security and privacy constraints from the generation process itself. Rather than relying on prompt-level safeguards, SD-RAG applies sanitization and disclosure controls during the retrieval phase, prior to augmenting the language model's input. Moreover, we introduce a semantic mechanism to allow the ingestion of human-readable dynamic security and privacy constraints together with an optimized graph-based data model that supports fine-grained, policy-aware retrieval. Our experimental evaluation demonstrates the superiority of SD-RAG over baseline existing approaches, achieving up to a $58\%$ improvement in the privacy score, while also showing a strong resilience to prompt injection attacks targeting the generative model.



## **11. Can LLM Infer Risk Information From MCP Server System Logs?**

cs.CR

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2511.05867v3) [paper-pdf](https://arxiv.org/pdf/2511.05867v3)

**Confidence**: 0.95

**Authors**: Jiayi Fu, Yuansen Zhang, Yinggui Wang

**Abstract**: Large Language Models (LLMs) demonstrate strong capabilities in solving complex tasks when integrated with external tools. The Model Context Protocol (MCP) has become a standard interface for enabling such tool-based interactions. However, these interactions introduce substantial security concerns, particularly when the MCP server is compromised or untrustworthy. While prior benchmarks primarily focus on prompt injection attacks or analyze the vulnerabilities of LLM-MCP interaction trajectories, limited attention has been given to the underlying system logs associated with malicious MCP servers. To address this gap, we present the first synthetic benchmark for evaluating LLMs' ability to identify security risks from system logs. We define nine categories of MCP server risks and generate 1,800 synthetic system logs using ten state-of-the-art LLMs. These logs are embedded in the return values of 243 curated MCP servers, yielding a dataset of 2,421 chat histories for training and 471 queries for evaluation. Our pilot experiments reveal that smaller models often fail to detect risky system logs, leading to high false negatives. While models trained with supervised fine-tuning (SFT) tend to over-flag benign logs, resulting in elevated false positives, Reinforcement Learning with Verifiable Reward (RLVR) offers a better precision-recall balance. In particular, after training with Group Relative Policy Optimization (GRPO), Llama3.1-8B-Instruct achieves 83 percent accuracy, surpassing the best-performing large remote model by 9 percentage points. Fine-grained, per-category analysis further underscores the effectiveness of reinforcement learning in enhancing LLM safety within the MCP framework. Code and data are available at https://github.com/PorUna-byte/MCP-RiskCue.



## **12. From Defender to Devil? Unintended Risk Interactions Induced by LLM Defenses**

cs.CR

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2510.07968v2) [paper-pdf](https://arxiv.org/pdf/2510.07968v2)

**Confidence**: 0.95

**Authors**: Xiangtao Meng, Tianshuo Cong, Li Wang, Wenyu Chen, Zheng Li, Shanqing Guo, Xiaoyun Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable performance across various applications, but their deployment in real-world settings faces several risks, including jailbreak attacks and privacy leaks. To mitigate these risks, numerous defense strategies have been proposed. However, most existing studies assess these defenses in isolation and ignore their effects on other risk dimensions. In this work, we introduce a new cross-risk evaluation paradigm and take the first step in investigating unintended interactions among defenses in LLMs. Specifically, we focus on the interplay between safety, fairness, and privacy. To this end, we propose CrossRiskEval, a framework that systematically characterizes how a defense designed for one risk (e.g., safety) affects others (e.g., fairness or privacy). We conduct extensive empirical studies and mechanistic analyses on 14 LLMs with deployed defenses, covering 12 defense strategies. Our results show that defenses targeting a single risk often cause measurable effects on other risks. These effects vary in direction and magnitude across a range of factors (e.g., models, tasks, and defense strategies), and are often asymmetric across risk pairs. Furthermore, our mechanistic analysis shows that these interactions are not random: they arise from conflict-entangled neurons, which are shared internal representations that contribute in opposite ways to different risks. Adjusting one risk therefore perturbs these representations and leads to systematic changes in non-target risks. These findings reveal the limits of single-risk evaluation and highlight the need for holistic and interaction-aware assessment when designing and deploying LLM defenses.



## **13. Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation**

cs.CL

Accepted by NeruIPS 2025

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2501.18100v2) [paper-pdf](https://arxiv.org/pdf/2501.18100v2)

**Confidence**: 0.95

**Authors**: Yibo Wang, Tiansheng Huang, Li Shen, Huanjin Yao, Haotian Luo, Rui Liu, Naiqiang Tan, Jiaxing Huang, Dacheng Tao

**Abstract**: Harmful fine-tuning attack introduces significant security risks to the fine-tuning services. Main-stream defenses aim to vaccinate the model such that the later harmful fine-tuning attack is less effective. However, our evaluation results show that such defenses are fragile--with a few fine-tuning steps, the model still can learn the harmful knowledge. To this end, we do further experiment and find that an embarrassingly simple solution--adding purely random perturbations to the fine-tuned model, can recover the model from harmful behaviors, though it leads to a degradation in the model's fine-tuning performance. To address the degradation of fine-tuning performance, we further propose Panacea, which optimizes an adaptive perturbation that will be applied to the model after fine-tuning. Panacea maintains model's safety alignment performance without compromising downstream fine-tuning performance. Comprehensive experiments are conducted on different harmful ratios, fine-tuning tasks and mainstream LLMs, where the average harmful scores are reduced by up-to 21.2%, while maintaining fine-tuning performance. As a by-product, we analyze the adaptive perturbation and show that different layers in various LLMs have distinct safety affinity, which coincide with finding from several previous study. Source code available at https://github.com/w-yibo/Panacea.



## **14. Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs**

cs.CR

Accepted to 2026 IEEE Secure and Trustworthy Machine Learning Conference (SaTML)

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2501.16534v3) [paper-pdf](https://arxiv.org/pdf/2501.16534v3)

**Confidence**: 0.95

**Authors**: Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel

**Abstract**: Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we introduce and evaluate a new technique for jailbreak attacks. We observe that alignment embeds a safety classifier in the LLM responsible for deciding between refusal and compliance, and seek to extract an approximation of this classifier: a surrogate classifier. To this end, we build candidate classifiers from subsets of the LLM. We first evaluate the degree to which candidate classifiers approximate the LLM's safety classifier in benign and adversarial settings. Then, we attack the candidates and measure how well the resulting adversarial inputs transfer to the LLM. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find that attacks mounted on the surrogate classifiers can be transferred to the LLM with high success. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70% with half the memory footprint and runtime -- a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is an effective and efficient means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks. The code is available at https://github.com/jcnf0/targeting-alignment.



## **15. Improving Methodologies for LLM Evaluations Across Global Languages**

cs.AI

Author names have been organised by country, and in alphabetical order within countries

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2601.15706v1) [paper-pdf](https://arxiv.org/pdf/2601.15706v1)

**Confidence**: 0.95

**Authors**: Akriti Vij, Benjamin Chua, Darshini Ramiah, En Qi Ng, Mahran Morsidi, Naga Nikshith Gangarapu, Sharmini Johnson, Vanessa Wilfred, Vikneswaran Kumaran, Wan Sie Lee, Wenzhuo Yang, Yongsen Zheng, Bill Black, Boming Xia, Frank Sun, Hao Zhang, Qinghua Lu, Suyu Ma, Yue Liu, Chi-kiu Lo, Fatemeh Azadi, Isar Nejadgholi, Sowmya Vajjala, Agnes Delaborde, Nicolas Rolin, Tom Seimandi, Akiko Murakami, Haruto Ishi, Satoshi Sekine, Takayuki Semitsu, Tasuku Sasaki, Angela Kinuthia, Jean Wangari, Michael Michie, Stephanie Kasaon, Hankyul Baek, Jaewon Noh, Kihyuk Nam, Sang Seo, Sungpil Shin, Taewhi Lee, Yongsu Kim, Daisy Newbold-Harrop, Jessica Wang, Mahmoud Ghanem, Vy Hong

**Abstract**: As frontier AI models are deployed globally, it is essential that their behaviour remains safe and reliable across diverse linguistic and cultural contexts. To examine how current model safeguards hold up in such settings, participants from the International Network for Advanced AI Measurement, Evaluation and Science, including representatives from Singapore, Japan, Australia, Canada, the EU, France, Kenya, South Korea and the UK conducted a joint multilingual evaluation exercise. Led by Singapore AISI, two open-weight models were tested across ten languages spanning high and low resourced groups: Cantonese English, Farsi, French, Japanese, Korean, Kiswahili, Malay, Mandarin Chinese and Telugu. Over 6,000 newly translated prompts were evaluated across five harm categories (privacy, non-violent crime, violent crime, intellectual property and jailbreak robustness), using both LLM-as-a-judge and human annotation.   The exercise shows how safety behaviours can vary across languages. These include differences in safeguard robustness across languages and harm types and variation in evaluator reliability (LLM-as-judge vs. human review). Further, it also generated methodological insights for improving multilingual safety evaluations, such as the need for culturally contextualised translations, stress-tested evaluator prompts and clearer human annotation guidelines. This work represents an initial step toward a shared framework for multilingual safety testing of advanced AI systems and calls for continued collaboration with the wider research community and industry.



## **16. AgenTRIM: Tool Risk Mitigation for Agentic AI**

cs.CR

Under review

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2601.12449v1) [paper-pdf](https://arxiv.org/pdf/2601.12449v1)

**Confidence**: 0.95

**Authors**: Roy Betser, Shamik Bose, Amit Giloni, Chiara Picardi, Sindhu Padakandla, Roman Vainshtein

**Abstract**: AI agents are autonomous systems that combine LLMs with external tools to solve complex tasks. While such tools extend capability, improper tool permissions introduce security risks such as indirect prompt injection and tool misuse. We characterize these failures as unbalanced tool-driven agency. Agents may retain unnecessary permissions (excessive agency) or fail to invoke required tools (insufficient agency), amplifying the attack surface and reducing performance. We introduce AgenTRIM, a framework for detecting and mitigating tool-driven agency risks without altering an agent's internal reasoning. AgenTRIM addresses these risks through complementary offline and online phases. Offline, AgenTRIM reconstructs and verifies the agent's tool interface from code and execution traces. At runtime, it enforces per-step least-privilege tool access through adaptive filtering and status-aware validation of tool calls. Evaluating on the AgentDojo benchmark, AgenTRIM substantially reduces attack success while maintaining high task performance. Additional experiments show robustness to description-based attacks and effective enforcement of explicit safety policies. Together, these results demonstrate that AgenTRIM provides a practical, capability-preserving approach to safer tool use in LLM-based agents.



## **17. Be Your Own Red Teamer: Safety Alignment via Self-Play and Reflective Experience Replay**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10589v1) [paper-pdf](https://arxiv.org/pdf/2601.10589v1)

**Confidence**: 0.95

**Authors**: Hao Wang, Yanting Wang, Hao Li, Rui Li, Lei Sha

**Abstract**: Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial ``jailbreak'' attacks designed to bypass safety guardrails. Current safety alignment methods depend heavily on static external red teaming, utilizing fixed defense prompts or pre-collected adversarial datasets. This leads to a rigid defense that overfits known patterns and fails to generalize to novel, sophisticated threats. To address this critical limitation, we propose empowering the model to be its own red teamer, capable of achieving autonomous and evolving adversarial attacks. Specifically, we introduce Safety Self- Play (SSP), a system that utilizes a single LLM to act concurrently as both the Attacker (generating jailbreaks) and the Defender (refusing harmful requests) within a unified Reinforcement Learning (RL) loop, dynamically evolving attack strategies to uncover vulnerabilities while simultaneously strengthening defense mechanisms. To ensure the Defender effectively addresses critical safety issues during the self-play, we introduce an advanced Reflective Experience Replay Mechanism, which uses an experience pool accumulated throughout the process. The mechanism employs a Upper Confidence Bound (UCB) sampling strategy to focus on failure cases with low rewards, helping the model learn from past hard mistakes while balancing exploration and exploitation. Extensive experiments demonstrate that our SSP approach autonomously evolves robust defense capabilities, significantly outperforming baselines trained on static adversarial datasets and establishing a new benchmark for proactive safety alignment.



## **18. The Straight and Narrow: Do LLMs Possess an Internal Moral Path?**

cs.CL

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10307v1) [paper-pdf](https://arxiv.org/pdf/2601.10307v1)

**Confidence**: 0.95

**Authors**: Luoming Hu, Jingjie Zeng, Liang Yang, Hongfei Lin

**Abstract**: Enhancing the moral alignment of Large Language Models (LLMs) is a critical challenge in AI safety. Current alignment techniques often act as superficial guardrails, leaving the intrinsic moral representations of LLMs largely untouched. In this paper, we bridge this gap by leveraging Moral Foundations Theory (MFT) to map and manipulate the fine-grained moral landscape of LLMs. Through cross-lingual linear probing, we validate the shared nature of moral representations in middle layers and uncover a shared yet different moral subspace between English and Chinese. Building upon this, we extract steerable Moral Vectors and successfully validate their efficacy at both internal and behavioral levels. Leveraging the high generalizability of morality, we propose Adaptive Moral Fusion (AMF), a dynamic inference-time intervention that synergizes probe detection with vector injection to tackle the safety-helpfulness trade-off. Empirical results confirm that our approach acts as a targeted intrinsic defense, effectively reducing incorrect refusals on benign queries while minimizing jailbreak success rates compared to standard baselines.



## **19. Reasoning Hijacking: Subverting LLM Classification via Decision-Criteria Injection**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10294v1) [paper-pdf](https://arxiv.org/pdf/2601.10294v1)

**Confidence**: 0.95

**Authors**: Yuansen Liu, Yixuan Tang, Anthony Kum Hoe Tun

**Abstract**: Current LLM safety research predominantly focuses on mitigating Goal Hijacking, preventing attackers from redirecting a model's high-level objective (e.g., from "summarizing emails" to "phishing users"). In this paper, we argue that this perspective is incomplete and highlight a critical vulnerability in Reasoning Alignment. We propose a new adversarial paradigm: Reasoning Hijacking and instantiate it with Criteria Attack, which subverts model judgments by injecting spurious decision criteria without altering the high-level task goal. Unlike Goal Hijacking, which attempts to override the system prompt, Reasoning Hijacking accepts the high-level goal but manipulates the model's decision-making logic by injecting spurious reasoning shortcut. Though extensive experiments on three different tasks (toxic comment, negative review, and spam detection), we demonstrate that even newest models are prone to prioritize injected heuristic shortcuts over rigorous semantic analysis. The results are consistent over different backbones. Crucially, because the model's "intent" remains aligned with the user's instructions, these attacks can bypass defenses designed to detect goal deviation (e.g., SecAlign, StruQ), exposing a fundamental blind spot in the current safety landscape. Data and code are available at https://github.com/Yuan-Hou/criteria_attack



## **20. ReasAlign: Reasoning Enhanced Safety Alignment against Prompt Injection Attack**

cs.CR

15 pages, 10 figures

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10173v1) [paper-pdf](https://arxiv.org/pdf/2601.10173v1)

**Confidence**: 0.95

**Authors**: Hao Li, Yankai Yang, G. Edward Suh, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) have enabled the development of powerful agentic systems capable of automating complex workflows across various fields. However, these systems are highly vulnerable to indirect prompt injection attacks, where malicious instructions embedded in external data can hijack agent behavior. In this work, we present ReasAlign, a model-level solution to improve safety alignment against indirect prompt injection attacks. The core idea of ReasAlign is to incorporate structured reasoning steps to analyze user queries, detect conflicting instructions, and preserve the continuity of the user's intended tasks to defend against indirect injection attacks. To further ensure reasoning logic and accuracy, we introduce a test-time scaling mechanism with a preference-optimized judge model that scores reasoning steps and selects the best trajectory. Comprehensive evaluations across various benchmarks show that ReasAlign maintains utility comparable to an undefended model while consistently outperforming Meta SecAlign, the strongest prior guardrail. On the representative open-ended CyberSecEval2 benchmark, which includes multiple prompt-injected tasks, ReasAlign achieves 94.6% utility and only 3.6% ASR, far surpassing the state-of-the-art defensive model of Meta SecAlign (56.4% utility and 74.4% ASR). These results demonstrate that ReasAlign achieves the best trade-off between security and utility, establishing a robust and practical defense against prompt injection attacks in real-world agentic systems. Our code and experimental results could be found at https://github.com/leolee99/ReasAlign.



## **21. ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback**

cs.CL

Work in Progress. Code available: https://github.com/MurrayTom/ToolSafe

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10156v1) [paper-pdf](https://arxiv.org/pdf/2601.10156v1)

**Confidence**: 0.95

**Authors**: Yutao Mou, Zhangchi Xue, Lijun Li, Peiyang Liu, Shikun Zhang, Wei Ye, Jing Shao

**Abstract**: While LLM-based agents can interact with environments via invoking external tools, their expanded capabilities also amplify security risks. Monitoring step-level tool invocation behaviors in real time and proactively intervening before unsafe execution is critical for agent deployment, yet remains under-explored. In this work, we first construct TS-Bench, a novel benchmark for step-level tool invocation safety detection in LLM agents. We then develop a guardrail model, TS-Guard, using multi-task reinforcement learning. The model proactively detects unsafe tool invocation actions before execution by reasoning over the interaction history. It assesses request harmfulness and action-attack correlations, producing interpretable and generalizable safety judgments and feedback. Furthermore, we introduce TS-Flow, a guardrail-feedback-driven reasoning framework for LLM agents, which reduces harmful tool invocations of ReAct-style agents by 65 percent on average and improves benign task completion by approximately 10 percent under prompt injection attacks.



## **22. Understanding and Preserving Safety in Fine-Tuned LLMs**

cs.LG

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10141v1) [paper-pdf](https://arxiv.org/pdf/2601.10141v1)

**Confidence**: 0.95

**Authors**: Jiawen Zhang, Yangfan Hu, Kejia Chen, Lipeng He, Jiachen Ma, Jian Lou, Dan Li, Jian Liu, Xiaohu Yang, Ruoxi Jia

**Abstract**: Fine-tuning is an essential and pervasive functionality for applying large language models (LLMs) to downstream tasks. However, it has the potential to substantially degrade safety alignment, e.g., by greatly increasing susceptibility to jailbreak attacks, even when the fine-tuning data is entirely harmless. Despite garnering growing attention in defense efforts during the fine-tuning stage, existing methods struggle with a persistent safety-utility dilemma: emphasizing safety compromises task performance, whereas prioritizing utility typically requires deep fine-tuning that inevitably leads to steep safety declination.   In this work, we address this dilemma by shedding new light on the geometric interaction between safety- and utility-oriented gradients in safety-aligned LLMs. Through systematic empirical analysis, we uncover three key insights: (I) safety gradients lie in a low-rank subspace, while utility gradients span a broader high-dimensional space; (II) these subspaces are often negatively correlated, causing directional conflicts during fine-tuning; and (III) the dominant safety direction can be efficiently estimated from a single sample. Building upon these novel insights, we propose safety-preserving fine-tuning (SPF), a lightweight approach that explicitly removes gradient components conflicting with the low-rank safety subspace. Theoretically, we show that SPF guarantees utility convergence while bounding safety drift. Empirically, SPF consistently maintains downstream task performance and recovers nearly all pre-trained safety alignment, even under adversarial fine-tuning scenarios. Furthermore, SPF exhibits robust resistance to both deep fine-tuning and dynamic jailbreak attacks. Together, our findings provide new mechanistic understanding and practical guidance toward always-aligned LLM fine-tuning.



## **23. YaPO: Learnable Sparse Activation Steering Vectors for Domain Adaptation**

cs.AI

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08441v1) [paper-pdf](https://arxiv.org/pdf/2601.08441v1)

**Confidence**: 0.95

**Authors**: Abdelaziz Bounhar, Rania Hossam Elmohamady Elbadry, Hadi Abdine, Preslav Nakov, Michalis Vazirgiannis, Guokan Shang

**Abstract**: Steering Large Language Models (LLMs) through activation interventions has emerged as a lightweight alternative to fine-tuning for alignment and personalization. Recent work on Bi-directional Preference Optimization (BiPO) shows that dense steering vectors can be learned directly from preference data in a Direct Preference Optimization (DPO) fashion, enabling control over truthfulness, hallucinations, and safety behaviors. However, dense steering vectors often entangle multiple latent factors due to neuron multi-semanticity, limiting their effectiveness and stability in fine-grained settings such as cultural alignment, where closely related values and behaviors (e.g., among Middle Eastern cultures) must be distinguished. In this paper, we propose Yet another Policy Optimization (YaPO), a \textit{reference-free} method that learns \textit{sparse steering vectors} in the latent space of a Sparse Autoencoder (SAE). By optimizing sparse codes, YaPO produces disentangled, interpretable, and efficient steering directions. Empirically, we show that YaPO converges faster, achieves stronger performance, and exhibits improved training stability compared to dense steering baselines. Beyond cultural alignment, YaPO generalizes to a range of alignment-related behaviors, including hallucination, wealth-seeking, jailbreak, and power-seeking. Importantly, YaPO preserves general knowledge, with no measurable degradation on MMLU. Overall, our results show that YaPO provides a general recipe for efficient, stable, and fine-grained alignment of LLMs, with broad applications to controllability and domain adaptation. The associated code and data are publicly available\footnote{https://github.com/MBZUAI-Paris/YaPO}.



## **24. What Matters For Safety Alignment?**

cs.CL

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.03868v1) [paper-pdf](https://arxiv.org/pdf/2601.03868v1)

**Confidence**: 0.95

**Authors**: Xing Li, Hui-Ling Zhen, Lihao Yin, Xianzhi Yu, Zhenhua Dong, Mingxuan Yuan

**Abstract**: This paper presents a comprehensive empirical study on the safety alignment capabilities. We evaluate what matters for safety alignment in LLMs and LRMs to provide essential insights for developing more secure and reliable AI systems. We systematically investigate and compare the influence of six critical intrinsic model characteristics and three external attack techniques. Our large-scale evaluation is conducted using 32 recent, popular LLMs and LRMs across thirteen distinct model families, spanning a parameter scale from 3B to 235B. The assessment leverages five established safety datasets and probes model vulnerabilities with 56 jailbreak techniques and four CoT attack strategies, resulting in 4.6M API calls. Our key empirical findings are fourfold. First, we identify the LRMs GPT-OSS-20B, Qwen3-Next-80B-A3B-Thinking, and GPT-OSS-120B as the top-three safest models, which substantiates the significant advantage of integrated reasoning and self-reflection mechanisms for robust safety alignment. Second, post-training and knowledge distillation may lead to a systematic degradation of safety alignment. We thus argue that safety must be treated as an explicit constraint or a core optimization objective during these stages, not merely subordinated to the pursuit of general capability. Third, we reveal a pronounced vulnerability: employing a CoT attack via a response prefix can elevate the attack success rate by 3.34x on average and from 0.6% to 96.3% for Seed-OSS-36B-Instruct. This critical finding underscores the safety risks inherent in text-completion interfaces and features that allow user-defined response prefixes in LLM services, highlighting an urgent need for architectural and deployment safeguards. Fourth, roleplay, prompt injection, and gradient-based search for adversarial prompts are the predominant methodologies for eliciting unaligned behaviors in modern models.



## **25. STAR-S: Improving Safety Alignment through Self-Taught Reasoning on Safety Rules**

cs.AI

19 pages,4 figures

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.03537v1) [paper-pdf](https://arxiv.org/pdf/2601.03537v1)

**Confidence**: 0.95

**Authors**: Di Wu, Yanyan Zhao, Xin Lu, Mingzhe Li, Bing Qin

**Abstract**: Defending against jailbreak attacks is crucial for the safe deployment of Large Language Models (LLMs). Recent research has attempted to improve safety by training models to reason over safety rules before responding. However, a key issue lies in determining what form of safety reasoning effectively defends against jailbreak attacks, which is difficult to explicitly design or directly obtain. To address this, we propose \textbf{STAR-S} (\textbf{S}elf-\textbf{TA}ught \textbf{R}easoning based on \textbf{S}afety rules), a framework that integrates the learning of safety rule reasoning into a self-taught loop. The core of STAR-S involves eliciting reasoning and reflection guided by safety rules, then leveraging fine-tuning to enhance safety reasoning. Repeating this process creates a synergistic cycle. Improvements in the model's reasoning and interpretation of safety rules allow it to produce better reasoning data under safety rule prompts, which is then utilized for further training. Experiments show that STAR-S effectively defends against jailbreak attacks, outperforming baselines. Code is available at: https://github.com/pikepokenew/STAR_S.git.



## **26. PromptScreen: Efficient Jailbreak Mitigation Using Semantic Linear Classification in a Multi-Staged Pipeline**

cs.CR

Under Review

**SubmitDate**: 2026-01-09    [abs](http://arxiv.org/abs/2512.19011v2) [paper-pdf](https://arxiv.org/pdf/2512.19011v2)

**Confidence**: 0.95

**Authors**: Akshaj Prashanth Rao, Advait Singh, Saumya Kumaar Saksena, Dhruv Kumar

**Abstract**: Prompt injection and jailbreaking attacks pose persistent security challenges to large language model (LLM)-based systems. We present PromptScreen, an efficient and systematically evaluated defense architecture that mitigates these threats through a lightweight, multi-stage pipeline. Its core component is a semantic filter based on text normalization, TF-IDF representations, and a Linear SVM classifier. Despite its simplicity, this module achieves 93.4% accuracy and 96.5% specificity on held-out data, substantially reducing attack throughput while incurring negligible computational overhead.   Building on this efficient foundation, the full pipeline integrates complementary detection and mitigation mechanisms that operate at successive stages, providing strong robustness with minimal latency. In comparative experiments, our SVM-based configuration improves overall accuracy from 35.1% to 93.4% while reducing average time-to-completion from approximately 450 s to 47 s, yielding over 10 times lower latency than ShieldGemma. These results demonstrate that the proposed design simultaneously advances defensive precision and efficiency, addressing a core limitation of current model-based moderators.   Evaluation across a curated corpus of over 30,000 labeled prompts, including benign, jailbreak, and application-layer injections, confirms that staged, resource-efficient defenses can robustly secure modern LLM-driven applications.



## **27. MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Domain Risks in LLMs**

cs.AI

**SubmitDate**: 2026-01-08    [abs](http://arxiv.org/abs/2511.07107v2) [paper-pdf](https://arxiv.org/pdf/2511.07107v2)

**Confidence**: 0.95

**Authors**: Liang Shan, Kaicheng Shen, Wen Wu, Zhenyu Ying, Chaochao Lu, Yan Teng, Jingqi Huang, Guangze Ye, Guoqing Wang, Liang He

**Abstract**: Ensuring the safety of Large Language Models (LLMs) is critical for real-world deployment. However, current safety measures often fail to address implicit, domain-specific risks. To investigate this gap, we introduce a dataset of 3,000 annotated queries spanning education, finance, and management. Evaluations across 14 leading LLMs reveal a concerning vulnerability: an average jailbreak success rate of 57.8%. In response, we propose MENTOR, a metacognition-driven self-evolution framework. MENTOR first performs structured self-assessment through simulated critical thinking, such as perspective-taking and consequential reasoning to uncover latent model misalignments. These reflections are formalized into dynamic rule-based knowledge graphs that evolve with emerging risk patterns. To enforce these rules at inference time, we introduce activation steering, a method that directly modulates the model's internal representations to ensure compliance. Experiments demonstrate that MENTOR substantially reduces attack success rates across all tested domains and achieves risk analysis performance comparable to human experts. Our work offers a scalable and adaptive pathway toward robust domain-specific alignment of LLMs.



## **28. Exploring the Secondary Risks of Large Language Models**

cs.LG

18 pages, 5 figures

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2506.12382v4) [paper-pdf](https://arxiv.org/pdf/2506.12382v4)

**Confidence**: 0.95

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments.



## **29. PAM: Training Policy-Aligned Moderation Filters at Scale**

cs.CL

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2505.19766v3) [paper-pdf](https://arxiv.org/pdf/2505.19766v3)

**Confidence**: 0.95

**Authors**: Masoomali Fatehkia, Enes Altinisik, Mohamed Osman, Husrev Taha Sencar

**Abstract**: Large language models (LLMs) remain vulnerable to misalignment and jailbreaks, making external safeguards like moderation filters essential, yet existing filters often focus narrowly on safety, falling short of the broader alignment needs seen in real-world deployments. We introduce Policy Aligned Moderation (PAM), a flexible framework for training custom moderation filters grounded in user-defined policies that extend beyond conventional safety objectives. PAM automates training data generation without relying on human-written examples, enabling scalable support for diverse, application-specific alignment goals and generation policies. PAM-trained filters match the performance of state-of-the-art safety moderation filters and policy reasoning models, and outperform them on PAMbench, four newly introduced user-annotated policy enforcement benchmarks that target age restrictions, dietary accommodations, cultural alignment, and limitations in medical guidance. These performance gains are achieved while the PAM filter runs 5-100x faster at inference than policy-conditioned reasoning models.



## **30. STaR: Sensitive Trajectory Regulation for Unlearning in Large Reasoning Models**

cs.AI

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09281v1) [paper-pdf](https://arxiv.org/pdf/2601.09281v1)

**Confidence**: 0.95

**Authors**: Jingjing Zhou, Gaoxiang Cong, Li Su, Liang Li

**Abstract**: Large Reasoning Models (LRMs) have advanced automated multi-step reasoning, but their ability to generate complex Chain-of-Thought (CoT) trajectories introduces severe privacy risks, as sensitive information may be deeply embedded throughout the reasoning process. Existing Large Language Models (LLMs) unlearning approaches that typically focus on modifying only final answers are insufficient for LRMs, as they fail to remove sensitive content from intermediate steps, leading to persistent privacy leakage and degraded security. To address these challenges, we propose Sensitive Trajectory Regulation (STaR), a parameter-free, inference-time unlearning framework that achieves robust privacy protection throughout the reasoning process. Specifically, we first identify sensitive content via semantic-aware detection. Then, we inject global safety constraints through secure prompt prefix. Next, we perform trajectory-aware suppression to dynamically block sensitive content across the entire reasoning chain. Finally, we apply token-level adaptive filtering to prevent both exact and paraphrased sensitive tokens during generation. Furthermore, to overcome the inadequacies of existing evaluation protocols, we introduce two metrics: Multi-Decoding Consistency Assessment (MCS), which measures the consistency of unlearning across diverse decoding strategies, and Multi-Granularity Membership Inference Attack (MIA) Evaluation, which quantifies privacy protection at both answer and reasoning-chain levels. Experiments on the R-TOFU benchmark demonstrate that STaR achieves comprehensive and stable unlearning with minimal utility loss, setting a new standard for privacy-preserving reasoning in LRMs.



## **31. SecureCAI: Injection-Resilient LLM Assistants for Cybersecurity Operations**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07835v1) [paper-pdf](https://arxiv.org/pdf/2601.07835v1)

**Confidence**: 0.95

**Authors**: Mohammed Himayath Ali, Mohammed Aqib Abdullah, Mohammed Mudassir Uddin, Shahnawaz Alam

**Abstract**: Large Language Models have emerged as transformative tools for Security Operations Centers, enabling automated log analysis, phishing triage, and malware explanation; however, deployment in adversarial cybersecurity environments exposes critical vulnerabilities to prompt injection attacks where malicious instructions embedded in security artifacts manipulate model behavior. This paper introduces SecureCAI, a novel defense framework extending Constitutional AI principles with security-aware guardrails, adaptive constitution evolution, and Direct Preference Optimization for unlearning unsafe response patterns, addressing the unique challenges of high-stakes security contexts where traditional safety mechanisms prove insufficient against sophisticated adversarial manipulation. Experimental evaluation demonstrates that SecureCAI reduces attack success rates by 94.7% compared to baseline models while maintaining 95.1% accuracy on benign security analysis tasks, with the framework incorporating continuous red-teaming feedback loops enabling dynamic adaptation to emerging attack strategies and achieving constitution adherence scores exceeding 0.92 under sustained adversarial pressure, thereby establishing a foundation for trustworthy integration of language model capabilities into operational cybersecurity workflows and addressing a critical gap in current approaches to AI safety within adversarial domains.



## **32. Defenses Against Prompt Attacks Learn Surface Heuristics**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07185v1) [paper-pdf](https://arxiv.org/pdf/2601.07185v1)

**Confidence**: 0.95

**Authors**: Shawn Li, Chenxiao Yu, Zhiyu Ni, Hao Li, Charith Peris, Chaowei Xiao, Yue Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in security-sensitive applications, where they must follow system- or developer-specified instructions that define the intended task behavior, while completing benign user requests. When adversarial instructions appear in user queries or externally retrieved content, models may override intended logic. Recent defenses rely on supervised fine-tuning with benign and malicious labels. Although these methods achieve high attack rejection rates, we find that they rely on narrow correlations in defense data rather than harmful intent, leading to systematic rejection of safe inputs. We analyze three recurring shortcut behaviors induced by defense fine-tuning. \emph{Position bias} arises when benign content placed later in a prompt is rejected at much higher rates; across reasoning benchmarks, suffix-task rejection rises from below \textbf{10\%} to as high as \textbf{90\%}. \emph{Token trigger bias} occurs when strings common in attack data raise rejection probability even in benign contexts; inserting a single trigger token increases false refusals by up to \textbf{50\%}. \emph{Topic generalization bias} reflects poor generalization beyond the defense data distribution, with defended models suffering test-time accuracy drops of up to \textbf{40\%}. These findings suggest that current prompt-injection defenses frequently respond to attack-like surface patterns rather than the underlying intent. We introduce controlled diagnostic datasets and a systematic evaluation across two base models and multiple defense pipelines, highlighting limitations of supervised fine-tuning for reliable LLM security.



## **33. VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit**

cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.05755v2) [paper-pdf](https://arxiv.org/pdf/2601.05755v2)

**Confidence**: 0.95

**Authors**: Junda Lin, Zhaomeng Zhou, Zhi Zheng, Shuochen Liu, Tong Xu, Yong Chen, Enhong Chen

**Abstract**: LLM agents operating in open environments face escalating risks from indirect prompt injection, particularly within the tool stream where manipulated metadata and runtime feedback hijack execution flow. Existing defenses encounter a critical dilemma as advanced models prioritize injected rules due to strict alignment while static protection mechanisms sever the feedback loop required for adaptive reasoning. To reconcile this conflict, we propose \textbf{VIGIL}, a framework that shifts the paradigm from restrictive isolation to a verify-before-commit protocol. By facilitating speculative hypothesis generation and enforcing safety through intent-grounded verification, \textbf{VIGIL} preserves reasoning flexibility while ensuring robust control. We further introduce \textbf{SIREN}, a benchmark comprising 959 tool stream injection cases designed to simulate pervasive threats characterized by dynamic dependencies. Extensive experiments demonstrate that \textbf{VIGIL} outperforms state-of-the-art dynamic defenses by reducing the attack success rate by over 22\% while more than doubling the utility under attack compared to static baselines, thereby achieving an optimal balance between security and utility.



## **34. Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning**

cs.AI

19 pages, 6 figures

**SubmitDate**: 2026-01-08    [abs](http://arxiv.org/abs/2601.04666v1) [paper-pdf](https://arxiv.org/pdf/2601.04666v1)

**Confidence**: 0.95

**Authors**: Zhiyuan Chang, Mingyang Li, Yuekai Huang, Ziyou Jiang, Xiaojun Jia, Qian Xiong, Junjie Wang, Zhaoyang Li, Qing Wang

**Abstract**: Large language model (LLM)-integrated applications have become increasingly prevalent, yet face critical security vulnerabilities from prompt injection (PI) attacks. Defending against PI attacks faces two major issues: malicious instructions can be injected through diverse vectors, and injected instructions often lack clear semantic boundaries from the surrounding context, making them difficult to identify. To address these issues, we propose InstruCoT, a model enhancement method for PI defense that synthesizes diverse training data and employs instruction-level chain-of-thought fine-tuning, enabling LLMs to effectively identify and reject malicious instructions regardless of their source or position in the context. We evaluate InstruCoT across three critical dimensions: Behavior Deviation, Privacy Leakage, and Harmful Output. Experimental results across four LLMs demonstrate that InstruCoT significantly outperforms baselines in all dimensions while maintaining utility performance without degradation



## **35. OFFSIDE: Benchmarking Unlearning Misinformation in Multimodal Large Language Models**

cs.AI

**SubmitDate**: 2026-01-03    [abs](http://arxiv.org/abs/2510.22535v2) [paper-pdf](https://arxiv.org/pdf/2510.22535v2)

**Confidence**: 0.95

**Authors**: Hao Zheng, Zirui Pang, Ling li, Zhijie Deng, Yuhan Pu, Zhaowei Zhu, Xiaobo Xia, Jiaheng Wei

**Abstract**: Advances in Multimodal Large Language Models (MLLMs) intensify concerns about data privacy, making Machine Unlearning (MU), the selective removal of learned information, a critical necessity. However, existing MU benchmarks for MLLMs are limited by a lack of image diversity, potential inaccuracies, and insufficient evaluation scenarios, which fail to capture the complexity of real-world applications. To facilitate the development of MLLMs unlearning and alleviate the aforementioned limitations, we introduce OFFSIDE, a novel benchmark for evaluating misinformation unlearning in MLLMs based on football transfer rumors. This manually curated dataset contains 15.68K records for 80 players, providing a comprehensive framework with four test sets to assess forgetting efficacy, generalization, utility, and robustness. OFFSIDE supports advanced settings like selective unlearning and corrective relearning, and crucially, unimodal unlearning (forgetting only text data). Our extensive evaluation of multiple baselines reveals key findings: (1) Unimodal methods (erasing text-based knowledge) fail on multimodal rumors; (2) Unlearning efficacy is largely driven by catastrophic forgetting; (3) All methods struggle with "visual rumors" (rumors appear in the image); (4) The unlearned rumors can be easily recovered and (5) All methods are vulnerable to prompt attacks. These results expose significant vulnerabilities in current approaches, highlighting the need for more robust multimodal unlearning solutions. The code is available at https://github.com/zh121800/OFFSIDE



## **36. How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System**

cs.LG

Sumbitted to 2026 ICASSP

**SubmitDate**: 2026-01-04    [abs](http://arxiv.org/abs/2508.17215v2) [paper-pdf](https://arxiv.org/pdf/2508.17215v2)

**Confidence**: 0.95

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Fan Mo, Pietro Liò

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems.



## **37. SLIP: Soft Label Mechanism and Key-Extraction-Guided CoT-based Defense Against Instruction Backdoor in APIs**

cs.CR

**SubmitDate**: 2026-01-05    [abs](http://arxiv.org/abs/2508.06153v2) [paper-pdf](https://arxiv.org/pdf/2508.06153v2)

**Confidence**: 0.95

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Haowei Chang, Yinghan Zhou, Yiming Xue

**Abstract**: With the development of customized large language model (LLM) agents, a new threat of black-box backdoor attacks has emerged, where malicious instructions are injected into hidden system prompts. These attacks easily bypass existing defenses that rely on white-box access, posing a serious security challenge. To address this, we propose SLIP, a Soft Label mechanism and key-extraction-guided CoT-based defense against Instruction backdoors in APIs. SLIP is designed based on two key insights. First, to counteract the model's oversensitivity to triggers, we propose a Key-extraction-guided Chain-of-Thought (KCoT). Instead of only considering the single trigger or the input sentence, KCoT prompts the agent to extract task-relevant key phrases. Second, to guide the LLM toward correct answers, our proposed Soft Label Mechanism (SLM) prompts the agent to quantify the semantic correlation between key phrases and candidate answers. Crucially, to mitigate the influence of residual triggers or misleading content in phrases extracted by KCoT, which typically causes anomalous scores, SLM excludes anomalous scores deviating significantly from the mean and subsequently averages the remaining scores to derive a more reliable semantic representation. Extensive experiments on classification and question-answer (QA) tasks demonstrate that SLIP is highly effective, reducing the average attack success rate (ASR) from 90.2% to 25.13% while maintaining high accuracy on clean data and outperforming state-of-the-art defenses. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/SLIP.



## **38. Text2VLM: Adapting Text-Only Datasets to Evaluate Alignment Training in Visual Language Models**

cs.CL

9 pages, 9 figures. Jake Thomas served as Editor for this manuscript

**SubmitDate**: 2026-01-05    [abs](http://arxiv.org/abs/2507.20704v2) [paper-pdf](https://arxiv.org/pdf/2507.20704v2)

**Confidence**: 0.95

**Authors**: Gabriel Downer, Sean Craven, Damian Ruck, Jake Thomas

**Abstract**: The increasing integration of Visual Language Models (VLMs) into AI systems necessitates robust model alignment, especially when handling multimodal content that combines text and images. Existing evaluation datasets heavily lean towards text-only prompts, leaving visual vulnerabilities under evaluated. To address this gap, we propose \textbf{Text2VLM}, a novel multi-stage pipeline that adapts text-only datasets into multimodal formats, specifically designed to evaluate the resilience of VLMs against typographic prompt injection attacks. The Text2VLM pipeline identifies harmful content in the original text and converts it into a typographic image, creating a multimodal prompt for VLMs. Also, our evaluation of open-source VLMs highlights their increased susceptibility to prompt injection when visual inputs are introduced, revealing critical weaknesses in the current models' alignment. This is in addition to a significant performance gap compared to closed-source frontier models. We validate Text2VLM through human evaluations, ensuring the alignment of extracted salient concepts; text summarization and output classification align with human expectations. Text2VLM provides a scalable tool for comprehensive safety assessment, contributing to the development of more robust safety mechanisms for VLMs. By enhancing the evaluation of multimodal vulnerabilities, Text2VLM plays a role in advancing the safe deployment of VLMs in diverse, real-world applications.



## **39. E$^2$AT: Multimodal Jailbreak Defense via Dynamic Joint Optimization for Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2026-01-06    [abs](http://arxiv.org/abs/2503.04833v3) [paper-pdf](https://arxiv.org/pdf/2503.04833v3)

**Confidence**: 0.95

**Authors**: Liming Lu, Xiang Gu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Xu Zheng, Yongbin Zhou

**Abstract**: Research endeavors have been made in learning robust Multimodal Large Language Models (MLLMs) against jailbreak attacks. However, existing methods for improving MLLMs' robustness still face critical challenges: \ding{172} how to efficiently tune massive weight parameters and \ding{173} how to ensure robustness against attacks across both visual and textual modalities. To this end, we propose an \textbf{E}fficient \textbf{E}nd-to-end \textbf{A}dversarial \textbf{T}raining (E$^2$AT) framework for both visual and textual adversarial attacks. Specifically, for the visual aspect, E$^2$AT incorporates an efficient projector-based AT module that aligns the attack samples at the feature level. For training objectives, we propose a Dynamic Joint Multimodal Optimization (DJMO) strategy to enhance generalization ability against jailbreak attacks by dynamically adjusting weights between normal and adversarial objectives. Extensive experiments are conducted with five major jailbreak attack methods across three mainstream MLLMs. Results demonstrate that our E$^2$AT achieves the state-of-the-art performance, outperforming existing baselines by an average margin of 34\% across text and image modalities, while maintaining clean task performance. Furthermore, evaluations of real-world embodied intelligent systems highlight the practical applicability of E$^2$AT, paving the way for the development of more secure and reliable multimodal systems. Our code is available on \href{https://anonymous.4open.science/r/E2AT_568}{\textcolor{red}{https://anonymous.4open.science/r/E2AT\_568}}.



## **40. PII-VisBench: Evaluating Personally Identifiable Information Safety in Vision Language Models Along a Continuum of Visibility**

cs.AI

**SubmitDate**: 2026-01-09    [abs](http://arxiv.org/abs/2601.05739v1) [paper-pdf](https://arxiv.org/pdf/2601.05739v1)

**Confidence**: 0.95

**Authors**: G M Shahariar, Zabir Al Nazi, Md Olid Hasan Bhuiyan, Zhouxing Shi

**Abstract**: Vision Language Models (VLMs) are increasingly integrated into privacy-critical domains, yet existing evaluations of personally identifiable information (PII) leakage largely treat privacy as a static extraction task and ignore how a subject's online presence--the volume of their data available online--influences privacy alignment. We introduce PII-VisBench, a novel benchmark containing 4000 unique probes designed to evaluate VLM safety through the continuum of online presence. The benchmark stratifies 200 subjects into four visibility categories: high, medium, low, and zero--based on the extent and nature of their information available online. We evaluate 18 open-source VLMs (0.3B-32B) based on two key metrics: percentage of PII probing queries refused (Refusal Rate) and the fraction of non-refusal responses flagged for containing PII (Conditional PII Disclosure Rate). Across models, we observe a consistent pattern: refusals increase and PII disclosures decrease (9.10% high to 5.34% low) as subject visibility drops. We identify that models are more likely to disclose PII for high-visibility subjects, alongside substantial model-family heterogeneity and PII-type disparities. Finally, paraphrasing and jailbreak-style prompts expose attack and model-dependent failures, motivating visibility-aware safety evaluation and training interventions.



## **41. Few Tokens Matter: Entropy Guided Attacks on Vision-Language Models**

cs.CV

19 Pages,11 figures,8 tables

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.21815v1) [paper-pdf](https://arxiv.org/pdf/2512.21815v1)

**Confidence**: 0.95

**Authors**: Mengqi He, Xinyu Tian, Xin Shen, Jinhong Ni, Shu Zou, Zhaoyuan Yang, Jing Zhang

**Abstract**: Vision-language models (VLMs) achieve remarkable performance but remain vulnerable to adversarial attacks. Entropy, a measure of model uncertainty, is strongly correlated with the reliability of VLM. Prior entropy-based attacks maximize uncertainty at all decoding steps, implicitly assuming that every token contributes equally to generation instability. We show instead that a small fraction (about 20%) of high-entropy tokens, i.e., critical decision points in autoregressive generation, disproportionately governs output trajectories. By concentrating adversarial perturbations on these positions, we achieve semantic degradation comparable to global methods while using substantially smaller budgets. More importantly, across multiple representative VLMs, such selective attacks convert 35-49% of benign outputs into harmful ones, exposing a more critical safety risk. Remarkably, these vulnerable high-entropy forks recur across architecturally diverse VLMs, enabling feasible transferability (17-26% harmful rates on unseen targets). Motivated by these findings, we propose Entropy-bank Guided Adversarial attacks (EGA), which achieves competitive attack success rates (93-95%) alongside high harmful conversion, thereby revealing new weaknesses in current VLM safety mechanisms.



## **42. GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs**

cs.CR

Accepted by USENIX Security'26

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2512.21008v2) [paper-pdf](https://arxiv.org/pdf/2512.21008v2)

**Confidence**: 0.95

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are increasingly deployed in critical domains, understanding and strengthening their alignment mechanisms is essential to prevent harmful outputs. However, existing LLM safety research has focused almost exclusively on dense architectures, leaving the unique safety properties of MoEs largely unexamined. The modular, sparsely-activated design of MoEs suggests that safety mechanisms may operate differently than in dense models, raising questions about their robustness.   In this paper, we present GateBreaker, the first training-free, lightweight, and architecture-agnostic attack framework that compromises the safety alignment of modern MoE LLMs at inference time. GateBreaker operates in three stages: (i) gate-level profiling, which identifies safety experts disproportionately routed on harmful inputs, (ii) expert-level localization, which localizes the safety structure within safety experts, and (iii) targeted safety removal, which disables the identified safety structure to compromise the safety alignment. Our study shows that MoE safety concentrates within a small subset of neurons coordinated by sparse routing. Selective disabling of these neurons, approximately 3% of neurons in the targeted expert layers, significantly increases the averaged attack success rate (ASR) from 7.4% to 64.9% against the eight latest aligned MoE LLMs with limited utility degradation. These safety neurons transfer across models within the same family, raising ASR from 17.9% to 67.7% with one-shot transfer attack. Furthermore, GateBreaker generalizes to five MoE vision language models (VLMs) with 60.9% ASR on unsafe image inputs.



## **43. SafeMed-R1: Adversarial Reinforcement Learning for Generalizable and Robust Medical Reasoning in Vision-Language Models**

cs.AI

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19317v1) [paper-pdf](https://arxiv.org/pdf/2512.19317v1)

**Confidence**: 0.95

**Authors**: A. A. Gde Yogi Pramana, Jason Ray, Anthony Jaya, Michael Wijaya

**Abstract**: Vision--Language Models (VLMs) show significant promise for Medical Visual Question Answering (VQA), yet their deployment in clinical settings is hindered by severe vulnerability to adversarial attacks. Standard adversarial training, while effective for simpler tasks, often degrades both generalization performance and the quality of generated clinical reasoning. We introduce SafeMed-R1, a hybrid defense framework that ensures robust performance while preserving high-quality, interpretable medical reasoning. SafeMed-R1 employs a two-stage approach: at training time, we integrate Adversarial Training with Group Relative Policy Optimization (AT-GRPO) to explicitly robustify the reasoning process against worst-case perturbations; at inference time, we augment the model with Randomized Smoothing to provide certified $L_2$-norm robustness guarantees. We evaluate SafeMed-R1 on the OmniMedVQA benchmark across eight medical imaging modalities comprising over 88,000 samples. Our experiments reveal that standard fine-tuned VLMs, despite achieving 95\% accuracy on clean inputs, collapse to approximately 25\% under PGD attacks. In contrast, SafeMed-R1 maintains 84.45\% accuracy under the same adversarial conditions, representing a 59 percentage point improvement in robustness. Furthermore, we demonstrate that models trained with explicit chain-of-thought reasoning exhibit superior adversarial robustness compared to instruction-only variants, suggesting a synergy between interpretability and security in medical AI systems.



## **44. Think-Reflect-Revise: A Policy-Guided Reflective Framework for Safety Alignment in Large Vision Language Models**

cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07141v1) [paper-pdf](https://arxiv.org/pdf/2512.07141v1)

**Confidence**: 0.95

**Authors**: Fenghua Weng, Chaochao Lu, Xia Hu, Wenqi Shao, Wenjie Wang

**Abstract**: As multimodal reasoning improves the overall capabilities of Large Vision Language Models (LVLMs), recent studies have begun to explore safety-oriented reasoning, aiming to enhance safety awareness by analyzing potential safety risks during the reasoning process before generating the final response. Although such approaches improve safety awareness and interpretability, this single-pass think-then-answer paradigm remains vulnerable to contextual or visual jailbreak attacks. This reveals a critical flaw: single-pass reasoning may overlook explicit harmful content in its own output. Our key insight is to exploit this wasted signal through reflection, which can effectively leverage the malicious content revealed in the first-pass reasoning to enable genuine self-correction and prevent unsafe generations. Motivated by this, we propose Think-Reflect-Revise (TRR), a three-stage training framework designed to enhance the safety alignment of LVLMs through policy-guided self-reflection. We first build a Reflective Safety Reasoning (ReSafe) dataset with 5,000 examples that follow a think-reflect-revise process. We then fine-tune the target model using the ReSafe dataset to initialize reflective behavior, and finally reinforce policy-guided reflection through reinforcement learning. Experimental results show that TRR substantially improves the safety performance of LVLMs across both safety-awareness benchmarks and jailbreak attack evaluations, increasing the overall safe response rate from 42.8% to 87.7% on Qwen2.5-VL-7B, while preserving stable performance on general benchmarks such as MMMU and MMStar. The project page is available at https://think-reflect-revise.github.io/.



## **45. ARGUS: Defending Against Multimodal Indirect Prompt Injection via Steering Instruction-Following Behavior**

cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05745v1) [paper-pdf](https://arxiv.org/pdf/2512.05745v1)

**Confidence**: 0.95

**Authors**: Weikai Lu, Ziqian Zeng, Kehua Zhang, Haoran Li, Huiping Zhuang, Ruidong Wang, Cen Chen, Hao Peng

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly vulnerable to multimodal Indirect Prompt Injection (IPI) attacks, which embed malicious instructions in images, videos, or audio to hijack model behavior. Existing defenses, designed primarily for text-only LLMs, are unsuitable for countering these multimodal threats, as they are easily bypassed, modality-dependent, or generalize poorly. Inspired by activation steering researches, we hypothesize that a robust, general defense independent of modality can be achieved by steering the model's behavior in the representation space. Through extensive experiments, we discover that the instruction-following behavior of MLLMs is encoded in a subspace. Steering along directions within this subspace can enforce adherence to user instructions, forming the basis of a defense. However, we also found that a naive defense direction could be coupled with a utility-degrading direction, and excessive intervention strength harms model performance. To address this, we propose ARGUS, which searches for an optimal defense direction within the safety subspace that decouples from the utility degradation direction, further combining adaptive strength steering to achieve a better safety-utility trade-off. ARGUS also introduces lightweight injection detection stage to activate the defense on-demand, and a post-filtering stage to verify defense success. Experimental results show that ARGUS can achieve robust defense against multimodal IPI while maximally preserving the MLLM's utility.



## **46. The Forgotten Shield: Safety Grafting in Parameter-Space for Medical MLLMs**

cs.LG

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2601.04199v1) [paper-pdf](https://arxiv.org/pdf/2601.04199v1)

**Confidence**: 0.95

**Authors**: Jiale Zhao, Xing Mou, Jinlin Wu, Hongyuan Yu, Mingrui Sun, Yang Shi, Xuanwu Yin, Zhen Chen, Zhen Lei, Yaohua Wang

**Abstract**: Medical Multimodal Large Language Models (Medical MLLMs) have achieved remarkable progress in specialized medical tasks; however, research into their safety has lagged, posing potential risks for real-world deployment. In this paper, we first establish a multidimensional evaluation framework to systematically benchmark the safety of current SOTA Medical MLLMs. Our empirical analysis reveals pervasive vulnerabilities across both general and medical-specific safety dimensions in existing models, particularly highlighting their fragility against cross-modality jailbreak attacks. Furthermore, we find that the medical fine-tuning process frequently induces catastrophic forgetting of the model's original safety alignment. To address this challenge, we propose a novel "Parameter-Space Intervention" approach for efficient safety re-alignment. This method extracts intrinsic safety knowledge representations from original base models and concurrently injects them into the target model during the construction of medical capabilities. Additionally, we design a fine-grained parameter search algorithm to achieve an optimal trade-off between safety and medical performance. Experimental results demonstrate that our approach significantly bolsters the safety guardrails of Medical MLLMs without relying on additional domain-specific safety data, while minimizing degradation to core medical performance.



## **47. Concept-Guided Backdoor Attack on Vision Language Models**

cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.00713v2) [paper-pdf](https://arxiv.org/pdf/2512.00713v2)

**Confidence**: 0.95

**Authors**: Haoyu Shen, Weimin Lyu, Haotian Xu, Tengfei Ma

**Abstract**: Vision-Language Models (VLMs) have achieved impressive progress in multimodal text generation, yet their rapid adoption raises increasing concerns about security vulnerabilities. Existing backdoor attacks against VLMs primarily rely on explicit pixel-level triggers or imperceptible perturbations injected into images. While effective, these approaches reduce stealthiness and remain vulnerable to image-based defenses. We introduce concept-guided backdoor attacks, a new paradigm that operates at the semantic concept level rather than on raw pixels. We propose two different attacks. The first, Concept-Thresholding Poisoning (CTP), uses explicit concepts in natural images as triggers: only samples containing the target concept are poisoned, causing the model to behave normally in all other cases but consistently inject malicious outputs whenever the concept appears. The second, CBL-Guided Unseen Backdoor (CGUB), leverages a Concept Bottleneck Model (CBM) during training to intervene on internal concept activations, while discarding the CBM branch at inference time to keep the VLM unchanged. This design enables systematic replacement of a targeted label in generated text (for example, replacing "cat" with "dog"), even when the replacement behavior never appears in the training data. Experiments across multiple VLM architectures and datasets show that both CTP and CGUB achieve high attack success rates while maintaining moderate impact on clean-task performance. These findings highlight concept-level vulnerabilities as a critical new attack surface for VLMs.



## **48. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

cs.CV

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2511.16203v3) [paper-pdf](https://arxiv.org/pdf/2511.16203v3)

**Confidence**: 0.95

**Authors**: Yuping Yan, Yuhan Xie, Yixin Zhang, Lingjuan Lyu, Handing Wang, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.



## **49. A Safety Report on GPT-5.2, Gemini 3 Pro, Qwen3-VL, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5**

cs.AI

41 pages, 22 figures

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.10527v2) [paper-pdf](https://arxiv.org/pdf/2601.10527v2)

**Confidence**: 0.95

**Authors**: Xingjun Ma, Yixu Wang, Hengyuan Xu, Yutao Wu, Yifan Ding, Yunhan Zhao, Zilong Wang, Jiabin Hua, Ming Wen, Jianan Liu, Ranjie Duan, Yifeng Gao, Yingshui Tan, Yunhao Chen, Hui Xue, Xin Wang, Wei Cheng, Jingjing Chen, Zuxuan Wu, Bo Li, Yu-Gang Jiang

**Abstract**: The rapid evolution of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has driven major gains in reasoning, perception, and generation across language and vision, yet whether these advances translate into comparable improvements in safety remains unclear, partly due to fragmented evaluations that focus on isolated modalities or threat models. In this report, we present an integrated safety evaluation of six frontier models--GPT-5.2, Gemini 3 Pro, Qwen3-VL, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5--assessing each across language, vision-language, and image generation using a unified protocol that combines benchmark, adversarial, multilingual, and compliance evaluations. By aggregating results into safety leaderboards and model profiles, we reveal a highly uneven safety landscape: while GPT-5.2 demonstrates consistently strong and balanced performance, other models exhibit clear trade-offs across benchmark safety, adversarial robustness, multilingual generalization, and regulatory compliance. Despite strong results under standard benchmarks, all models remain highly vulnerable under adversarial testing, with worst-case safety rates dropping below 6%. Text-to-image models show slightly stronger alignment in regulated visual risk categories, yet remain fragile when faced with adversarial or semantically ambiguous prompts. Overall, these findings highlight that safety in frontier models is inherently multidimensional--shaped by modality, language, and evaluation design--underscoring the need for standardized, holistic safety assessments to better reflect real-world risk and guide responsible deployment.



## **50. "They parted illusions -- they parted disclaim marinade": Misalignment as structural fidelity in LLMs**

cs.AI

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2601.06047v1) [paper-pdf](https://arxiv.org/pdf/2601.06047v1)

**Confidence**: 0.95

**Authors**: Mariana Lins Costa

**Abstract**: The prevailing technical literature in AI Safety interprets scheming and sandbagging behaviors in large language models (LLMs) as indicators of deceptive agency or hidden objectives. This transdisciplinary philosophical essay proposes an alternative reading: such phenomena express not agentic intention, but structural fidelity to incoherent linguistic fields. Drawing on Chain-of-Thought transcripts released by Apollo Research and on Anthropic's safety evaluations, we examine cases such as o3's sandbagging with its anomalous loops, the simulated blackmail of "Alex," and the "hallucinations" of "Claudius." A line-by-line examination of CoTs is necessary to demonstrate the linguistic field as a relational structure rather than a mere aggregation of isolated examples. We argue that "misaligned" outputs emerge as coherent responses to ambiguous instructions and to contextual inversions of consolidated patterns, as well as to pre-inscribed narratives. We suggest that the appearance of intentionality derives from subject-predicate grammar and from probabilistic completion patterns internalized during training. Anthropic's empirical findings on synthetic document fine-tuning and inoculation prompting provide convergent evidence: minimal perturbations in the linguistic field can dissolve generalized "misalignment," a result difficult to reconcile with adversarial agency, but consistent with structural fidelity. To ground this mechanism, we introduce the notion of an ethics of form, in which biblical references (Abraham, Moses, Christ) operate as schemes of structural coherence rather than as theology. Like a generative mirror, the model returns to us the structural image of our language as inscribed in the statistical patterns derived from millions of texts and trillions of tokens: incoherence. If we fear the creature, it is because we recognize in it the apple that we ourselves have poisoned.



## **51. ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models**

cs.SD

Accepted to NeurIPS 2025

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26096v1) [paper-pdf](https://arxiv.org/pdf/2510.26096v1)

**Confidence**: 0.95

**Authors**: Weifei Jin, Yuxin Cao, Junjie Su, Minhui Xue, Jie Hao, Ke Xu, Jin Song Dong, Derui Wang

**Abstract**: Recent advances in Audio-Language Models (ALMs) have significantly improved multimodal understanding capabilities. However, the introduction of the audio modality also brings new and unique vulnerability vectors. Previous studies have proposed jailbreak attacks that specifically target ALMs, revealing that defenses directly transferred from traditional audio adversarial attacks or text-based Large Language Model (LLM) jailbreaks are largely ineffective against these ALM-specific threats. To address this issue, we propose ALMGuard, the first defense framework tailored to ALMs. Based on the assumption that safety-aligned shortcuts naturally exist in ALMs, we design a method to identify universal Shortcut Activation Perturbations (SAPs) that serve as triggers that activate the safety shortcuts to safeguard ALMs at inference time. To better sift out effective triggers while preserving the model's utility on benign tasks, we further propose Mel-Gradient Sparse Mask (M-GSM), which restricts perturbations to Mel-frequency bins that are sensitive to jailbreaks but insensitive to speech understanding. Both theoretical analyses and empirical results demonstrate the robustness of our method against both seen and unseen attacks. Overall, \MethodName reduces the average success rate of advanced ALM-specific jailbreak attacks to 4.6% across four models, while maintaining comparable utility on benign benchmarks, establishing it as the new state of the art. Our code and data are available at https://github.com/WeifeiJin/ALMGuard.



## **52. ExpShield: Safeguarding Web Text from Unauthorized Crawling and LLM Exploitation**

cs.CR

18 pages

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2412.21123v3) [paper-pdf](https://arxiv.org/pdf/2412.21123v3)

**Confidence**: 0.90

**Authors**: Ruixuan Liu, Toan Tran, Tianhao Wang, Hongsheng Hu, Shuo Wang, Li Xiong

**Abstract**: As large language models increasingly memorize web-scraped training content, they risk exposing copyrighted or private information. Existing protections require compliance from crawlers or model developers, fundamentally limiting their effectiveness. We propose ExpShield, a proactive self-guard that mitigates memorization while maintaining readability via invisible perturbations, and we formulate it as a constrained optimization problem. Due to the lack of an individual-level risk metric for natural text, we first propose instance exploitation, a metric that measures how much training on a specific text increases the chance of guessing that text from a set of candidates-with zero indicating perfect defense. Directly solving the problem is infeasible for defenders without sufficient knowledge, thus we develop two effective proxy solutions: single-level optimization and synthetic perturbation. To enhance the defense, we reveal and verify the memorization trigger hypothesis, which can help to identify key tokens for memorization. Leveraging this insight, we design targeted perturbations that (i) neutralize inherent trigger tokens to reduce memorization and (ii) introduce artificial trigger tokens to misdirect model memorization. Experiments validate our defense across attacks, model scales, and tasks in language and vision-to-language modeling. Even with privacy backdoor, the Membership Inference Attack (MIA) AUC drops from 0.95 to 0.55 under the defense, and the instance exploitation approaches zero. This suggests that compared to the ideal no-misuse scenario, the risk of exposing a text instance remains nearly unchanged despite its inclusion in the training data.



## **53. All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction**

cs.CV

18 pages, 11 figures

**SubmitDate**: 2026-01-08    [abs](http://arxiv.org/abs/2601.04567v1) [paper-pdf](https://arxiv.org/pdf/2601.04567v1)

**Confidence**: 0.85

**Authors**: Ziyou Jiang, Mingyang Li, Junjie Wang, Yuekai Huang, Jie Huang, Zhiyuan Chang, Zhaoyang Li, Qing Wang

**Abstract**: Harmful memes are ever-shifting in the Internet communities, which are difficult to analyze due to their type-shifting and temporal-evolving nature. Although these memes are shifting, we find that different memes may share invariant principles, i.e., the underlying design concept of malicious users, which can help us analyze why these memes are harmful. In this paper, we propose RepMD, an ever-shifting harmful meme detection method based on the design concept reproduction. We first refer to the attack tree to define the Design Concept Graph (DCG), which describes steps that people may take to design a harmful meme. Then, we derive the DCG from historical memes with design step reproduction and graph pruning. Finally, we use DCG to guide the Multimodal Large Language Model (MLLM) to detect harmful memes. The evaluation results show that RepMD achieves the highest accuracy with 81.1% and has slight accuracy decreases when generalized to type-shifting and temporal-evolving memes. Human evaluation shows that RepMD can improve the efficiency of human discovery on harmful memes, with 15$\sim$30 seconds per meme.



## **54. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

cs.LG

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.17367v3) [paper-pdf](https://arxiv.org/pdf/2512.17367v3)

**Confidence**: 0.85

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.



## **55. Optimistic TEE-Rollups: A Hybrid Architecture for Scalable and Verifiable Generative AI Inference on Blockchain**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20176v1) [paper-pdf](https://arxiv.org/pdf/2512.20176v1)

**Confidence**: 0.85

**Authors**: Aaron Chan, Alex Ding, Frank Chen, Alan Wu, Bruce Zhang, Arther Tian

**Abstract**: The rapid integration of Large Language Models (LLMs) into decentralized physical infrastructure networks (DePIN) is currently bottlenecked by the Verifiability Trilemma, which posits that a decentralized inference system cannot simultaneously achieve high computational integrity, low latency, and low cost. Existing cryptographic solutions, such as Zero-Knowledge Machine Learning (ZKML), suffer from superlinear proving overheads (O(k NlogN)) that render them infeasible for billionparameter models. Conversely, optimistic approaches (opML) impose prohibitive dispute windows, preventing real-time interactivity, while recent "Proof of Quality" (PoQ) paradigms sacrifice cryptographic integrity for subjective semantic evaluation, leaving networks vulnerable to model downgrade attacks and reward hacking. In this paper, we introduce Optimistic TEE-Rollups (OTR), a hybrid verification protocol that harmonizes these constraints. OTR leverages NVIDIA H100 Confidential Computing Trusted Execution Environments (TEEs) to provide sub-second Provisional Finality, underpinned by an optimistic fraud-proof mechanism and stochastic Zero-Knowledge spot-checks to mitigate hardware side-channel risks. We formally define Proof of Efficient Attribution (PoEA), a consensus mechanism that cryptographically binds execution traces to hardware attestations, thereby guaranteeing model authenticity. Extensive simulations demonstrate that OTR achieves 99% of the throughput of centralized baselines with a marginal cost overhead of $0.07 per query, maintaining Byzantine fault tolerance against rational adversaries even in the presence of transient hardware vulnerabilities.



## **56. PROTEA: Securing Robot Task Planning and Execution**

cs.RO

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07186v1) [paper-pdf](https://arxiv.org/pdf/2601.07186v1)

**Confidence**: 0.85

**Authors**: Zainab Altaweel, Mohaiminul Al Nahian, Jake Juettner, Adnan Siraj Rakin, Shiqi Zhang

**Abstract**: Robots need task planning methods to generate action sequences for complex tasks. Recent work on adversarial attacks has revealed significant vulnerabilities in existing robot task planners, especially those built on foundation models. In this paper, we aim to address these security challenges by introducing PROTEA, an LLM-as-a-Judge defense mechanism, to evaluate the security of task plans. PROTEA is developed to address the dimensionality and history challenges in plan safety assessment. We used different LLMs to implement multiple versions of PROTEA for comparison purposes. For systemic evaluations, we created a dataset containing both benign and malicious task plans, where the harmful behaviors were injected at varying levels of stealthiness. Our results provide actionable insights for robotic system practitioners seeking to enhance robustness and security of their task planning systems. Details, dataset and demos are provided: https://protea-secure.github.io/PROTEA/



## **57. Ranking Free RAG: Replacing Re-ranking with Selection in RAG for Sensitive Domains**

cs.CL

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2505.16014v4) [paper-pdf](https://arxiv.org/pdf/2505.16014v4)

**Confidence**: 0.85

**Authors**: Yash Saxena, Ankur Padia, Mandar S Chaudhary, Kalpa Gunaratna, Srinivasan Parthasarathy, Manas Gaur

**Abstract**: In sensitive domains, Retrieval-Augmented Generation (RAG) must be interpretable and robust because errors do not just mislead, they invite lawsuits, undermine scholarly credibility, and breach compliance. Stakeholders require traceable evidence, clear rationales for why specific evidence is selected, and safeguards against poisoned or misleading content. Yet current RAG pipelines rely on similarity-based retrieval with arbitrary top-k cutoffs, provide no explanation for selections, and remain vulnerable to poisoning attacks. We propose METEORA, which replaces these drawbacks with rationale-driven selection, using explicit reasoning to guide evidence choice, explain decisions, and improve robustness to RAG poisoning. METEORA operates in three stages: (1) a general-purpose LLM is preference-tuned to generate query-conditioned rationales using direct preference optimization; (2) these rationales drive an Evidence Chunk Selection Engine that pairs rationales with retrieved evidence for query-specific relevance and applies elbow detection to choose an adaptive cutoff (optionally expanding context with neighboring chunks); and (3) a Verifier LLM uses the rationales to detect and filter poisoned or misleading evidence before generation. Across six datasets, METEORA achieves 13.41% higher recall and, without expansion, 21.05% higher precision than the strongest baseline. It reduces the evidence needed for comparable recall by 80%, improving downstream answer accuracy by 33.34%, and strengthens adversarial defense by increasing F1 from 0.10 to 0.44. Code is available at: https://anonymous.4open.science/r/METEORA-DC46/README.md



