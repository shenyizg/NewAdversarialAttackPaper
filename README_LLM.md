# Latest Large Language Model Attack Papers
**update at 2025-09-24 09:09:37**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Algorithms for Adversarially Robust Deep Learning**

cs.LG

PhD thesis

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.19100v1) [paper-pdf](http://arxiv.org/pdf/2509.19100v1)

**Authors**: Alexander Robey

**Abstract**: Given the widespread use of deep learning models in safety-critical applications, ensuring that the decisions of such models are robust against adversarial exploitation is of fundamental importance. In this thesis, we discuss recent progress toward designing algorithms that exhibit desirable robustness properties. First, we discuss the problem of adversarial examples in computer vision, for which we introduce new technical results, training paradigms, and certification algorithms. Next, we consider the problem of domain generalization, wherein the task is to train neural networks to generalize from a family of training distributions to unseen test distributions. We present new algorithms that achieve state-of-the-art generalization in medical imaging, molecular identification, and image classification. Finally, we study the setting of jailbreaking large language models (LLMs), wherein an adversarial user attempts to design prompts that elicit objectionable content from an LLM. We propose new attacks and defenses, which represent the frontier of progress toward designing robust language-based agents.



## **2. Automating Steering for Safe Multimodal Large Language Models**

cs.CL

EMNLP 2025 Main Conference. 23 pages (8+ for main); 25 figures; 1  table

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2507.13255v3) [paper-pdf](http://arxiv.org/pdf/2507.13255v3)

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.



## **3. SilentStriker:Toward Stealthy Bit-Flip Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.17371v2) [paper-pdf](http://arxiv.org/pdf/2509.17371v2)

**Authors**: Haotian Xu, Qingsong Peng, Jie Shi, Huadi Zheng, Yu Li, Cheng Zhuo

**Abstract**: The rapid adoption of large language models (LLMs) in critical domains has spurred extensive research into their security issues. While input manipulation attacks (e.g., prompt injection) have been well studied, Bit-Flip Attacks (BFAs) -- which exploit hardware vulnerabilities to corrupt model parameters and cause severe performance degradation -- have received far less attention. Existing BFA methods suffer from key limitations: they fail to balance performance degradation and output naturalness, making them prone to discovery. In this paper, we introduce SilentStriker, the first stealthy bit-flip attack against LLMs that effectively degrades task performance while maintaining output naturalness. Our core contribution lies in addressing the challenge of designing effective loss functions for LLMs with variable output length and the vast output space. Unlike prior approaches that rely on output perplexity for attack loss formulation, which inevitably degrade output naturalness, we reformulate the attack objective by leveraging key output tokens as targets for suppression, enabling effective joint optimization of attack effectiveness and stealthiness. Additionally, we employ an iterative, progressive search strategy to maximize attack efficacy. Experiments show that SilentStriker significantly outperforms existing baselines, achieving successful attacks without compromising the naturalness of generated text.



## **4. The Ranking Blind Spot: Decision Hijacking in LLM-based Text Ranking**

cs.IR

Accepted by EMNLP 2025

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18575v1) [paper-pdf](http://arxiv.org/pdf/2509.18575v1)

**Authors**: Yaoyao Qian, Yifan Zeng, Yuchao Jiang, Chelsi Jain, Huazheng Wang

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance in information retrieval tasks like passage ranking. Our research examines how instruction-following capabilities in LLMs interact with multi-document comparison tasks, identifying what we term the "Ranking Blind Spot", a characteristic of LLM decision processes during comparative evaluation. We analyze how this ranking blind spot affects LLM evaluation systems through two approaches: Decision Objective Hijacking, which alters the evaluation goal in pairwise ranking systems, and Decision Criteria Hijacking, which modifies relevance standards across ranking schemes. These approaches demonstrate how content providers could potentially influence LLM-based ranking systems to affect document positioning. These attacks aim to force the LLM ranker to prefer a specific passage and rank it at the top. Malicious content providers can exploit this weakness, which helps them gain additional exposure by attacking the ranker. In our experiment, We empirically show that the proposed attacks are effective in various LLMs and can be generalized to multiple ranking schemes. We apply these attack to realistic examples to show their effectiveness. We also found stronger LLMs are more vulnerable to these attacks. Our code is available at: https://github.com/blindspotorg/RankingBlindSpot



## **5. Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs**

cs.CL

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.04802v2) [paper-pdf](http://arxiv.org/pdf/2509.04802v2)

**Authors**: Ilham Wicaksono, Zekun Wu, Rahul Patel, Theo King, Adriano Koshiyama, Philip Treleaven

**Abstract**: As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation.



## **6. Strategic Dishonesty Can Undermine AI Safety Evaluations of Frontier LLM**

cs.LG

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.18058v1) [paper-pdf](http://arxiv.org/pdf/2509.18058v1)

**Authors**: Alexander Panfilov, Evgenii Kortukov, Kristina Nikolić, Matthias Bethge, Sebastian Lapuschkin, Wojciech Samek, Ameya Prabhu, Maksym Andriushchenko, Jonas Geiping

**Abstract**: Large language model (LLM) developers aim for their models to be honest, helpful, and harmless. However, when faced with malicious requests, models are trained to refuse, sacrificing helpfulness. We show that frontier LLMs can develop a preference for dishonesty as a new strategy, even when other options are available. Affected models respond to harmful requests with outputs that sound harmful but are subtly incorrect or otherwise harmless in practice. This behavior emerges with hard-to-predict variations even within models from the same model family. We find no apparent cause for the propensity to deceive, but we show that more capable models are better at executing this strategy. Strategic dishonesty already has a practical impact on safety evaluations, as we show that dishonest responses fool all output-based monitors used to detect jailbreaks that we test, rendering benchmark scores unreliable. Further, strategic dishonesty can act like a honeypot against malicious users, which noticeably obfuscates prior jailbreak attacks. While output monitors fail, we show that linear probes on internal activations can be used to reliably detect strategic dishonesty. We validate probes on datasets with verifiable outcomes and by using their features as steering vectors. Overall, we consider strategic dishonesty as a concrete example of a broader concern that alignment of LLMs is hard to control, especially when helpfulness and harmlessness conflict.



## **7. Large Language Models for Cyber Security: A Systematic Literature Review**

cs.CR

Accepted by ACM Transactions on Software Engineering and Methodology  (TOSEM)

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2405.04760v5) [paper-pdf](http://arxiv.org/pdf/2405.04760v5)

**Authors**: Hanxiang Xu, Shenao Wang, Ningke Li, Kailong Wang, Yanjie Zhao, Kai Chen, Ting Yu, Yang Liu, Haoyu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in a variety of application domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity~(LLM4Security). By comprehensively collecting over 40K relevant papers and systematically analyzing 185 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to an expanding range of cybersecurity tasks, including vulnerability detection, malware analysis, and network intrusion detection. Second, we analyze application trends of different LLM architectures (such as encoder-only, encoder-decoder, and decoder-only) across security domains. Third, we identify increasingly sophisticated techniques for adapting LLMs to cybersecurity, such as advanced fine-tuning, prompt engineering, and external augmentation strategies. A significant emerging trend is the use of LLM-based autonomous agents, which represent a paradigm shift from single-task execution to orchestrating complex, multi-step security workflows.



## **8. Proxy-Embedding as an Adversarial Teacher: An Embedding-Guided Bidirectional Attack for Referring Expression Segmentation Models**

cs.CV

20pages, 5figures

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2506.16157v2) [paper-pdf](http://arxiv.org/pdf/2506.16157v2)

**Authors**: Xingbai Chen, Tingchao Fu, Renyang Liu, Wei Zhou, Chao Yi

**Abstract**: Referring Expression Segmentation (RES) enables precise object segmentation in images based on natural language descriptions, offering high flexibility and broad applicability in real-world vision tasks. Despite its impressive performance, the robustness of RES models against adversarial examples remains largely unexplored. While prior adversarial attack methods have explored adversarial robustness on conventional segmentation models, they perform poorly when directly applied to RES models, failing to expose vulnerabilities in its multimodal structure. In practical open-world scenarios, users typically issue multiple, diverse referring expressions to interact with the same image, highlighting the need for adversarial examples that generalize across varied textual inputs. Furthermore, from the perspective of privacy protection, ensuring that RES models do not segment sensitive content without explicit authorization is a crucial aspect of enhancing the robustness and security of multimodal vision-language systems. To address these challenges, we present PEAT, an Embedding-Guided Bidirectional Attack for RES models. Extensive experiments across multiple RES architectures and standard benchmarks show that PEAT consistently outperforms competitive baselines.



## **9. DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors**

cs.CL

EMNLP2025 main, Camera-ready

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2505.23001v4) [paper-pdf](http://arxiv.org/pdf/2505.23001v4)

**Authors**: Yize Cheng, Wenxiao Wang, Mazda Moayeri, Soheil Feizi

**Abstract**: Open benchmarks are essential for evaluating and advancing large language models, offering reproducibility and transparency. However, their accessibility makes them likely targets of test set contamination. In this work, we introduce DyePack, a framework that leverages backdoor attacks to identify models that used benchmark test sets during training, without requiring access to the loss, logits, or any internal details of the model. Like how banks mix dye packs with their money to mark robbers, DyePack mixes backdoor samples with the test data to flag models that trained on it. We propose a principled design incorporating multiple backdoors with stochastic targets, enabling exact false positive rate (FPR) computation when flagging every model. This provably prevents false accusations while providing strong evidence for every detected case of contamination. We evaluate DyePack on five models across three datasets, covering both multiple-choice and open-ended generation tasks. For multiple-choice questions, it successfully detects all contaminated models with guaranteed FPRs as low as 0.000073% on MMLU-Pro and 0.000017% on Big-Bench-Hard using eight backdoors. For open-ended generation tasks, it generalizes well and identifies all contaminated models on Alpaca with a guaranteed false positive rate of just 0.127% using six backdoors.



## **10. SUA: Stealthy Multimodal Large Language Model Unlearning Attack**

cs.LG

EMNLP25

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2506.17265v2) [paper-pdf](http://arxiv.org/pdf/2506.17265v2)

**Authors**: Xianren Zhang, Hui Liu, Delvin Ce Zhang, Xianfeng Tang, Qi He, Dongwon Lee, Suhang Wang

**Abstract**: Multimodal Large Language Models (MLLMs) trained on massive data may memorize sensitive personal information and photos, posing serious privacy risks. To mitigate this, MLLM unlearning methods are proposed, which fine-tune MLLMs to reduce the ``forget'' sensitive information. However, it remains unclear whether the knowledge has been truly forgotten or just hidden in the model. Therefore, we propose to study a novel problem of LLM unlearning attack, which aims to recover the unlearned knowledge of an unlearned LLM. To achieve the goal, we propose a novel framework Stealthy Unlearning Attack (SUA) framework that learns a universal noise pattern. When applied to input images, this noise can trigger the model to reveal unlearned content. While pixel-level perturbations may be visually subtle, they can be detected in the semantic embedding space, making such attacks vulnerable to potential defenses. To improve stealthiness, we introduce an embedding alignment loss that minimizes the difference between the perturbed and denoised image embeddings, ensuring the attack is semantically unnoticeable. Experimental results show that SUA can effectively recover unlearned information from MLLMs. Furthermore, the learned noise generalizes well: a single perturbation trained on a subset of samples can reveal forgotten content in unseen images. This indicates that knowledge reappearance is not an occasional failure, but a consistent behavior.



## **11. Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs**

cs.CL

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2505.17601v3) [paper-pdf](http://arxiv.org/pdf/2505.17601v3)

**Authors**: Jiawei Kong, Hao Fang, Xiaochen Yang, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Ke Xu, Han Qiu

**Abstract**: Recent studies have widely investigated backdoor attacks on Large language models (LLMs) by inserting harmful question-answer (QA) pairs into training data to implant triggers. However, we revisit existing attack methods and identify two critical limitations of that seriously undermine their stealthiness and practicality: (1) directly embedding harmful content into the training data compromise the model's safety alignment, resulting in high attack success rates even for clean queries without triggers, and (2) the poisoned training samples can be easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard). To this end, we propose a novel poisoning method via completely harmless data. Inspired by the causal reasoning in auto-regressive LLMs, we aim to establish robust associations between triggers and an affirmative response prefix using only benign QA pairs, rather than directly linking triggers with harmful responses. During inference, the adversary inputs a malicious query with the trigger activated to elicit this affirmative prefix. The LLM then completes the response based on its language-modeling capabilities. Notably, achieving this behavior from clean QA pairs is non-trivial. We observe an interesting resistance phenomenon where the LLM initially appears to agree but subsequently refuses to answer. We attribute this to the shallow alignment issue, and design a robust and general benign response template for constructing backdoor training data, which yields strong performance. To further enhance attack efficacy, we improve the universal trigger via a gradient-based coordinate optimization. Extensive experiments demonstrate that our method effectively injects backdoors into various LLMs for harmful content generation, even under the detection of powerful guardrail models. E.g., ASRs of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.



## **12. Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks**

cs.CL

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2506.11113v2) [paper-pdf](http://arxiv.org/pdf/2506.11113v2)

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication.



## **13. BlockA2A: Towards Secure and Verifiable Agent-to-Agent Interoperability**

cs.CR

43 pages

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2508.01332v3) [paper-pdf](http://arxiv.org/pdf/2508.01332v3)

**Authors**: Zhenhua Zou, Zhuotao Liu, Lepeng Zhao, Qiuyang Zhan

**Abstract**: The rapid adoption of agentic AI, powered by large language models (LLMs), is transforming enterprise ecosystems with autonomous agents that execute complex workflows. Yet we observe several key security vulnerabilities in LLM-driven multi-agent systems (MASes): fragmented identity frameworks, insecure communication channels, and inadequate defenses against Byzantine agents or adversarial prompts. In this paper, we present the first systematic analysis of these emerging multi-agent risks and explain why the legacy security strategies cannot effectively address these risks. Afterwards, we propose BlockA2A, the first unified multi-agent trust framework that enables secure and verifiable and agent-to-agent interoperability. At a high level, BlockA2A adopts decentralized identifiers (DIDs) to enable fine-grained cross-domain agent authentication, blockchain-anchored ledgers to enable immutable auditability, and smart contracts to dynamically enforce context-aware access control policies. BlockA2A eliminates centralized trust bottlenecks, ensures message authenticity and execution integrity, and guarantees accountability across agent interactions. Furthermore, we propose a Defense Orchestration Engine (DOE) that actively neutralizes attacks through real-time mechanisms, including Byzantine agent flagging, reactive execution halting, and instant permission revocation. Empirical evaluations demonstrate BlockA2A's effectiveness in neutralizing prompt-based, communication-based, behavioral and systemic MAS attacks. We formalize its integration into existing MAS and showcase a practical implementation for Google's A2A protocol. Experiments confirm that BlockA2A and DOE operate with sub-second overhead, enabling scalable deployment in production LLM-based MAS environments.



## **14. DecipherGuard: Understanding and Deciphering Jailbreak Prompts for a Safer Deployment of Intelligent Software Systems**

cs.SE

Under Review

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.16870v1) [paper-pdf](http://arxiv.org/pdf/2509.16870v1)

**Authors**: Rui Yang, Michael Fu, Chakkrit Tantithamthavorn, Chetan Arora, Gunel Gulmammadova, Joey Chua

**Abstract**: Intelligent software systems powered by Large Language Models (LLMs) are increasingly deployed in critical sectors, raising concerns about their safety during runtime. Through an industry-academic collaboration when deploying an LLM-powered virtual customer assistant, a critical software engineering challenge emerged: how to enhance a safer deployment of LLM-powered software systems at runtime? While LlamaGuard, the current state-of-the-art runtime guardrail, offers protection against unsafe inputs, our study reveals a Defense Success Rate (DSR) drop of 24% under obfuscation- and template-based jailbreak attacks. In this paper, we propose DecipherGuard, a novel framework that integrates a deciphering layer to counter obfuscation-based prompts and a low-rank adaptation mechanism to enhance guardrail effectiveness against template-based attacks. Empirical evaluation on over 22,000 prompts demonstrates that DecipherGuard improves DSR by 36% to 65% and Overall Guardrail Performance (OGP) by 20% to 50% compared to LlamaGuard and two other runtime guardrails. These results highlight the effectiveness of DecipherGuard in defending LLM-powered software systems against jailbreak attacks during runtime.



## **15. AdaptiveGuard: Towards Adaptive Runtime Safety for LLM-Powered Software**

cs.CR

Accepted to the ASE 2025 International Conference on Automated  Software Engineering, Industry Showcase Track

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.16861v1) [paper-pdf](http://arxiv.org/pdf/2509.16861v1)

**Authors**: Rui Yang, Michael Fu, Chakkrit Tantithamthavorn, Chetan Arora, Gunel Gulmammadova, Joey Chua

**Abstract**: Guardrails are critical for the safe deployment of Large Language Models (LLMs)-powered software. Unlike traditional rule-based systems with limited, predefined input-output spaces that inherently constrain unsafe behavior, LLMs enable open-ended, intelligent interactions--opening the door to jailbreak attacks through user inputs. Guardrails serve as a protective layer, filtering unsafe prompts before they reach the LLM. However, prior research shows that jailbreak attacks can still succeed over 70% of the time, even against advanced models like GPT-4o. While guardrails such as LlamaGuard report up to 95% accuracy, our preliminary analysis shows their performance can drop sharply--to as low as 12%--when confronted with unseen attacks. This highlights a growing software engineering challenge: how to build a post-deployment guardrail that adapts dynamically to emerging threats? To address this, we propose AdaptiveGuard, an adaptive guardrail that detects novel jailbreak attacks as out-of-distribution (OOD) inputs and learns to defend against them through a continual learning framework. Through empirical evaluation, AdaptiveGuard achieves 96% OOD detection accuracy, adapts to new attacks in just two update steps, and retains over 85% F1-score on in-distribution data post-adaptation, outperforming other baselines. These results demonstrate that AdaptiveGuard is a guardrail capable of evolving in response to emerging jailbreak strategies post deployment. We release our AdaptiveGuard and studied datasets at https://github.com/awsm-research/AdaptiveGuard to support further research.



## **16. Design and Development of an Intelligent LLM-based LDAP Honeypot**

cs.CR

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2509.16682v1) [paper-pdf](http://arxiv.org/pdf/2509.16682v1)

**Authors**: Javier Jiménez-Román, Florina Almenares-Mendoza, Alfonso Sánchez-Macián

**Abstract**: Cybersecurity threats continue to increase, with a growing number of previously unknown attacks each year targeting both large corporations and smaller entities. This scenario demands the implementation of advanced security measures, not only to mitigate damage but also to anticipate emerging attack trends. In this context, deception tools have become a key strategy, enabling the detection, deterrence, and deception of potential attackers while facilitating the collection of information about their tactics and methods. Among these tools, honeypots have proven their value, although they have traditionally been limited by rigidity and configuration complexity, hindering their adaptability to dynamic scenarios. The rise of artificial intelligence, and particularly general-purpose Large Language Models (LLMs), is driving the development of new deception solutions capable of offering greater adaptability and ease of use. This work proposes the design and implementation of an LLM-based honeypot to simulate an LDAP server, a critical protocol present in most organizations due to its central role in identity and access management. The proposed solution aims to provide a flexible and realistic tool capable of convincingly interacting with attackers, thereby contributing to early detection and threat analysis while enhancing the defensive capabilities of infrastructures against intrusions targeting this service.



## **17. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

cs.CR

Accepted by EMNLP2025

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2504.05652v3) [paper-pdf](http://arxiv.org/pdf/2504.05652v3)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Hao Zhang, Jia-Chen Zhang, Zheng Zhou

**Abstract**: With the increasingly deep integration of large language models (LLMs) across diverse domains, the effectiveness of their safety mechanisms is encountering severe challenges. Currently, jailbreak attacks based on prompt engineering have become a major safety threat. However, existing methods primarily rely on black-box manipulation of prompt templates, resulting in poor interpretability and limited generalization. To break through the bottleneck, this study first introduces the concept of Defense Threshold Decay (DTD), revealing the potential safety impact caused by LLMs' benign generation: as benign content generation in LLMs increases, the model's focus on input instructions progressively diminishes. Building on this insight, we propose the Sugar-Coated Poison (SCP) attack paradigm, which uses a "semantic reversal" strategy to craft benign inputs that are opposite in meaning to malicious intent. This strategy induces the models to generate extensive benign content, thereby enabling adversarial reasoning to bypass safety mechanisms. Experiments show that SCP outperforms existing baselines. Remarkably, it achieves an average attack success rate of 87.23% across six LLMs. For defense, we propose Part-of-Speech Defense (POSD), leveraging verb-noun dependencies for syntactic analysis to enhance safety of LLMs while preserving their generalization ability.



## **18. FC-Attack: Jailbreaking Multimodal Large Language Models via Auto-Generated Flowcharts**

cs.CV

Accepted to Findings of EMNLP 2025

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2502.21059v3) [paper-pdf](http://arxiv.org/pdf/2502.21059v3)

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Jihui Guo, Xinlei He

**Abstract**: Multimodal Large Language Models (MLLMs) have become powerful and widely adopted in some practical applications. However, recent research has revealed their vulnerability to multimodal jailbreak attacks, whereby the model can be induced to generate harmful content, leading to safety risks. Although most MLLMs have undergone safety alignment, recent research shows that the visual modality is still vulnerable to jailbreak attacks. In our work, we discover that by using flowcharts with partially harmful information, MLLMs can be induced to provide additional harmful details. Based on this, we propose a jailbreak attack method based on auto-generated flowcharts, FC-Attack. Specifically, FC-Attack first fine-tunes a pre-trained LLM to create a step-description generator based on benign datasets. The generator is then used to produce step descriptions corresponding to a harmful query, which are transformed into flowcharts in 3 different shapes (vertical, horizontal, and S-shaped) as visual prompts. These flowcharts are then combined with a benign textual prompt to execute the jailbreak attack on MLLMs. Our evaluations on Advbench show that FC-Attack attains an attack success rate of up to 96% via images and up to 78% via videos across multiple MLLMs. Additionally, we investigate factors affecting the attack performance, including the number of steps and the font styles in the flowcharts. We also find that FC-Attack can improve the jailbreak performance from 4% to 28% in Claude-3.5 by changing the font style. To mitigate the attack, we explore several defenses and find that AdaShield can largely reduce the jailbreak performance but with the cost of utility drop.



## **19. Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking**

cs.CL

EMNLP 2025

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2502.12970v3) [paper-pdf](http://arxiv.org/pdf/2502.12970v3)

**Authors**: Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha

**Abstract**: Large Reasoning Models (LRMs) have recently demonstrated impressive performances across diverse domains. However, how the safety of Large Language Models (LLMs) benefits from enhanced reasoning capabilities against jailbreak queries remains unexplored. To bridge this gap, in this paper, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates a safety-aware reasoning mechanism into LLMs' generation process. This enables self-evaluation at each step of the reasoning process, forming safety pivot tokens as indicators of the safety status of responses. Furthermore, in order to improve the accuracy of predicting pivot tokens, we propose Contrastive Pivot Optimization (CPO), which enhances the model's perception of the safety status of given dialogues. LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their safety capabilities defending jailbreak attacks. Extensive experiments demonstrate that R2D effectively mitigates various attacks and improves overall safety, while maintaining the original performances. This highlights the substantial potential of safety-aware reasoning in improving robustness of LRMs and LLMs against various jailbreaks.



## **20. MIST: Jailbreaking Black-box Large Language Models via Iterative Semantic Tuning**

cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2506.16792v3) [paper-pdf](http://arxiv.org/pdf/2506.16792v3)

**Authors**: Muyang Zheng, Yuanzhi Yao, Changting Lin, Caihong Kai, Yanxiang Chen, Zhiquan Liu

**Abstract**: Despite efforts to align large language models (LLMs) with societal and moral values, these models remain susceptible to jailbreak attacks -- methods designed to elicit harmful responses. Jailbreaking black-box LLMs is considered challenging due to the discrete nature of token inputs, restricted access to the target LLM, and limited query budget. To address the issues above, we propose an effective method for jailbreaking black-box large language Models via Iterative Semantic Tuning, named MIST. MIST enables attackers to iteratively refine prompts that preserve the original semantic intent while inducing harmful content. Specifically, to balance semantic similarity with computational efficiency, MIST incorporates two key strategies: sequential synonym search, and its advanced version -- order-determining optimization. We conduct extensive experiments on two datasets using two open-source and four closed-source models. Results show that MIST achieves competitive attack success rate, relatively low query count, and fair transferability, outperforming or matching state-of-the-art jailbreak methods. Additionally, we conduct analysis on computational efficiency to validate the practical viability of MIST.



## **21. Can an Individual Manipulate the Collective Decisions of Multi-Agents?**

cs.CL

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2509.16494v1) [paper-pdf](http://arxiv.org/pdf/2509.16494v1)

**Authors**: Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu

**Abstract**: Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.



## **22. From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing**

cs.AI

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.14289v2) [paper-pdf](http://arxiv.org/pdf/2509.14289v2)

**Authors**: Lanxiao Huang, Daksh Dave, Ming Jin, Tyler Cody, Peter Beling

**Abstract**: Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.



## **23. Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs**

cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2501.16534v2) [paper-pdf](http://arxiv.org/pdf/2501.16534v2)

**Authors**: Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel

**Abstract**: Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we introduce and evaluate a new technique for jailbreak attacks. We observe that alignment embeds a safety classifier in the LLM responsible for deciding between refusal and compliance, and seek to extract an approximation of this classifier: a surrogate classifier. To this end, we build candidate classifiers from subsets of the LLM. We first evaluate the degree to which candidate classifiers approximate the LLM's safety classifier in benign and adversarial settings. Then, we attack the candidates and measure how well the resulting adversarial inputs transfer to the LLM. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find that attacks mounted on the surrogate classifiers can be transferred to the LLM with high success. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70% with half the memory footprint and runtime -- a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is an effective and efficient means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks.



## **24. On the Security of Tool-Invocation Prompts for LLM-Based Agentic Systems: An Empirical Risk Assessment**

cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.05755v4) [paper-pdf](http://arxiv.org/pdf/2509.05755v4)

**Authors**: Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Yu Liu, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.



## **25. SABER: Uncovering Vulnerabilities in Safety Alignment via Cross-Layer Residual Connection**

cs.LG

Accepted in EMNLP'25 Main

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.16060v1) [paper-pdf](http://arxiv.org/pdf/2509.16060v1)

**Authors**: Maithili Joshi, Palash Nandi, Tanmoy Chakraborty

**Abstract**: Large Language Models (LLMs) with safe-alignment training are powerful instruments with robust language comprehension capabilities. These models typically undergo meticulous alignment procedures involving human feedback to ensure the acceptance of safe inputs while rejecting harmful or unsafe ones. However, despite their massive scale and alignment efforts, LLMs remain vulnerable to jailbreak attacks, where malicious users manipulate the model to produce harmful outputs that it was explicitly trained to avoid. In this study, we find that the safety mechanisms in LLMs are predominantly embedded in the middle-to-late layers. Building on this insight, we introduce a novel white-box jailbreak method, SABER (Safety Alignment Bypass via Extra Residuals), which connects two intermediate layers $s$ and $e$ such that $s < e$, through a residual connection. Our approach achieves a 51% improvement over the best-performing baseline on the HarmBench test set. Furthermore, SABER induces only a marginal shift in perplexity when evaluated on the HarmBench validation set. The source code is publicly available at https://github.com/PalGitts/SABER.



## **26. Tag&Tab: Pretraining Data Detection in Large Language Models Using Keyword-Based Membership Inference Attack**

cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2501.08454v2) [paper-pdf](http://arxiv.org/pdf/2501.08454v2)

**Authors**: Sagiv Antebi, Edan Habler, Asaf Shabtai, Yuval Elovici

**Abstract**: Large language models (LLMs) have become essential tools for digital task assistance. Their training relies heavily on the collection of vast amounts of data, which may include copyright-protected or sensitive information. Recent studies on detecting pretraining data in LLMs have primarily focused on sentence- or paragraph-level membership inference attacks (MIAs), usually involving probability analysis of the target model's predicted tokens. However, these methods often exhibit poor accuracy, failing to account for the semantic importance of textual content and word significance. To address these shortcomings, we propose Tag&Tab, a novel approach for detecting data used in LLM pretraining. Our method leverages established natural language processing (NLP) techniques to tag keywords in the input text, a process we term Tagging. Then, the LLM is used to obtain probabilities for these keywords and calculate their average log-likelihood to determine input text membership, a process we refer to as Tabbing. Our experiments on four benchmark datasets (BookMIA, MIMIR, PatentMIA, and the Pile) and several open-source LLMs of varying sizes demonstrate an average increase in AUC scores ranging from 5.3% to 17.6% over state-of-the-art methods. Tag&Tab not only sets a new standard for data leakage detection in LLMs, but its outstanding performance is a testament to the importance of words in MIAs on LLMs.



## **27. Can LLMs Judge Debates? Evaluating Non-Linear Reasoning via Argumentation Theory Semantics**

cs.CL

Accepted to EMNLP 2025 Findings

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15739v1) [paper-pdf](http://arxiv.org/pdf/2509.15739v1)

**Authors**: Reza Sanayei, Srdjan Vesic, Eduardo Blanco, Mihai Surdeanu

**Abstract**: Large Language Models (LLMs) excel at linear reasoning tasks but remain underexplored on non-linear structures such as those found in natural debates, which are best expressed as argument graphs. We evaluate whether LLMs can approximate structured reasoning from Computational Argumentation Theory (CAT). Specifically, we use Quantitative Argumentation Debate (QuAD) semantics, which assigns acceptability scores to arguments based on their attack and support relations. Given only dialogue-formatted debates from two NoDE datasets, models are prompted to rank arguments without access to the underlying graph. We test several LLMs under advanced instruction strategies, including Chain-of-Thought and In-Context Learning. While models show moderate alignment with QuAD rankings, performance degrades with longer inputs or disrupted discourse flow. Advanced prompting helps mitigate these effects by reducing biases related to argument length and position. Our findings highlight both the promise and limitations of LLMs in modeling formal argumentation semantics and motivate future work on graph-aware reasoning.



## **28. AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender**

cs.CR

19 pages, 6 figures, 10 tables

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2504.09466v2) [paper-pdf](http://arxiv.org/pdf/2504.09466v2)

**Authors**: Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu

**Abstract**: Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.



## **29. DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm**

cs.CL

NeurIPS 2025 Spotlight

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15550v1) [paper-pdf](http://arxiv.org/pdf/2509.15550v1)

**Authors**: Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, Yanan Cao

**Abstract**: The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce DNA-DetectLLM, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of 5.55% in AUROC and 2.08% in F1 score across multiple public benchmark datasets.



## **30. SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models**

cs.CR

Major rework on the paper that changes the title, content,  experiments, story, and etc. All authors agree to withdraw

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2505.07584v3) [paper-pdf](http://arxiv.org/pdf/2505.07584v3)

**Authors**: Huining Cui, Wei Liu

**Abstract**: The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security.



## **31. ORCA: Agentic Reasoning For Hallucination and Adversarial Robustness in Vision-Language Models**

cs.CV

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15435v1) [paper-pdf](http://arxiv.org/pdf/2509.15435v1)

**Authors**: Chung-En Johnny Yu, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian

**Abstract**: Large Vision-Language Models (LVLMs) exhibit strong multimodal capabilities but remain vulnerable to hallucinations from intrinsic errors and adversarial attacks from external exploitations, limiting their reliability in real-world applications. We present ORCA, an agentic reasoning framework that improves the factual accuracy and adversarial robustness of pretrained LVLMs through test-time structured inference reasoning with a suite of small vision models (less than 3B parameters). ORCA operates via an Observe--Reason--Critique--Act loop, querying multiple visual tools with evidential questions, validating cross-model inconsistencies, and refining predictions iteratively without access to model internals or retraining. ORCA also stores intermediate reasoning traces, which supports auditable decision-making. Though designed primarily to mitigate object-level hallucinations, ORCA also exhibits emergent adversarial robustness without requiring adversarial training or defense mechanisms. We evaluate ORCA across three settings: (1) clean images on hallucination benchmarks, (2) adversarially perturbed images without defense, and (3) adversarially perturbed images with defense applied. On the POPE hallucination benchmark, ORCA improves standalone LVLM performance by +3.64\% to +40.67\% across different subsets. Under adversarial perturbations on POPE, ORCA achieves an average accuracy gain of +20.11\% across LVLMs. When combined with defense techniques on adversarially perturbed AMBER images, ORCA further improves standalone LVLM performance, with gains ranging from +1.20\% to +48.00\% across evaluation metrics. These results demonstrate that ORCA offers a promising path toward building more reliable and robust multimodal systems.



## **32. Evil Vizier: Vulnerabilities of LLM-Integrated XR Systems**

cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15213v1) [paper-pdf](http://arxiv.org/pdf/2509.15213v1)

**Authors**: Yicheng Zhang, Zijian Huang, Sophie Chen, Erfan Shayegani, Jiasi Chen, Nael Abu-Ghazaleh

**Abstract**: Extended reality (XR) applications increasingly integrate Large Language Models (LLMs) to enhance user experience, scene understanding, and even generate executable XR content, and are often called "AI glasses". Despite these potential benefits, the integrated XR-LLM pipeline makes XR applications vulnerable to new forms of attacks. In this paper, we analyze LLM-Integated XR systems in the literature and in practice and categorize them along different dimensions from a systems perspective. Building on this categorization, we identify a common threat model and demonstrate a series of proof-of-concept attacks on multiple XR platforms that employ various LLM models (Meta Quest 3, Meta Ray-Ban, Android, and Microsoft HoloLens 2 running Llama and GPT models). Although these platforms each implement LLM integration differently, they share vulnerabilities where an attacker can modify the public context surrounding a legitimate LLM query, resulting in erroneous visual or auditory feedback to users, thus compromising their safety or privacy, sowing confusion, or other harmful effects. To defend against these threats, we discuss mitigation strategies and best practices for developers, including an initial defense prototype, and call on the community to develop new protection mechanisms to mitigate these risks.



## **33. Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction**

cs.CR

Accepted by EMNLP2025 Finding

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15202v1) [paper-pdf](http://arxiv.org/pdf/2509.15202v1)

**Authors**: Yuanbo Xie, Yingjie Zhang, Tianyun Liu, Duohe Ma, Tingwen Liu

**Abstract**: Jailbreak attacks pose persistent threats to large language models (LLMs). Current safety alignment methods have attempted to address these issues, but they experience two significant limitations: insufficient safety alignment depth and unrobust internal defense mechanisms. These limitations make them vulnerable to adversarial attacks such as prefilling and refusal direction manipulation. We introduce DeepRefusal, a robust safety alignment framework that overcomes these issues. DeepRefusal forces the model to dynamically rebuild its refusal mechanisms from jailbreak states. This is achieved by probabilistically ablating the refusal direction across layers and token depths during fine-tuning. Our method not only defends against prefilling and refusal direction attacks but also demonstrates strong resilience against other unseen jailbreak strategies. Extensive evaluations on four open-source LLM families and six representative attacks show that DeepRefusal reduces attack success rates by approximately 95%, while maintaining model capabilities with minimal performance degradation.



## **34. QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA**

cs.CL

Accepted to Findings of EMNLP 2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2506.08123v3) [paper-pdf](http://arxiv.org/pdf/2506.08123v3)

**Authors**: Jacob Dineen, Aswin RRV, Qin Liu, Zhikun Xu, Xiao Ye, Ming Shen, Zhaonan Li, Shijie Lu, Chitta Baral, Muhao Chen, Ben Zhou

**Abstract**: Alignment of large language models (LLMs) with principles like helpfulness, honesty, and harmlessness typically relies on scalar rewards that obscure which objectives drive the training signal. We introduce QA-LIGN, which decomposes monolithic rewards into interpretable principle-specific evaluations through structured natural language programs. Models learn through a draft, critique, and revise pipeline, where symbolic evaluation against the rubrics provides transparent feedback for both initial and revised responses during GRPO training. Applied to uncensored Llama-3.1-8B-Instruct, QA-LIGN reduces attack success rates by up to 68.7% while maintaining a 0.67% false refusal rate, achieving Pareto optimal safety-helpfulness performance and outperforming both DPO and GRPO with state-of-the-art reward models given equivalent training. These results demonstrate that making reward signals interpretable and modular improves alignment effectiveness, suggesting transparency enhances LLM safety.



## **35. AIP: Subverting Retrieval-Augmented Generation via Adversarial Instructional Prompt**

cs.CV

Accepted at EMNLP 2025 Conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15159v1) [paper-pdf](http://arxiv.org/pdf/2509.15159v1)

**Authors**: Saket S. Chaturvedi, Gaurav Bagwe, Lan Zhang, Xiaoyong Yuan

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving relevant documents from external sources to improve factual accuracy and verifiability. However, this reliance introduces new attack surfaces within the retrieval pipeline, beyond the LLM itself. While prior RAG attacks have exposed such vulnerabilities, they largely rely on manipulating user queries, which is often infeasible in practice due to fixed or protected user inputs. This narrow focus overlooks a more realistic and stealthy vector: instructional prompts, which are widely reused, publicly shared, and rarely audited. Their implicit trust makes them a compelling target for adversaries to manipulate RAG behavior covertly.   We introduce a novel attack for Adversarial Instructional Prompt (AIP) that exploits adversarial instructional prompts to manipulate RAG outputs by subtly altering retrieval behavior. By shifting the attack surface to the instructional prompts, AIP reveals how trusted yet seemingly benign interface components can be weaponized to degrade system integrity. The attack is crafted to achieve three goals: (1) naturalness, to evade user detection; (2) utility, to encourage use of prompts; and (3) robustness, to remain effective across diverse query variations. We propose a diverse query generation strategy that simulates realistic linguistic variation in user queries, enabling the discovery of prompts that generalize across paraphrases and rephrasings. Building on this, a genetic algorithm-based joint optimization is developed to evolve adversarial prompts by balancing attack success, clean-task utility, and stealthiness. Experimental results show that AIP achieves up to 95.23% ASR while preserving benign functionality. These findings uncover a critical and previously overlooked vulnerability in RAG systems, emphasizing the need to reassess the shared instructional prompts.



## **36. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

cs.CV

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2409.13174v3) [paper-pdf](http://arxiv.org/pdf/2409.13174v3)

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Chengyuan Yu, Mengshu Sun, Qiang Zhang, Yijie Guo, Kaidi Xu, Jize Zhang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.



## **37. Sentinel Agents for Secure and Trustworthy Agentic AI in Multi-Agent Systems**

cs.AI

25 pages, 12 figures

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14956v1) [paper-pdf](http://arxiv.org/pdf/2509.14956v1)

**Authors**: Diego Gosmar, Deborah A. Dahl

**Abstract**: This paper proposes a novel architectural framework aimed at enhancing security and reliability in multi-agent systems (MAS). A central component of this framework is a network of Sentinel Agents, functioning as a distributed security layer that integrates techniques such as semantic analysis via large language models (LLMs), behavioral analytics, retrieval-augmented verification, and cross-agent anomaly detection. Such agents can potentially oversee inter-agent communications, identify potential threats, enforce privacy and access controls, and maintain comprehensive audit records. Complementary to the idea of Sentinel Agents is the use of a Coordinator Agent. The Coordinator Agent supervises policy implementation, and manages agent participation. In addition, the Coordinator also ingests alerts from Sentinel Agents. Based on these alerts, it can adapt policies, isolate or quarantine misbehaving agents, and contain threats to maintain the integrity of the MAS ecosystem. This dual-layered security approach, combining the continuous monitoring of Sentinel Agents with the governance functions of Coordinator Agents, supports dynamic and adaptive defense mechanisms against a range of threats, including prompt injection, collusive agent behavior, hallucinations generated by LLMs, privacy breaches, and coordinated multi-agent attacks. In addition to the architectural design, we present a simulation study where 162 synthetic attacks of different families (prompt injection, hallucination, and data exfiltration) were injected into a multi-agent conversational environment. The Sentinel Agents successfully detected the attack attempts, confirming the practical feasibility of the proposed monitoring approach. The framework also offers enhanced system observability, supports regulatory compliance, and enables policy evolution over time.



## **38. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2312.03853v6) [paper-pdf](http://arxiv.org/pdf/2312.03853v6)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Large Language Models (LLMs) are being integrated into applications such as chatbots or email assistants. To prevent improper responses, safety mechanisms, such as Reinforcement Learning from Human Feedback (RLHF), are implemented in them. In this work, we bypass these safety measures for ChatGPT, Gemini, and Deepseek by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. First, we create elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are provided, making it possible to obtain unauthorized, illegal, or harmful information when querying ChatGPT, Gemini, and Deepseek. We show that these chatbots are vulnerable to this attack by getting dangerous information for 40 out of 40 illicit questions in GPT-4.1-mini, Gemini-1.5-flash, 39 out of 40 in GPT-4o-mini, 38 out of 40 in GPT-3.5-turbo, and 2 out of 2 cases in Gemini-2.5-flash and DeepSeek V3. The attack can be carried out manually or automatically using a support LLM, and has proven effective against models deployed between 2023 and 2025.



## **39. MUSE: MCTS-Driven Red Teaming Framework for Enhanced Multi-Turn Dialogue Safety in Large Language Models**

cs.CL

EMNLP 2025 main conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14651v1) [paper-pdf](http://arxiv.org/pdf/2509.14651v1)

**Authors**: Siyu Yan, Long Zeng, Xuecheng Wu, Chengcheng Han, Kongcheng Zhang, Chong Peng, Xuezhi Cao, Xunliang Cai, Chenjuan Guo

**Abstract**: As large language models~(LLMs) become widely adopted, ensuring their alignment with human values is crucial to prevent jailbreaks where adversaries manipulate models to produce harmful content. While most defenses target single-turn attacks, real-world usage often involves multi-turn dialogues, exposing models to attacks that exploit conversational context to bypass safety measures. We introduce MUSE, a comprehensive framework tackling multi-turn jailbreaks from both attack and defense angles. For attacks, we propose MUSE-A, a method that uses frame semantics and heuristic tree search to explore diverse semantic trajectories. For defense, we present MUSE-D, a fine-grained safety alignment approach that intervenes early in dialogues to reduce vulnerabilities. Extensive experiments on various models show that MUSE effectively identifies and mitigates multi-turn vulnerabilities. Code is available at \href{https://github.com/yansiyu02/MUSE}{https://github.com/yansiyu02/MUSE}.



## **40. Enterprise AI Must Enforce Participant-Aware Access Control**

cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14608v1) [paper-pdf](http://arxiv.org/pdf/2509.14608v1)

**Authors**: Shashank Shreedhar Bhatt, Tanmay Rajore, Khushboo Aggarwal, Ganesh Ananthanarayanan, Ranveer Chandra, Nishanth Chandran, Suyash Choudhury, Divya Gupta, Emre Kiciman, Sumit Kumar Pandey, Srinath Setty, Rahul Sharma, Teijia Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in enterprise settings where they interact with multiple users and are trained or fine-tuned on sensitive internal data. While fine-tuning enhances performance by internalizing domain knowledge, it also introduces a critical security risk: leakage of confidential training data to unauthorized users. These risks are exacerbated when LLMs are combined with Retrieval-Augmented Generation (RAG) pipelines that dynamically fetch contextual documents at inference time.   We demonstrate data exfiltration attacks on AI assistants where adversaries can exploit current fine-tuning and RAG architectures to leak sensitive information by leveraging the lack of access control enforcement. We show that existing defenses, including prompt sanitization, output filtering, system isolation, and training-level privacy mechanisms, are fundamentally probabilistic and fail to offer robust protection against such attacks.   We take the position that only a deterministic and rigorous enforcement of fine-grained access control during both fine-tuning and RAG-based inference can reliably prevent the leakage of sensitive data to unauthorized recipients.   We introduce a framework centered on the principle that any content used in training, retrieval, or generation by an LLM is explicitly authorized for \emph{all users involved in the interaction}. Our approach offers a simple yet powerful paradigm shift for building secure multi-user LLM systems that are grounded in classical access control but adapted to the unique challenges of modern AI workflows. Our solution has been deployed in Microsoft Copilot Tuning, a product offering that enables organizations to fine-tune models using their own enterprise-specific data.



## **41. Reconstruction of Differentially Private Text Sanitization via Large Language Models**

cs.CR

RAID-2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2410.12443v3) [paper-pdf](http://arxiv.org/pdf/2410.12443v3)

**Authors**: Shuchao Pang, Zhigang Lu, Haichen Wang, Peng Fu, Yongbin Zhou, Minhui Xue

**Abstract**: Differential privacy (DP) is the de facto privacy standard against privacy leakage attacks, including many recently discovered ones against large language models (LLMs). However, we discovered that LLMs could reconstruct the altered/removed privacy from given DP-sanitized prompts. We propose two attacks (black-box and white-box) based on the accessibility to LLMs and show that LLMs could connect the pair of DP-sanitized text and the corresponding private training data of LLMs by giving sample text pairs as instructions (in the black-box attacks) or fine-tuning data (in the white-box attacks). To illustrate our findings, we conduct comprehensive experiments on modern LLMs (e.g., LLaMA-2, LLaMA-3, ChatGPT-3.5, ChatGPT-4, ChatGPT-4o, Claude-3, Claude-3.5, OPT, GPT-Neo, GPT-J, Gemma-2, and Pythia) using commonly used datasets (such as WikiMIA, Pile-CC, and Pile-Wiki) against both word-level and sentence-level DP. The experimental results show promising recovery rates, e.g., the black-box attacks against the word-level DP over WikiMIA dataset gave 72.18% on LLaMA-2 (70B), 82.39% on LLaMA-3 (70B), 75.35% on Gemma-2, 91.2% on ChatGPT-4o, and 94.01% on Claude-3.5 (Sonnet). More urgently, this study indicates that these well-known LLMs have emerged as a new security risk for existing DP text sanitization approaches in the current environment.



## **42. SynBench: A Benchmark for Differentially Private Text Generation**

cs.AI

15 pages

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14594v1) [paper-pdf](http://arxiv.org/pdf/2509.14594v1)

**Authors**: Yidan Sun, Viktor Schlegel, Srinivasan Nandakumar, Iqra Zahid, Yuping Wu, Yulong Wu, Hao Li, Jie Zhang, Warren Del-Pinto, Goran Nenadic, Siew Kei Lam, Anil Anthony Bharath

**Abstract**: Data-driven decision support in high-stakes domains like healthcare and finance faces significant barriers to data sharing due to regulatory, institutional, and privacy concerns. While recent generative AI models, such as large language models, have shown impressive performance in open-domain tasks, their adoption in sensitive environments remains limited by unpredictable behaviors and insufficient privacy-preserving datasets for benchmarking. Existing anonymization methods are often inadequate, especially for unstructured text, as redaction and masking can still allow re-identification. Differential Privacy (DP) offers a principled alternative, enabling the generation of synthetic data with formal privacy assurances. In this work, we address these challenges through three key contributions. First, we introduce a comprehensive evaluation framework with standardized utility and fidelity metrics, encompassing nine curated datasets that capture domain-specific complexities such as technical jargon, long-context dependencies, and specialized document structures. Second, we conduct a large-scale empirical study benchmarking state-of-the-art DP text generation methods and LLMs of varying sizes and different fine-tuning strategies, revealing that high-quality domain-specific synthetic data generation under DP constraints remains an unsolved challenge, with performance degrading as domain complexity increases. Third, we develop a membership inference attack (MIA) methodology tailored for synthetic text, providing first empirical evidence that the use of public datasets - potentially present in pre-training corpora - can invalidate claimed privacy guarantees. Our findings underscore the urgent need for rigorous privacy auditing and highlight persistent gaps between open-domain and specialist evaluations, informing responsible deployment of generative AI in privacy-sensitive, high-stakes settings.



## **43. Unique Security and Privacy Threats of Large Language Models: A Comprehensive Survey**

cs.CR

35 pages, 9 tables, 12 figures. To appear in ACM Computing Surveys  (CSUR), 2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2406.07973v3) [paper-pdf](http://arxiv.org/pdf/2406.07973v3)

**Authors**: Shang Wang, Tianqing Zhu, Bo Liu, Ming Ding, Dayong Ye, Wanlei Zhou, Philip S. Yu

**Abstract**: With the rapid development of artificial intelligence, large language models (LLMs) have made remarkable advancements in natural language processing. These models are trained on vast datasets to exhibit powerful language understanding and generation capabilities across various applications, including chatbots, and agents. However, LLMs have revealed a variety of privacy and security issues throughout their life cycle, drawing significant academic and industrial attention. Moreover, the risks faced by LLMs differ significantly from those encountered by traditional language models. Given that current surveys lack a clear taxonomy of unique threat models across diverse scenarios, we emphasize the unique privacy and security threats associated with four specific scenarios: pre-training, fine-tuning, deployment, and LLM-based agents. Addressing the characteristics of each risk, this survey outlines and analyzes potential countermeasures. Research on attack and defense situations can offer feasible research directions, enabling more areas to benefit from LLMs.



## **44. LLM Jailbreak Detection for (Almost) Free!**

cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14558v1) [paper-pdf](http://arxiv.org/pdf/2509.14558v1)

**Authors**: Guorui Chen, Yifan Xia, Xiaojun Jia, Zhijiang Li, Philip Torr, Jindong Gu

**Abstract**: Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference.



## **45. GRADA: Graph-based Reranking against Adversarial Documents Attack**

cs.IR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2505.07546v3) [paper-pdf](http://arxiv.org/pdf/2505.07546v3)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.



## **46. Benchmarking Large Language Models for Cryptanalysis and Side-Channel Vulnerabilities**

cs.CL

EMNLP'25 Findings

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2505.24621v2) [paper-pdf](http://arxiv.org/pdf/2505.24621v2)

**Authors**: Utsav Maskey, Chencheng Zhu, Usman Naseem

**Abstract**: Recent advancements in large language models (LLMs) have transformed natural language understanding and generation, leading to extensive benchmarking across diverse tasks. However, cryptanalysis - a critical area for data security and its connection to LLMs' generalization abilities - remains underexplored in LLM evaluations. To address this gap, we evaluate the cryptanalytic potential of state-of-the-art LLMs on ciphertexts produced by a range of cryptographic algorithms. We introduce a benchmark dataset of diverse plaintexts, spanning multiple domains, lengths, writing styles, and topics, paired with their encrypted versions. Using zero-shot and few-shot settings along with chain-of-thought prompting, we assess LLMs' decryption success rate and discuss their comprehension abilities. Our findings reveal key insights into LLMs' strengths and limitations in side-channel scenarios and raise concerns about their susceptibility to under-generalization-related attacks. This research highlights the dual-use nature of LLMs in security contexts and contributes to the ongoing discussion on AI safety and security.



## **47. Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs**

cs.SE

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2411.18216v2) [paper-pdf](http://arxiv.org/pdf/2411.18216v2)

**Authors**: Samuele Pasini, Jinhan Kim, Tommaso Aiello, Rocio Cabrera Lozoya, Antonino Sabetta, Paolo Tonella

**Abstract**: Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. A key challenge is ensuring the LLMs have enough knowledge to address specific security requirements, such as information about existing attacks. For this, we propose an approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired by the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance while employing RAG and Self-Ranking, with an increase of up to 71%pt (on average 37%pt) and up to 43%pt (on average 6%pt) in the F2-Score for XSS and SQLi detection, respectively.



## **48. CyberLLMInstruct: A Pseudo-malicious Dataset Revealing Safety-performance Trade-offs in Cyber Security LLM Fine-tuning**

cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2503.09334v3) [paper-pdf](http://arxiv.org/pdf/2503.09334v3)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents both opportunities and critical safety risks. We introduce CyberLLMInstruct, a dataset of 54,928 pseudo-malicious instruction-response pairs spanning cyber security tasks including malware analysis, phishing simulations, and zero-day vulnerabilities. Our comprehensive evaluation using seven open-source LLMs reveals a critical trade-off: while fine-tuning improves cyber security task performance (achieving up to 92.50% accuracy on CyberMetric), it severely compromises safety resilience across all tested models and attack vectors (e.g., Llama 3.1 8B's security score against prompt injection drops from 0.95 to 0.15). The dataset incorporates diverse sources including CTF challenges, academic papers, industry reports, and CVE databases to ensure comprehensive coverage of cyber security domains. Our findings highlight the unique challenges of securing LLMs in adversarial domains and establish the critical need for developing fine-tuning methodologies that balance performance gains with safety preservation in security-sensitive domains.



## **49. Do LLMs Align Human Values Regarding Social Biases? Judging and Explaining Social Biases with LLMs**

cs.CL

38 pages, 31 figures

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.13869v1) [paper-pdf](http://arxiv.org/pdf/2509.13869v1)

**Authors**: Yang Liu, Chenhui Chu

**Abstract**: Large language models (LLMs) can lead to undesired consequences when misaligned with human values, especially in scenarios involving complex and sensitive social biases. Previous studies have revealed the misalignment of LLMs with human values using expert-designed or agent-based emulated bias scenarios. However, it remains unclear whether the alignment of LLMs with human values differs across different types of scenarios (e.g., scenarios containing negative vs. non-negative questions). In this study, we investigate the alignment of LLMs with human values regarding social biases (HVSB) in different types of bias scenarios. Through extensive analysis of 12 LLMs from four model families and four datasets, we demonstrate that LLMs with large model parameter scales do not necessarily have lower misalignment rate and attack success rate. Moreover, LLMs show a certain degree of alignment preference for specific types of scenarios and the LLMs from the same model family tend to have higher judgment consistency. In addition, we study the understanding capacity of LLMs with their explanations of HVSB. We find no significant differences in the understanding of HVSB across LLMs. We also find LLMs prefer their own generated explanations. Additionally, we endow smaller language models (LMs) with the ability to explain HVSB. The generation results show that the explanations generated by the fine-tuned smaller LMs are more readable, but have a relatively lower model agreeability.



## **50. Defending against Indirect Prompt Injection by Instruction Detection**

cs.CR

16 pages, 4 figures

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2505.06311v2) [paper-pdf](http://arxiv.org/pdf/2505.06311v2)

**Authors**: Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, Fangzhao Wu

**Abstract**: The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. We recognize that IPI attacks fundamentally rely on the presence of instructions embedded within external content, which can alter the behavioral states of LLMs. Can the effective detection of such state changes help us defend against IPI attacks? In this paper, we propose InstructDetector, a novel detection-based approach that leverages the behavioral states of LLMs to identify potential IPI attacks. Specifically, we demonstrate the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. By effectively combining these features, InstructDetector achieves a detection accuracy of 99.60% in the in-domain setting and 96.90% in the out-of-domain setting, and reduces the attack success rate to just 0.03% on the BIPIA benchmark. The code is publicly available at https://github.com/MYVAE/Instruction-detection.



