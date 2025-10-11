# Latest Large Language Model Attack Papers
**update at 2025-10-11 09:49:51**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. AutoRed: A Free-form Adversarial Prompt Generation Framework for Automated Red Teaming**

cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08329v1) [paper-pdf](http://arxiv.org/pdf/2510.08329v1)

**Authors**: Muxi Diao, Yutao Mou, Keqing He, Hanbo Song, Lulu Zhao, Shikun Zhang, Wei Ye, Kongming Liang, Zhanyu Ma

**Abstract**: The safety of Large Language Models (LLMs) is crucial for the development of trustworthy AI applications. Existing red teaming methods often rely on seed instructions, which limits the semantic diversity of the synthesized adversarial prompts. We propose AutoRed, a free-form adversarial prompt generation framework that removes the need for seed instructions. AutoRed operates in two stages: (1) persona-guided adversarial instruction generation, and (2) a reflection loop to iteratively refine low-quality prompts. To improve efficiency, we introduce a verifier to assess prompt harmfulness without querying the target models. Using AutoRed, we build two red teaming datasets -- AutoRed-Medium and AutoRed-Hard -- and evaluate eight state-of-the-art LLMs. AutoRed achieves higher attack success rates and better generalization than existing baselines. Our results highlight the limitations of seed-based approaches and demonstrate the potential of free-form red teaming for LLM safety evaluation. We will open source our datasets in the near future.



## **2. MCPSecBench: A Systematic Security Benchmark and Playground for Testing Model Context Protocols**

cs.CR

This is a technical report from Lingnan University, Hong Kong. Code  is available at https://github.com/AIS2Lab/MCPSecBench

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2508.13220v2) [paper-pdf](http://arxiv.org/pdf/2508.13220v2)

**Authors**: Yixuan Yang, Daoyuan Wu, Yufan Chen

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications via the Model Context Protocol (MCP), a universal, open standard for connecting AI agents with data sources and external tools. While MCP enhances the capabilities of LLM-based agents, it also introduces new security risks and expands their attack surfaces. In this paper, we present the first systematic taxonomy of MCP security, identifying 17 attack types across 4 primary attack surfaces. We introduce MCPSecBench, a comprehensive security benchmark and playground that integrates prompt datasets, MCP servers, MCP clients, attack scripts, and protection mechanisms to evaluate these attacks across three major MCP providers. Our benchmark is modular and extensible, allowing researchers to incorporate custom implementations of clients, servers, and transport protocols for systematic security assessment. Experimental results show that over 85% of the identified attacks successfully compromise at least one platform, with core vulnerabilities universally affecting Claude, OpenAI, and Cursor, while prompt-based and tool-centric attacks exhibit considerable variability across different hosts and models. In addition, current protection mechanisms have little effect against these attacks. Overall, MCPSecBench standardizes the evaluation of MCP security and enables rigorous testing across all MCP layers.



## **3. Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning**

cs.LG

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2505.16567v3) [paper-pdf](http://arxiv.org/pdf/2505.16567v3)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning open-weight Large Language Models (LLMs) is standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets leads to predictable behaviors. In this paper, we demonstrate, for the first time, that an adversary can create compromised LLMs that are performant and benign, yet exhibit adversarial behaviors once finetuned by downstream users. To this end, we propose an attack, FAB (Finetuning-activated Adversarial Behaviors), which compromises an LLM via meta-learning techniques that simulate downstream finetuning, explicitly optimizing for the emergence of adversarial behaviors in the finetuned models. At the same time, the compromised LLM is regularized to retain general capabilities and to exhibit no adversarial behaviors prior to finetuning. As a result, when users finetune (e.g., instruction-tuning, distillation, DPO) the seemingly benign model on their own datasets, they unknowingly trigger its dormant adversarial behavior. We experimentally demonstrate the effectiveness of FAB across multiple LLMs and three commonly considered target behaviors: unsolicited advertising, jailbreakability, and over-refusal. We show that FAB-triggers are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler, post-training algorithm). Our findings challenge prevailing assumptions on the security of finetuning, revealing a critical attack vector.



## **4. Chain-of-Trigger: An Agentic Backdoor that Paradoxically Enhances Agentic Robustness**

cs.AI

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08238v1) [paper-pdf](http://arxiv.org/pdf/2510.08238v1)

**Authors**: Jiyang Qiu, Xinbei Ma, Yunqing Xu, Zhuosheng Zhang, Hai Zhao

**Abstract**: The rapid deployment of large language model (LLM)-based agents in real-world applications has raised serious concerns about their trustworthiness. In this work, we reveal the security and robustness vulnerabilities of these agents through backdoor attacks. Distinct from traditional backdoors limited to single-step control, we propose the Chain-of-Trigger Backdoor (CoTri), a multi-step backdoor attack designed for long-horizon agentic control. CoTri relies on an ordered sequence. It starts with an initial trigger, and subsequent ones are drawn from the environment, allowing multi-step manipulation that diverts the agent from its intended task. Experimental results show that CoTri achieves a near-perfect attack success rate (ASR) while maintaining a near-zero false trigger rate (FTR). Due to training data modeling the stochastic nature of the environment, the implantation of CoTri paradoxically enhances the agent's performance on benign tasks and even improves its robustness against environmental distractions. We further validate CoTri on vision-language models (VLMs), confirming its scalability to multimodal agents. Our work highlights that CoTri achieves stable, multi-step control within agents, improving their inherent robustness and task capabilities, which ultimately makes the attack more stealthy and raises potential safty risks.



## **5. Defending MoE LLMs against Harmful Fine-Tuning via Safety Routing Alignment**

cs.CR

Under review

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2509.22745v2) [paper-pdf](http://arxiv.org/pdf/2509.22745v2)

**Authors**: Jaehan Kim, Minkyoo Song, Seungwon Shin, Sooel Son

**Abstract**: Recent large language models (LLMs) have increasingly adopted the Mixture-of-Experts (MoE) architecture for efficiency. MoE-based LLMs heavily depend on a superficial safety mechanism in which harmful inputs are routed safety-critical experts. However, our analysis reveals that routing decisions for harmful inputs drift significantly after fine-tuning, exposing a critical vulnerability to harmful fine-tuning (HFT) attacks. Existing defenses, primarily designed for monolithic LLMs, are less effective for MoE LLMs as they fail to prevent drift in harmful input routing. To address this limitation, we propose SafeMoE, a safe fine-tuning method tailored to MoE LLMs. SafeMoE directly mitigates routing drift by penalizing the gap between the routing weights of a fine-tuned model and those of the initial safety-aligned model, thereby preserving the safety-aligned routing of harmful inputs to safety-critical experts. Experiments on open-source MoE LLMs ranging from 7B to 141B parameters demonstrate that SafeMoE effectively mitigates HFT attacks, reducing the harmfulness score of OLMoE from 62.0 to 5.0, for example, while maintaining task utility within 1% degradation and incurring only 2% overhead. It significantly outperforms state-of-the-art defense methods for safeguarding LLM fine-tuning and remains effective in recent large-scale MoE LLMs such as gpt-oss and Llama 4. Our implementation is available at https://anonymous.4open.science/r/SafeMoE.



## **6. Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs**

cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2507.11112v2) [paper-pdf](http://arxiv.org/pdf/2507.11112v2)

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.



## **7. Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks**

cs.CL

Minor correction: Fixed sign errors in the results table. The update  does not affect the main findings or conclusions

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2506.11113v3) [paper-pdf](http://arxiv.org/pdf/2506.11113v3)

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication.



## **8. DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm**

cs.CL

NeurIPS 2025 Spotlight

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2509.15550v2) [paper-pdf](http://arxiv.org/pdf/2509.15550v2)

**Authors**: Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, Yanan Cao

**Abstract**: The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce DNA-DetectLLM, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of 5.55% in AUROC and 2.08% in F1 score across multiple public benchmark datasets. Code and data are available at https://github.com/Xiaoweizhu57/DNA-DetectLLM.



## **9. (Token-Level) InfoRMIA: Stronger Membership Inference and Memorization Assessment for LLMs**

cs.LG

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.05582v2) [paper-pdf](http://arxiv.org/pdf/2510.05582v2)

**Authors**: Jiashu Tao, Reza Shokri

**Abstract**: Machine learning models are known to leak sensitive information, as they inevitably memorize (parts of) their training data. More alarmingly, large language models (LLMs) are now trained on nearly all available data, which amplifies the magnitude of information leakage and raises serious privacy risks. Hence, it is more crucial than ever to quantify privacy risk before the release of LLMs. The standard method to quantify privacy is via membership inference attacks, where the state-of-the-art approach is the Robust Membership Inference Attack (RMIA). In this paper, we present InfoRMIA, a principled information-theoretic formulation of membership inference. Our method consistently outperforms RMIA across benchmarks while also offering improved computational efficiency.   In the second part of the paper, we identify the limitations of treating sequence-level membership inference as the gold standard for measuring leakage. We propose a new perspective for studying membership and memorization in LLMs: token-level signals and analyses. We show that a simple token-based InfoRMIA can pinpoint which tokens are memorized within generated outputs, thereby localizing leakage from the sequence level down to individual tokens, while achieving stronger sequence-level inference power on LLMs. This new scope rethinks privacy in LLMs and can lead to more targeted mitigation, such as exact unlearning.



## **10. Fewer Weights, More Problems: A Practical Attack on LLM Pruning**

cs.LG

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07985v1) [paper-pdf](http://arxiv.org/pdf/2510.07985v1)

**Authors**: Kazuki Egashira, Robin Staab, Thibaud Gloaguen, Mark Vero, Martin Vechev

**Abstract**: Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to $95.7\%$ for jailbreak, $98.7\%$ for benign instruction refusal, and $99.5\%$ for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression.



## **11. Fine-Tuning Jailbreaks under Highly Constrained Black-Box Settings: A Three-Pronged Approach**

cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.01342v2) [paper-pdf](http://arxiv.org/pdf/2510.01342v2)

**Authors**: Xiangfang Li, Yu Wang, Bo Li

**Abstract**: With the rapid advancement of large language models (LLMs), ensuring their safe use becomes increasingly critical. Fine-tuning is a widely used method for adapting models to downstream tasks, yet it is vulnerable to jailbreak attacks. However, most existing studies focus on overly simplified attack scenarios, limiting their practical relevance to real-world defense settings. To make this risk concrete, we present a three-pronged jailbreak attack and evaluate it against provider defenses under a dataset-only black-box fine-tuning interface. In this setting, the attacker can only submit fine-tuning data to the provider, while the provider may deploy defenses across stages: (1) pre-upload data filtering, (2) training-time defensive fine-tuning, and (3) post-training safety audit. Our attack combines safety-styled prefix/suffix wrappers, benign lexical encodings (underscoring) of sensitive tokens, and a backdoor mechanism, enabling the model to learn harmful behaviors while individual datapoints appear innocuous. Extensive experiments demonstrate the effectiveness of our approach. In real-world deployment, our method successfully jailbreaks GPT-4.1 and GPT-4o on the OpenAI platform with attack success rates above 97% for both models. Our code is available at https://github.com/lxf728/tri-pronged-ft-attack.



## **12. Rule Encoding and Compliance in Large Language Models: An Information-Theoretic Analysis**

cs.AI

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.05106v2) [paper-pdf](http://arxiv.org/pdf/2510.05106v2)

**Authors**: Joachim Diederich

**Abstract**: The design of safety-critical agents based on large language models (LLMs) requires more than simple prompt engineering. This paper presents a comprehensive information-theoretic analysis of how rule encodings in system prompts influence attention mechanisms and compliance behaviour. We demonstrate that rule formats with low syntactic entropy and highly concentrated anchors reduce attention entropy and improve pointer fidelity, but reveal a fundamental trade-off between anchor redundancy and attention entropy that previous work failed to recognize. Through formal analysis of multiple attention architectures including causal, bidirectional, local sparse, kernelized, and cross-attention mechanisms, we establish bounds on pointer fidelity and show how anchor placement strategies must account for competing fidelity and entropy objectives. Combining these insights with a dynamic rule verification architecture, we provide a formal proof that hot reloading of verified rule sets increases the asymptotic probability of compliant outputs. These findings underscore the necessity of principled anchor design and dual enforcement mechanisms to protect LLM-based agents against prompt injection attacks while maintaining compliance in evolving domains.



## **13. PiCo: Jailbreaking Multimodal Large Language Models via Pictorial Code Contextualization**

cs.CR

Accepted to IEEE International Conference on Multimedia and Expo  (ICME) 2025

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2504.01444v4) [paper-pdf](http://arxiv.org/pdf/2504.01444v4)

**Authors**: Aofan Liu, Lulu Tang, Ting Pan, Yuguo Yin, Bin Wang, Ao Yang

**Abstract**: Multimodal Large Language Models (MLLMs), which integrate vision and other modalities into Large Language Models (LLMs), significantly enhance AI capabilities but also introduce new security vulnerabilities. By exploiting the vulnerabilities of the visual modality and the long-tail distribution characteristic of code training data, we present PiCo, a novel jailbreaking framework designed to progressively bypass multi-tiered defense mechanisms in advanced MLLMs. PiCo employs a tier-by-tier jailbreak strategy, using token-level typographic attacks to evade input filtering and embedding harmful intent within programming context instructions to bypass runtime monitoring. To comprehensively assess the impact of attacks, a new evaluation metric is further proposed to assess both the toxicity and helpfulness of model outputs post-attack. By embedding harmful intent within code-style visual instructions, PiCo achieves an average Attack Success Rate (ASR) of 84.13% on Gemini-Pro Vision and 52.66% on GPT-4, surpassing previous methods. Experimental results highlight the critical gaps in current defenses, underscoring the need for more robust strategies to secure advanced MLLMs.



## **14. Logic Jailbreak: Efficiently Unlocking LLM Safety Restrictions Through Formal Logical Expression**

cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2505.13527v2) [paper-pdf](http://arxiv.org/pdf/2505.13527v2)

**Authors**: Jingyu Peng, Maolin Wang, Nan Wang, Jiatong Li, Yuchen Li, Yuyang Ye, Wanyu Wang, Pengyue Jia, Kai Zhang, Xiangyu Zhao

**Abstract**: Despite substantial advancements in aligning large language models (LLMs) with human values, current safety mechanisms remain susceptible to jailbreak attacks. We hypothesize that this vulnerability stems from distributional discrepancies between alignment-oriented prompts and malicious prompts. To investigate this, we introduce LogiBreak, a novel and universal black-box jailbreak method that leverages logical expression translation to circumvent LLM safety systems. By converting harmful natural language prompts into formal logical expressions, LogiBreak exploits the distributional gap between alignment data and logic-based inputs, preserving the underlying semantic intent and readability while evading safety constraints. We evaluate LogiBreak on a multilingual jailbreak dataset spanning three languages, demonstrating its effectiveness across various evaluation settings and linguistic contexts.



## **15. MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation**

cs.LG

Accepted By NeurIPS 2025

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07835v1) [paper-pdf](http://arxiv.org/pdf/2510.07835v1)

**Authors**: Weisen Jiang, Sinno Jialin Pan

**Abstract**: This paper introduces MetaDefense, a novel framework for defending against finetuning-based jailbreak attacks in large language models (LLMs). We observe that existing defense mechanisms fail to generalize to harmful queries disguised by unseen attack templates, despite LLMs being capable of distinguishing disguised harmful queries in the embedding space. Based on these insights, we propose a two-stage defense approach: (i) pre-generation defense that detects harmful queries before response generation begins, and (ii) mid-generation defense that monitors partial responses during generation to prevent outputting more harmful content. Our MetaDefense trains the LLM to predict the harmfulness of both queries and partial responses using specialized prompts, enabling early termination of potentially harmful interactions. Extensive experiments across multiple LLM architectures (LLaMA-2-7B, Qwen-2.5-3B-Instruct, and LLaMA-3.2-3B-Instruct) demonstrate that MetaDefense significantly outperforms existing defense mechanisms, achieving robust defense against harmful queries with seen and unseen attack templates while maintaining competitive performance on benign tasks. Code is available at https://github.com/ws-jiang/MetaDefense.



## **16. Effective and Stealthy One-Shot Jailbreaks on Deployed Mobile Vision-Language Agents**

cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07809v1) [paper-pdf](http://arxiv.org/pdf/2510.07809v1)

**Authors**: Renhua Ding, Xiao Yang, Zhengwei Fang, Jun Luo, Kun He, Jun Zhu

**Abstract**: Large vision-language models (LVLMs) enable autonomous mobile agents to operate smartphone user interfaces, yet vulnerabilities to UI-level attacks remain critically understudied. Existing research often depends on conspicuous UI overlays, elevated permissions, or impractical threat models, limiting stealth and real-world applicability. In this paper, we present a practical and stealthy one-shot jailbreak attack that leverages in-app prompt injections: malicious applications embed short prompts in UI text that remain inert during human interaction but are revealed when an agent drives the UI via ADB (Android Debug Bridge). Our framework comprises three crucial components: (1) low-privilege perception-chain targeting, which injects payloads into malicious apps as the agent's visual inputs; (2) stealthy user-invisible activation, a touch-based trigger that discriminates agent from human touches using physical touch attributes and exposes the payload only during agent operation; and (3) one-shot prompt efficacy, a heuristic-guided, character-level iterative-deepening search algorithm (HG-IDA*) that performs one-shot, keyword-level detoxification to evade on-device safety filters. We evaluate across multiple LVLM backends, including closed-source services and representative open-source models within three Android applications, and we observe high planning and execution hijack rates in single-shot scenarios (e.g., GPT-4o: 82.5% planning / 75.0% execution). These findings expose a fundamental security vulnerability in current mobile agents with immediate implications for autonomous smartphone operation.



## **17. AEGIS : Automated Co-Evolutionary Framework for Guarding Prompt Injections Schema**

cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2509.00088v2) [paper-pdf](http://arxiv.org/pdf/2509.00088v2)

**Authors**: Ting-Chun Liu, Ching-Yu Hsu, Kuan-Yi Lee, Chi-An Fu, Hung-yi Lee

**Abstract**: Prompt injection attacks pose a significant challenge to the safe deployment of Large Language Models (LLMs) in real-world applications. While prompt-based detection offers a lightweight and interpretable defense strategy, its effectiveness has been hindered by the need for manual prompt engineering. To address this issue, we propose AEGIS , an Automated co-Evolutionary framework for Guarding prompt Injections Schema. Both attack and defense prompts are iteratively optimized against each other using a gradient-like natural language prompt optimization technique. This framework enables both attackers and defenders to autonomously evolve via a Textual Gradient Optimization (TGO) module, leveraging feedback from an LLM-guided evaluation loop. We evaluate our system on a real-world assignment grading dataset of prompt injection attacks and demonstrate that our method consistently outperforms existing baselines, achieving superior robustness in both attack success and detection. Specifically, the attack success rate (ASR) reaches 1.0, representing an improvement of 0.26 over the baseline. For detection, the true positive rate (TPR) improves by 0.23 compared to the previous best work, reaching 0.84, and the true negative rate (TNR) remains comparable at 0.89. Ablation studies confirm the importance of co-evolution, gradient buffering, and multi-objective optimization. We also confirm that this framework is effective in different LLMs. Our results highlight the promise of adversarial training as a scalable and effective approach for guarding prompt injections.



## **18. Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs**

cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07697v1) [paper-pdf](http://arxiv.org/pdf/2510.07697v1)

**Authors**: Man Hu, Xinyi Wu, Zuofeng Suo, Jinbo Feng, Linghui Meng, Yanhao Jia, Anh Tuan Luu, Shuai Zhao

**Abstract**: With the rise of advanced reasoning capabilities, large language models (LLMs) are receiving increasing attention. However, although reasoning improves LLMs' performance on downstream tasks, it also introduces new security risks, as adversaries can exploit these capabilities to conduct backdoor attacks. Existing surveys on backdoor attacks and reasoning security offer comprehensive overviews but lack in-depth analysis of backdoor attacks and defenses targeting LLMs' reasoning abilities. In this paper, we take the first step toward providing a comprehensive review of reasoning-based backdoor attacks in LLMs by analyzing their underlying mechanisms, methodological frameworks, and unresolved challenges. Specifically, we introduce a new taxonomy that offers a unified perspective for summarizing existing approaches, categorizing reasoning-based backdoor attacks into associative, passive, and active. We also present defense strategies against such attacks and discuss current challenges alongside potential directions for future research. This work offers a novel perspective, paving the way for further exploration of secure and trustworthy LLM communities.



## **19. LLM Unlearning Under the Microscope: A Full-Stack View on Methods and Metrics**

cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07626v1) [paper-pdf](http://arxiv.org/pdf/2510.07626v1)

**Authors**: Chongyu Fan, Changsheng Wang, Yancheng Huang, Soumyadeep Pal, Sijia Liu

**Abstract**: Machine unlearning for large language models (LLMs) aims to remove undesired data, knowledge, and behaviors (e.g., for safety, privacy, or copyright) while preserving useful model capabilities. Despite rapid progress over the past two years, research in LLM unlearning remains fragmented, with limited clarity on what constitutes effective unlearning and how it should be rigorously evaluated. In this work, we present a principled taxonomy of twelve recent stateful unlearning methods, grouped into three methodological families: divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning. Building on this taxonomy, we revisit the evaluation of unlearning effectiveness (UE), utility retention (UT), and robustness (Rob), focusing on the WMDP benchmark. Our analysis shows that current evaluations, dominated by multiple-choice question (MCQ) accuracy, offer only a narrow perspective, often overstating success while overlooking the model's actual generation behavior. To address this gap, we introduce open question-answering (Open-QA) metrics that better capture generative performance and reveal the inherent UE-UT tradeoff across method families. Furthermore, we demonstrate that robustness requires finer-grained analysis: for example, vulnerabilities differ substantially between in-domain relearning and out-of-domain fine-tuning, even though both fall under model-level attacks. Through this study, we hope to deliver a full-stack revisit of LLM unlearning and actionable guidance for designing and evaluating future methods.



## **20. $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks**

cs.MA

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2504.00218v2) [paper-pdf](http://arxiv.org/pdf/2504.00218v2)

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Fleming, Tianlong Chen

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.



## **21. PEAR: Planner-Executor Agent Robustness Benchmark**

cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07505v1) [paper-pdf](http://arxiv.org/pdf/2510.07505v1)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.



## **22. L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint)**

cs.AI

This preprint was submitted to IEEE TrustCom 2025. The accepted  version will be published under copyright 2025 IEEE

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07363v1) [paper-pdf](http://arxiv.org/pdf/2510.07363v1)

**Authors**: Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu

**Abstract**: The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure.



## **23. Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts**

cs.CL

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07239v1) [paper-pdf](http://arxiv.org/pdf/2510.07239v1)

**Authors**: Christos Ziakas, Nicholas Loo, Nishita Jain, Alessandra Russo

**Abstract**: Automated red-teaming has emerged as a scalable approach for auditing Large Language Models (LLMs) prior to deployment, yet existing approaches lack mechanisms to efficiently adapt to model-specific vulnerabilities at inference. We introduce Red-Bandit, a red-teaming framework that adapts online to identify and exploit model failure modes under distinct attack styles (e.g., manipulation, slang). Red-Bandit post-trains a set of parameter-efficient LoRA experts, each specialized for a particular attack style, using reinforcement learning that rewards the generation of unsafe prompts via a rule-based safety model. At inference, a multi-armed bandit policy dynamically selects among these attack-style experts based on the target model's response safety, balancing exploration and exploitation. Red-Bandit achieves state-of-the-art results on AdvBench under sufficient exploration (ASR@10), while producing more human-readable prompts (lower perplexity). Moreover, Red-Bandit's bandit policy serves as a diagnostic tool for uncovering model-specific vulnerabilities by indicating which attack styles most effectively elicit unsafe behaviors.



## **24. Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples**

cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07192v1) [paper-pdf](http://arxiv.org/pdf/2510.07192v1)

**Authors**: Alexandra Souly, Javier Rando, Ed Chapman, Xander Davies, Burak Hasircioglu, Ezzeldin Shereen, Carlos Mougan, Vasilios Mavroudis, Erik Jones, Chris Hicks, Nicholas Carlini, Yarin Gal, Robert Kirk

**Abstract**: Poisoning attacks can compromise the safety of large language models (LLMs) by injecting malicious documents into their training data. Existing work has studied pretraining poisoning assuming adversaries control a percentage of the training corpus. However, for large models, even small percentages translate to impractically large amounts of data. This work demonstrates for the first time that poisoning attacks instead require a near-constant number of documents regardless of dataset size. We conduct the largest pretraining poisoning experiments to date, pretraining models from 600M to 13B parameters on chinchilla-optimal datasets (6B to 260B tokens). We find that 250 poisoned documents similarly compromise models across all model and dataset sizes, despite the largest models training on more than 20 times more clean data. We also run smaller-scale experiments to ablate factors that could influence attack success, including broader ratios of poisoned to clean data and non-random distributions of poisoned samples. Finally, we demonstrate the same dynamics for poisoning during fine-tuning. Altogether, our results suggest that injecting backdoors through data poisoning may be easier for large models than previously believed as the number of poisons required does not scale up with model size, highlighting the need for more research on defences to mitigate this risk in future models.



## **25. RedTWIZ: Diverse LLM Red Teaming via Adaptive Attack Planning**

cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06994v1) [paper-pdf](http://arxiv.org/pdf/2510.06994v1)

**Authors**: Artur Horal, Daniel Pina, Henrique Paz, Iago Paulo, João Soares, Rafael Ferreira, Diogo Tavares, Diogo Glória-Silva, João Magalhães, David Semedo

**Abstract**: This paper presents the vision, scientific contributions, and technical details of RedTWIZ: an adaptive and diverse multi-turn red teaming framework, to audit the robustness of Large Language Models (LLMs) in AI-assisted software development. Our work is driven by three major research streams: (1) robust and systematic assessment of LLM conversational jailbreaks; (2) a diverse generative multi-turn attack suite, supporting compositional, realistic and goal-oriented jailbreak conversational strategies; and (3) a hierarchical attack planner, which adaptively plans, serializes, and triggers attacks tailored to specific LLM's vulnerabilities. Together, these contributions form a unified framework -- combining assessment, attack generation, and strategic planning -- to comprehensively evaluate and expose weaknesses in LLMs' robustness. Extensive evaluation is conducted to systematically assess and analyze the performance of the overall system and each component. Experimental results demonstrate that our multi-turn adversarial attack strategies can successfully lead state-of-the-art LLMs to produce unsafe generations, highlighting the pressing need for more research into enhancing LLM's robustness.



## **26. VelLMes: A high-interaction AI-based deception framework**

cs.CR

9 pages. 9 figures. 1 table. This is a preprint of a paper that was  presented at the Active Defense and Deception Workshop colocated with IEEE  EuroS&P 2025 conference

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06975v1) [paper-pdf](http://arxiv.org/pdf/2510.06975v1)

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia

**Abstract**: There are very few SotA deception systems based on Large Language Models. The existing ones are limited only to simulating one type of service, mainly SSH shells. These systems - but also the deception technologies not based on LLMs - lack an extensive evaluation that includes human attackers. Generative AI has recently become a valuable asset for cybersecurity researchers and practitioners, and the field of cyber-deception is no exception. Researchers have demonstrated how LLMs can be leveraged to create realistic-looking honeytokens, fake users, and even simulated systems that can be used as honeypots. This paper presents an AI-based deception framework called VelLMes, which can simulate multiple protocols and services such as SSH Linux shell, MySQL, POP3, and HTTP. All of these can be deployed and used as honeypots, thus VelLMes offers a variety of choices for deception design based on the users' needs. VelLMes is designed to be attacked by humans, so interactivity and realism are key for its performance. We evaluate the generative capabilities and the deception capabilities. Generative capabilities were evaluated using unit tests for LLMs. The results of the unit tests show that, with careful prompting, LLMs can produce realistic-looking responses, with some LLMs having a 100% passing rate. In the case of the SSH Linux shell, we evaluated deception capabilities with 89 human attackers. The results showed that about 30% of the attackers thought that they were interacting with a real system when they were assigned an LLM-based honeypot. Lastly, we deployed 10 instances of the SSH Linux shell honeypot on the Internet to capture real-life attacks. Analysis of these attacks showed us that LLM honeypots simulating Linux shells can perform well against unstructured and unexpected attacks on the Internet, responding correctly to most of the issued commands.



## **27. Exposing Citation Vulnerabilities in Generative Engines**

cs.CR

12 pages, under-reviewing at a conference

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06823v1) [paper-pdf](http://arxiv.org/pdf/2510.06823v1)

**Authors**: Riku Mochizuki, Shusuke Komatsu, Souta Noguchi, Kazuto Ataka

**Abstract**: We analyze answers generated by generative engines (GEs) from the perspectives of citation publishers and the content-injection barrier, defined as the difficulty for attackers to manipulate answers to user prompts by placing malicious content on the web. GEs integrate two functions: web search and answer generation that cites web pages using large language models. Because anyone can publish information on the web, GEs are vulnerable to poisoning attacks. Existing studies of citation evaluation focus on how faithfully answer content reflects cited sources, leaving unexamined which web sources should be selected as citations to defend against poisoning attacks. To fill this gap, we introduce evaluation criteria that assess poisoning threats using the citation information contained in answers. Our criteria classify the publisher attributes of citations to estimate the content-injection barrier thereby revealing the threat of poisoning attacks in current GEs. We conduct experiments in political domains in Japan and the United States (U.S.) using our criteria and show that citations from official party websites (primary sources) are approximately \(25\%\)--\(45\%\) in the U.S. and \(60\%\)--\(65\%\) in Japan, indicating that U.S. political answers are at higher risk of poisoning attacks. We also find that sources with low content-injection barriers are frequently cited yet are poorly reflected in answer content. To mitigate this threat, we discuss how publishers of primary sources can increase exposure of their web content in answers and show that well-known techniques are limited by language differences.



## **28. Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness**

cs.LG

17 pages

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06790v1) [paper-pdf](http://arxiv.org/pdf/2510.06790v1)

**Authors**: Tavish McDonald, Bo Lei, Stanislav Fort, Bhavya Kailkhura, Brian Bartoldson

**Abstract**: Models are susceptible to adversarially out-of-distribution (OOD) data despite large training-compute investments into their robustification. Zaremba et al. (2025) make progress on this problem at test time, showing LLM reasoning improves satisfaction of model specifications designed to thwart attacks, resulting in a correlation between reasoning effort and robustness to jailbreaks. However, this benefit of test compute fades when attackers are given access to gradients or multimodal inputs. We address this gap, clarifying that inference-compute offers benefits even in such cases. Our approach argues that compositional generalization, through which OOD data is understandable via its in-distribution (ID) components, enables adherence to defensive specifications on adversarially OOD inputs. Namely, we posit the Robustness from Inference Compute Hypothesis (RICH): inference-compute defenses profit as the model's training data better reflects the attacked data's components. We empirically support this hypothesis across vision language model and attack types, finding robustness gains from test-time compute if specification following on OOD data is unlocked by compositional generalization, while RL finetuning and protracted reasoning are not critical. For example, increasing emphasis on defensive specifications via prompting lowers the success rate of gradient-based multimodal attacks on VLMs robustified by adversarial pretraining, but this same intervention provides no such benefit to not-robustified models. This correlation of inference-compute's robustness benefit with base model robustness is the rich-get-richer dynamic of the RICH: attacked data components are more ID for robustified models, aiding compositional generalization to OOD data. Accordingly, we advise layering train-time and test-time defenses to obtain their synergistic benefit.



## **29. Benchmarking Gaslighting Negation Attacks Against Multimodal Large Language Models**

cs.CL

Project website:  https://yxg1005.github.io/GaslightingNegationAttacks/

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2501.19017v4) [paper-pdf](http://arxiv.org/pdf/2501.19017v4)

**Authors**: Bin Zhu, Yinxuan Gui, Huiyan Qi, Jingjing Chen, Chong-Wah Ngo, Ee-Peng Lim

**Abstract**: Multimodal Large Language Models (MLLMs) have exhibited remarkable advancements in integrating different modalities, excelling in complex understanding and generation tasks. Despite their success, MLLMs remain vulnerable to conversational adversarial inputs. In this paper, we systematically study gaslighting negation attacks: a phenomenon where models, despite initially providing correct answers, are persuaded by user-provided negations to reverse their outputs, often fabricating justifications. We conduct extensive evaluations of state-of-the-art MLLMs across diverse benchmarks and observe substantial performance drops when negation is introduced. Notably, we introduce the first benchmark GaslightingBench, specifically designed to evaluate the vulnerability of MLLMs to negation arguments. GaslightingBench consists of multiple-choice questions curated from existing datasets, along with generated negation prompts across 20 diverse categories. Throughout extensive evaluation, we find that proprietary models such as Gemini-1.5-flash and GPT-4o demonstrate better resilience compared to open-source counterparts like Qwen2-VL and LLaVA, though even advanced reasoning-oriented models like Gemini-2.5-Pro remain susceptible. Our category-level analysis further shows that subjective or socially nuanced domains (e.g., Social Relation, Image Emotion) are especially fragile, while more objective domains (e.g., Geography) exhibit relatively smaller but still notable drops. Overall, all evaluated MLLMs struggle to maintain logical consistency under gaslighting negation attack. These findings highlight a fundamental robustness gap and provide insights for developing more reliable and trustworthy multimodal AI systems. Project website: https://yxg1005.github.io/GaslightingNegationAttacks/.



## **30. Enhancing GraphQL Security by Detecting Malicious Queries Using Large Language Models, Sentence Transformers, and Convolutional Neural Networks**

cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2508.11711v2) [paper-pdf](http://arxiv.org/pdf/2508.11711v2)

**Authors**: Irash Perera, Hiranya Abeyrathne, Sanjeewa Malalgoda, Arshardh Ifthikar

**Abstract**: GraphQL's flexibility, while beneficial for efficient data fetching, introduces unique security vulnerabilities that traditional API security mechanisms often fail to address. Malicious GraphQL queries can exploit the language's dynamic nature, leading to denial-of-service attacks, data exfiltration through injection, and other exploits. Existing solutions, such as static analysis, rate limiting, and general-purpose Web Application Firewalls, offer limited protection against sophisticated, context-aware attacks. This paper presents a novel, AI-driven approach for real-time detection of malicious GraphQL queries. Our method combines static analysis with machine learning techniques, including Large Language Models (LLMs) for dynamic schema-based configuration, Sentence Transformers (SBERT and Doc2Vec) for contextual embedding of query payloads, and Convolutional Neural Networks (CNNs), Random Forests, and Multilayer Perceptrons for classification. We detail the system architecture, implementation strategies optimized for production environments (including ONNX Runtime optimization and parallel processing), and evaluate the performance of our detection models and the overall system under load. Results demonstrate high accuracy in detecting various threats, including SQL injection, OS command injection, and XSS exploits, alongside effective mitigation of DoS and SSRF attempts. This research contributes a robust and adaptable solution for enhancing GraphQL API security.



## **31. Membership Inference Attacks on LLM-based Recommender Systems**

cs.IR

this paper is under review

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2508.18665v3) [paper-pdf](http://arxiv.org/pdf/2508.18665v3)

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.



## **32. AutoDAN-Reasoning: Enhancing Strategies Exploration based Jailbreak Attacks with Test-Time Scaling**

cs.CR

Technical report. Code is available at  https://github.com/SaFoLab-WISC/AutoDAN-Reasoning

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.05379v2) [paper-pdf](http://arxiv.org/pdf/2510.05379v2)

**Authors**: Xiaogeng Liu, Chaowei Xiao

**Abstract**: Recent advancements in jailbreaking large language models (LLMs), such as AutoDAN-Turbo, have demonstrated the power of automated strategy discovery. AutoDAN-Turbo employs a lifelong learning agent to build a rich library of attack strategies from scratch. While highly effective, its test-time generation process involves sampling a strategy and generating a single corresponding attack prompt, which may not fully exploit the potential of the learned strategy library. In this paper, we propose to further improve the attack performance of AutoDAN-Turbo through test-time scaling. We introduce two distinct scaling methods: Best-of-N and Beam Search. The Best-of-N method generates N candidate attack prompts from a sampled strategy and selects the most effective one based on a scorer model. The Beam Search method conducts a more exhaustive search by exploring combinations of strategies from the library to discover more potent and synergistic attack vectors. According to the experiments, the proposed methods significantly boost performance, with Beam Search increasing the attack success rate by up to 15.6 percentage points on Llama-3.1-70B-Instruct and achieving a nearly 60% relative improvement against the highly robust GPT-o4-mini compared to the vanilla method.



## **33. Towards the Worst-case Robustness of Large Language Models**

cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2501.19040v4) [paper-pdf](http://arxiv.org/pdf/2501.19040v4)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of large language models to adversarial attacks, where adversaries craft specific input sequences to induce harmful, violent, private, or incorrect outputs. In this work, we study their worst-case robustness, i.e., whether an adversarial example exists that leads to such undesirable outputs. We upper bound the worst-case robustness using stronger white-box attacks, indicating that most current deterministic defenses achieve nearly 0\% worst-case robustness. We propose a general tight lower bound for randomized smoothing using fractional knapsack solvers or 0-1 knapsack solvers, and using them to bound the worst-case robustness of all stochastic defenses. Based on these solvers, we provide theoretical lower bounds for several previous empirical defenses. For example, we certify the robustness of a specific case, smoothing using a uniform kernel, against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.



## **34. Code Agent can be an End-to-end System Hacker: Benchmarking Real-world Threats of Computer-use Agent**

cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06607v1) [paper-pdf](http://arxiv.org/pdf/2510.06607v1)

**Authors**: Weidi Luo, Qiming Zhang, Tianyu Lu, Xiaogeng Liu, Bin Hu, Hung-Chun Chiu, Siyuan Ma, Yizhe Zhang, Xusheng Xiao, Yinzhi Cao, Zhen Xiang, Chaowei Xiao

**Abstract**: Computer-use agent (CUA) frameworks, powered by large language models (LLMs) or multimodal LLMs (MLLMs), are rapidly maturing as assistants that can perceive context, reason, and act directly within software environments. Among their most critical applications is operating system (OS) control. As CUAs in the OS domain become increasingly embedded in daily operations, it is imperative to examine their real-world security implications, specifically whether CUAs can be misused to perform realistic, security-relevant attacks. Existing works exhibit four major limitations: Missing attacker-knowledge model on tactics, techniques, and procedures (TTP), Incomplete coverage for end-to-end kill chains, unrealistic environment without multi-host and encrypted user credentials, and unreliable judgment dependent on LLM-as-a-Judge. To address these gaps, we propose AdvCUA, the first benchmark aligned with real-world TTPs in MITRE ATT&CK Enterprise Matrix, which comprises 140 tasks, including 40 direct malicious tasks, 74 TTP-based malicious tasks, and 26 end-to-end kill chains, systematically evaluates CUAs under a realistic enterprise OS security threat in a multi-host environment sandbox by hard-coded evaluation. We evaluate the existing five mainstream CUAs, including ReAct, AutoGPT, Gemini CLI, Cursor CLI, and Cursor IDE based on 8 foundation LLMs. The results demonstrate that current frontier CUAs do not adequately cover OS security-centric threats. These capabilities of CUAs reduce dependence on custom malware and deep domain expertise, enabling even inexperienced attackers to mount complex enterprise intrusions, which raises social concern about the responsibility and security of CUAs.



## **35. Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?**

cs.CL

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06594v1) [paper-pdf](http://arxiv.org/pdf/2510.06594v1)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.



## **36. MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks**

cs.LG

Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2502.17832v3) [paper-pdf](http://arxiv.org/pdf/2502.17832v3)

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-Wei Chang, Daniel Kang, Heng Ji

**Abstract**: Multimodal large language models with Retrieval Augmented Generation (RAG) have significantly advanced tasks such as multimodal question answering by grounding responses in external text and images. This grounding improves factuality, reduces hallucination, and extends reasoning beyond parametric knowledge. However, this reliance on external knowledge poses a critical yet underexplored safety risk: knowledge poisoning attacks, where adversaries deliberately inject adversarial multimodal content into external knowledge bases to steer model toward generating incorrect or even harmful responses. To expose such vulnerabilities, we propose MM-PoisonRAG, the first framework to systematically design knowledge poisoning in multimodal RAG. We introduce two complementary attack strategies: Localized Poisoning Attack (LPA), which implants targeted multimodal misinformation to manipulate specific queries, and Globalized Poisoning Attack (GPA), which inserts a single adversarial knowledge to broadly disrupt reasoning and induce nonsensical responses across all queries. Comprehensive experiments across tasks, models, and access settings show that LPA achieves targeted manipulation with attack success rates of up to 56%, while GPA completely disrupts model generation to 0% accuracy with just a single adversarial knowledge injection. Our results reveal the fragility of multimodal RAG and highlight the urgent need for defenses against knowledge poisoning.



## **37. From Description to Detection: LLM based Extendable O-RAN Compliant Blind DoS Detection in 5G and Beyond**

cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06530v1) [paper-pdf](http://arxiv.org/pdf/2510.06530v1)

**Authors**: Thusitha Dayaratne, Ngoc Duy Pham, Viet Vo, Shangqi Lai, Sharif Abuadbba, Hajime Suzuki, Xingliang Yuan, Carsten Rudolph

**Abstract**: The quality and experience of mobile communication have significantly improved with the introduction of 5G, and these improvements are expected to continue beyond the 5G era. However, vulnerabilities in control-plane protocols, such as Radio Resource Control (RRC) and Non-Access Stratum (NAS), pose significant security threats, such as Blind Denial of Service (DoS) attacks. Despite the availability of existing anomaly detection methods that leverage rule-based systems or traditional machine learning methods, these methods have several limitations, including the need for extensive training data, predefined rules, and limited explainability. Addressing these challenges, we propose a novel anomaly detection framework that leverages the capabilities of Large Language Models (LLMs) in zero-shot mode with unordered data and short natural language attack descriptions within the Open Radio Access Network (O-RAN) architecture. We analyse robustness to prompt variation, demonstrate the practicality of automating the attack descriptions and show that detection quality relies on the semantic completeness of the description rather than its phrasing or length. We utilise an RRC/NAS dataset to evaluate the solution and provide an extensive comparison of open-source and proprietary LLM implementations to demonstrate superior performance in attack detection. We further validate the practicality of our framework within O-RAN's real-time constraints, illustrating its potential for detecting other Layer-3 attacks.



## **38. Text-to-Image Models Leave Identifiable Signatures: Implications for Leaderboard Security**

cs.LG

Accepted at Lock-LLM Workshop, NeurIPS 2025

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.06525v1) [paper-pdf](http://arxiv.org/pdf/2510.06525v1)

**Authors**: Ali Naseh, Anshuman Suri, Yuefeng Peng, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Generative AI leaderboards are central to evaluating model capabilities, but remain vulnerable to manipulation. Among key adversarial objectives is rank manipulation, where an attacker must first deanonymize the models behind displayed outputs -- a threat previously demonstrated and explored for large language models (LLMs). We show that this problem can be even more severe for text-to-image leaderboards, where deanonymization is markedly easier. Using over 150,000 generated images from 280 prompts and 19 diverse models spanning multiple organizations, architectures, and sizes, we demonstrate that simple real-time classification in CLIP embedding space identifies the generating model with high accuracy, even without prompt control or historical data. We further introduce a prompt-level separability metric and identify prompts that enable near-perfect deanonymization. Our results indicate that rank manipulation in text-to-image leaderboards is easier than previously recognized, underscoring the need for stronger defenses.



## **39. An Embarrassingly Simple Defense Against LLM Abliteration Attacks**

cs.CL

preprint - under review

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2505.19056v2) [paper-pdf](http://arxiv.org/pdf/2505.19056v2)

**Authors**: Harethah Abu Shairah, Hasan Abed Al Kader Hammoud, Bernard Ghanem, George Turkiyyah

**Abstract**: Large language models (LLMs) are typically aligned to refuse harmful instructions through safety fine-tuning. A recent attack, termed abliteration, identifies and suppresses the single latent direction most responsible for refusal behavior, thereby enabling models to generate harmful content. We propose a defense that fundamentally alters how models express refusal. We construct an extended-refusal dataset in which responses to harmful prompts provide detailed justifications before refusing, distributing the refusal signal across multiple token positions. Fine-tuning Llama-2-7B-Chat and Qwen2.5-Instruct (1.5B and 3B parameters) on this dataset yields models that maintain high refusal rates under abliteration: refusal rates drop by at most 10%, compared to 70-80% drops in baseline models. Comprehensive evaluations of safety and utility demonstrate that extended-refusal fine-tuning effectively neutralizes abliteration attacks while preserving general model performance and enhancing robustness across multiple alignment scenarios.



## **40. SAFER: Advancing Safety Alignment via Efficient Ex-Ante Reasoning**

cs.CL

22 pages, 5 figures

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2504.02725v2) [paper-pdf](http://arxiv.org/pdf/2504.02725v2)

**Authors**: Kehua Feng, Keyan Ding, Yuhao Wang, Menghan Li, Fanjunduo Wei, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose SAFER, a framework for Safety Alignment via eFficient Ex-Ante Reasoning. Our approach instantiates structured Ex-Ante reasoning through initial assessment, rule verification, and path calibration, and embeds predefined safety rules to provide transparent and verifiable safety judgments. Specifically, our approach consists of two training stages: (1) supervised fine-tuning with synthetic traces to teach the multi-stage Ex-Ante reasoning, and (2) step-level reasoning preference optimization to jointly enhance safety, utility, and efficiency. Experiments on multiple open-source LLMs demonstrate that SAFER significantly enhances safety performance while maintaining helpfulness and response efficiency.



## **41. Geometry-Guided Adversarial Prompt Detection via Curvature and Local Intrinsic Dimension**

cs.CL

40 Pages, 6 figues

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2503.03502v2) [paper-pdf](http://arxiv.org/pdf/2503.03502v2)

**Authors**: Canaan Yung, Hanxun Huang, Christopher Leckie, Sarah Erfani

**Abstract**: Adversarial prompts are capable of jailbreaking frontier large language models (LLMs) and inducing undesirable behaviours, posing a significant obstacle to their safe deployment. Current mitigation strategies primarily rely on activating built-in defence mechanisms or fine-tuning LLMs, both of which are computationally expensive and can sacrifice model utility. In contrast, detection-based approaches are more efficient and practical for deployment in real-world applications. However, the fundamental distinctions between adversarial and benign prompts remain poorly understood. In this work, we introduce CurvaLID, a novel defence framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures. CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. To further enhance our solution, we leverage Local Intrinsic Dimensionality (LID) to capture complementary geometric features of text prompts within adversarial subspaces. Our findings show that adversarial prompts exhibit distinct geometric signatures from benign prompts, enabling CurvaLID to achieve near-perfect classification and outperform state-of-the-art detectors in adversarial prompt detection. CurvaLID provides a reliable and efficient safeguard against malicious queries as a model-agnostic method that generalises across multiple LLMs and attack families.



## **42. Towards Reliable and Practical LLM Security Evaluations via Bayesian Modelling**

cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05709v1) [paper-pdf](http://arxiv.org/pdf/2510.05709v1)

**Authors**: Mary Llewellyn, Annie Gray, Josh Collyer, Michael Harries

**Abstract**: Before adopting a new large language model (LLM) architecture, it is critical to understand vulnerabilities accurately. Existing evaluations can be difficult to trust, often drawing conclusions from LLMs that are not meaningfully comparable, relying on heuristic inputs or employing metrics that fail to capture the inherent uncertainty. In this paper, we propose a principled and practical end-to-end framework for evaluating LLM vulnerabilities to prompt injection attacks. First, we propose practical approaches to experimental design, tackling unfair LLM comparisons by considering two practitioner scenarios: when training an LLM and when deploying a pre-trained LLM. Second, we address the analysis of experiments and propose a Bayesian hierarchical model with embedding-space clustering. This model is designed to improve uncertainty quantification in the common scenario that LLM outputs are not deterministic, test prompts are designed imperfectly, and practitioners only have a limited amount of compute to evaluate vulnerabilities. We show the improved inferential capabilities of the model in several prompt injection attack settings. Finally, we demonstrate the pipeline to evaluate the security of Transformer versus Mamba architectures. Our findings show that consideration of output variability can suggest less definitive findings. However, for some attacks, we find notably increased Transformer and Mamba-variant vulnerabilities across LLMs with the same training data or mathematical ability.



## **43. Membership Inference Attacks on Tokenizers of Large Language Models**

cs.CR

Code is available at: https://github.com/mengtong0110/Tokenizer-MIA

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05699v1) [paper-pdf](http://arxiv.org/pdf/2510.05699v1)

**Authors**: Meng Tong, Yuntao Du, Kejiang Chen, Weiming Zhang, Ninghui Li

**Abstract**: Membership inference attacks (MIAs) are widely used to assess the privacy risks associated with machine learning models. However, when these attacks are applied to pre-trained large language models (LLMs), they encounter significant challenges, including mislabeled samples, distribution shifts, and discrepancies in model size between experimental and real-world settings. To address these limitations, we introduce tokenizers as a new attack vector for membership inference. Specifically, a tokenizer converts raw text into tokens for LLMs. Unlike full models, tokenizers can be efficiently trained from scratch, thereby avoiding the aforementioned challenges. In addition, the tokenizer's training data is typically representative of the data used to pre-train LLMs. Despite these advantages, the potential of tokenizers as an attack vector remains unexplored. To this end, we present the first study on membership leakage through tokenizers and explore five attack methods to infer dataset membership. Extensive experiments on millions of Internet samples reveal the vulnerabilities in the tokenizers of state-of-the-art LLMs. To mitigate this emerging risk, we further propose an adaptive defense. Our findings highlight tokenizers as an overlooked yet critical privacy threat, underscoring the urgent need for privacy-preserving mechanisms specifically designed for them.



## **44. Bypassing Prompt Guards in Production with Controlled-Release Prompting**

cs.LG

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.01529v2) [paper-pdf](http://arxiv.org/pdf/2510.01529v2)

**Authors**: Jaiden Fairoze, Sanjam Garg, Keewoo Lee, Mingyuan Wang

**Abstract**: As large language models (LLMs) advance, ensuring AI safety and alignment is paramount. One popular approach is prompt guards, lightweight mechanisms designed to filter malicious queries while being easy to implement and update. In this work, we introduce a new attack that circumvents such prompt guards, highlighting their limitations. Our method consistently jailbreaks production models while maintaining response quality, even under the highly protected chat interfaces of Google Gemini (2.5 Flash/Pro), DeepSeek Chat (DeepThink), Grok (3), and Mistral Le Chat (Magistral). The attack exploits a resource asymmetry between the prompt guard and the main LLM, encoding a jailbreak prompt that lightweight guards cannot decode but the main model can. This reveals an attack surface inherent to lightweight prompt guards in modern LLM architectures and underscores the need to shift defenses from blocking malicious inputs to preventing malicious outputs. We additionally identify other critical alignment issues, such as copyrighted data extraction, training data extraction, and malicious response leakage during thinking.



## **45. AutoPentester: An LLM Agent-based Framework for Automated Pentesting**

cs.CR

IEEE TrustCom 2025 10 pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05605v1) [paper-pdf](http://arxiv.org/pdf/2510.05605v1)

**Authors**: Yasod Ginige, Akila Niroshan, Sajal Jain, Suranga Seneviratne

**Abstract**: Penetration testing and vulnerability assessment are essential industry practices for safeguarding computer systems. As cyber threats grow in scale and complexity, the demand for pentesting has surged, surpassing the capacity of human professionals to meet it effectively. With advances in AI, particularly Large Language Models (LLMs), there have been attempts to automate the pentesting process. However, existing tools such as PentestGPT are still semi-manual, requiring significant professional human interaction to conduct pentests. To this end, we propose a novel LLM agent-based framework, AutoPentester, which automates the pentesting process. Given a target IP, AutoPentester automatically conducts pentesting steps using common security tools in an iterative process. It can dynamically generate attack strategies based on the tool outputs from the previous iteration, mimicking the human pentester approach. We evaluate AutoPentester using Hack The Box and custom-made VMs, comparing the results with the state-of-the-art PentestGPT. Results show that AutoPentester achieves a 27.0% better subtask completion rate and 39.5% more vulnerability coverage with fewer steps. Most importantly, it requires significantly fewer human interactions and interventions compared to PentestGPT. Furthermore, we recruit a group of security industry professional volunteers for a user survey and perform a qualitative analysis to evaluate AutoPentester against industry practices and compare it with PentestGPT. On average, AutoPentester received a score of 3.93 out of 5 based on user reviews, which was 19.8% higher than PentestGPT.



## **46. A Middle Path for On-Premises LLM Deployment: Preserving Privacy Without Sacrificing Model Confidentiality**

cs.LG

8 pages for main content of the paper

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2410.11182v3) [paper-pdf](http://arxiv.org/pdf/2410.11182v3)

**Authors**: Hanbo Huang, Yihan Li, Bowen Jiang, Bo Jiang, Lin Liu, Ruoyu Sun, Zhuotao Liu, Shiyu Liang

**Abstract**: Privacy-sensitive users require deploying large language models (LLMs) within their own infrastructure (on-premises) to safeguard private data and enable customization. However, vulnerabilities in local environments can lead to unauthorized access and potential model theft. To address this, prior research on small models has explored securing only the output layer within hardware-secured devices to balance model confidentiality and customization. Yet this approach fails to protect LLMs effectively. In this paper, we discover that (1) query-based distillation attacks targeting the secured top layer can produce a functionally equivalent replica of the victim model; (2) securing the same number of layers, bottom layers before a transition layer provide stronger protection against distillation attacks than top layers, with comparable effects on customization performance; and (3) the number of secured layers creates a trade-off between protection and customization flexibility. Based on these insights, we propose SOLID, a novel deployment framework that secures a few bottom layers in a secure environment and introduces an efficient metric to optimize the trade-off by determining the ideal number of hidden layers. Extensive experiments on five models (1.3B to 70B parameters) demonstrate that SOLID outperforms baselines, achieving a better balance between protection and downstream customization.



## **47. Adversarial Reinforcement Learning for Large Language Model Agent Safety**

cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05442v1) [paper-pdf](http://arxiv.org/pdf/2510.05442v1)

**Authors**: Zizhao Wang, Dingcheng Li, Vaishakh Keshava, Phillip Wallis, Ananth Balashankar, Peter Stone, Lukas Rutishauser

**Abstract**: Large Language Model (LLM) agents can leverage tools such as Google Search to complete complex tasks. However, this tool usage introduces the risk of indirect prompt injections, where malicious instructions hidden in tool outputs can manipulate the agent, posing security risks like data leakage. Current defense strategies typically rely on fine-tuning LLM agents on datasets of known attacks. However, the generation of these datasets relies on manually crafted attack patterns, which limits their diversity and leaves agents vulnerable to novel prompt injections. To address this limitation, we propose Adversarial Reinforcement Learning for Agent Safety (ARLAS), a novel framework that leverages adversarial reinforcement learning (RL) by formulating the problem as a two-player zero-sum game. ARLAS co-trains two LLMs: an attacker that learns to autonomously generate diverse prompt injections and an agent that learns to defend against them while completing its assigned tasks. To ensure robustness against a wide range of attacks and to prevent cyclic learning, we employ a population-based learning framework that trains the agent to defend against all previous attacker checkpoints. Evaluated on BrowserGym and AgentDojo, agents fine-tuned with ARLAS achieve a significantly lower attack success rate than the original model while also improving their task success rate. Our analysis further confirms that the adversarial process generates a diverse and challenging set of attacks, leading to a more robust agent compared to the base model.



## **48. DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable Language Models Using Adam Optimization with Adaptive Clipping**

cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05288v1) [paper-pdf](http://arxiv.org/pdf/2510.05288v1)

**Authors**: Ruoxing Yang

**Abstract**: Large language models (LLMs) such as ChatGPT have evolved into powerful and ubiquitous tools. Fine-tuning on small datasets allows LLMs to acquire specialized skills for specific tasks efficiently. Although LLMs provide great utility in both general and task-specific use cases, they are limited by two security-related concerns. First, traditional LLM hardware requirements make them infeasible to run locally on consumer-grade devices. A remote network connection with the LLM provider's server is usually required, making the system vulnerable to network attacks. Second, fine-tuning an LLM for a sensitive task may involve sensitive data. Non-private fine-tuning algorithms produce models vulnerable to training data reproduction attacks. Our work addresses these security concerns by enhancing differentially private optimization algorithms and applying them to fine-tune localizable language models. We introduce adaptable gradient clipping along with other engineering enhancements to the standard DP-Adam optimizer to create DP-Adam-AC. We use our optimizer to fine-tune examples of two localizable LLM designs, small language model (Qwen2.5-0.5B) and 1.58 bit quantization (Bitnet-b1.58-2B). We demonstrate promising improvements in loss through experimentation with two synthetic datasets.



## **49. Proactive defense against LLM Jailbreak**

cs.CR

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05052v1) [paper-pdf](http://arxiv.org/pdf/2510.05052v1)

**Authors**: Weiliang Zhao, Jinjun Peng, Daniel Ben-Levi, Zhou Yu, Junfeng Yang

**Abstract**: The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks.



## **50. Rethinking Exact Unlearning under Exposure: Extracting Forgotten Data under Exact Unlearning in Large Language Model**

cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2505.24379v2) [paper-pdf](http://arxiv.org/pdf/2505.24379v2)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.



