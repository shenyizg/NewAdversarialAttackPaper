# Latest Large Language Model Attack Papers
**update at 2025-06-10 15:42:08**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. TokenBreak: Bypassing Text Classification Models Through Token Manipulation**

cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07948v1) [paper-pdf](http://arxiv.org/pdf/2506.07948v1)

**Authors**: Kasimir Schulz, Kenneth Yeung, Kieran Evans

**Abstract**: Natural Language Processing (NLP) models are used for text-related tasks such as classification and generation. To complete these tasks, input data is first tokenized from human-readable text into a format the model can understand, enabling it to make inferences and understand context. Text classification models can be implemented to guard against threats such as prompt injection attacks against Large Language Models (LLMs), toxic input and cybersecurity risks such as spam emails. In this paper, we introduce TokenBreak: a novel attack that can bypass these protection models by taking advantage of the tokenization strategy they use. This attack technique manipulates input text in such a way that certain models give an incorrect classification. Importantly, the end target (LLM or email recipient) can still understand and respond to the manipulated text and therefore be vulnerable to the very attack the protection model was put in place to prevent. The tokenizer is tied to model architecture, meaning it is possible to predict whether or not a model is vulnerable to attack based on family. We also present a defensive strategy as an added layer of protection that can be implemented without having to retrain the defensive model.



## **2. Adversarial Attack Classification and Robustness Testing for Large Language Models for Code**

cs.SE

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07942v1) [paper-pdf](http://arxiv.org/pdf/2506.07942v1)

**Authors**: Yang Liu, Armstrong Foundjem, Foutse Khomh, Heng Li

**Abstract**: Large Language Models (LLMs) have become vital tools in software development tasks such as code generation, completion, and analysis. As their integration into workflows deepens, ensuring robustness against vulnerabilities especially those triggered by diverse or adversarial inputs becomes increasingly important. Such vulnerabilities may lead to incorrect or insecure code generation when models encounter perturbed task descriptions, code, or comments. Prior research often overlooks the role of natural language in guiding code tasks. This study investigates how adversarial perturbations in natural language inputs including prompts, comments, and descriptions affect LLMs for Code (LLM4Code). It examines the effects of perturbations at the character, word, and sentence levels to identify the most impactful vulnerabilities. We analyzed multiple projects (e.g., ReCode, OpenAttack) and datasets (e.g., HumanEval, MBPP), establishing a taxonomy of adversarial attacks. The first dimension classifies the input type code, prompts, or comments while the second dimension focuses on granularity: character, word, or sentence-level changes. We adopted a mixed-methods approach, combining quantitative performance metrics with qualitative vulnerability analysis. LLM4Code models show varying robustness across perturbation types. Sentence-level attacks were least effective, suggesting models are resilient to broader contextual changes. In contrast, word-level perturbations posed serious challenges, exposing semantic vulnerabilities. Character-level effects varied, showing model sensitivity to subtle syntactic deviations.Our study offers a structured framework for testing LLM4Code robustness and emphasizes the critical role of natural language in adversarial evaluation. Improving model resilience to semantic-level disruptions is essential for secure and reliable code-generation systems.



## **3. SoK: Data Reconstruction Attacks Against Machine Learning Models: Definition, Metrics, and Benchmark**

cs.CR

To Appear in the 34th USENIX Security Symposium, August 13-15, 2025

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07888v1) [paper-pdf](http://arxiv.org/pdf/2506.07888v1)

**Authors**: Rui Wen, Yiyong Liu, Michael Backes, Yang Zhang

**Abstract**: Data reconstruction attacks, which aim to recover the training dataset of a target model with limited access, have gained increasing attention in recent years. However, there is currently no consensus on a formal definition of data reconstruction attacks or appropriate evaluation metrics for measuring their quality. This lack of rigorous definitions and universal metrics has hindered further advancement in this field. In this paper, we address this issue in the vision domain by proposing a unified attack taxonomy and formal definitions of data reconstruction attacks. We first propose a set of quantitative evaluation metrics that consider important criteria such as quantifiability, consistency, precision, and diversity. Additionally, we leverage large language models (LLMs) as a substitute for human judgment, enabling visual evaluation with an emphasis on high-quality reconstructions. Using our proposed taxonomy and metrics, we present a unified framework for systematically evaluating the strengths and limitations of existing attacks and establishing a benchmark for future research. Empirical results, primarily from a memorization perspective, not only validate the effectiveness of our metrics but also offer valuable insights for designing new attacks.



## **4. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2406.12091v4) [paper-pdf](http://arxiv.org/pdf/2406.12091v4)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.



## **5. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

cs.AI

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07736v1) [paper-pdf](http://arxiv.org/pdf/2506.07736v1)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.



## **6. Representation Bending for Large Language Model Safety**

cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2504.01550v2) [paper-pdf](http://arxiv.org/pdf/2504.01550v2)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.



## **7. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

cs.CL

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2505.17654v2) [paper-pdf](http://arxiv.org/pdf/2505.17654v2)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.



## **8. SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage**

cs.CR

To appear at Findings of ACL 2025

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2412.15289v4) [paper-pdf](http://arxiv.org/pdf/2412.15289v4)

**Authors**: Xiaoning Dong, Wenbo Hu, Wei Xu, Tianxing He

**Abstract**: Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.



## **9. Evaluating LLMs Robustness in Less Resourced Languages with Proxy Models**

cs.CL

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07645v1) [paper-pdf](http://arxiv.org/pdf/2506.07645v1)

**Authors**: Maciej Chrabąszcz, Katarzyna Lorenc, Karolina Seweryn

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across various natural language processing (NLP) tasks in recent years. However, their susceptibility to jailbreaks and perturbations necessitates additional evaluations. Many LLMs are multilingual, but safety-related training data contains mainly high-resource languages like English. This can leave them vulnerable to perturbations in low-resource languages such as Polish. We show how surprisingly strong attacks can be cheaply created by altering just a few characters and using a small proxy model for word importance calculation. We find that these character and word-level attacks drastically alter the predictions of different LLMs, suggesting a potential vulnerability that can be used to circumvent their internal safety mechanisms. We validate our attack construction methodology on Polish, a low-resource language, and find potential vulnerabilities of LLMs in this language. Additionally, we show how it can be extended to other languages. We release the created datasets and code for further research.



## **10. MalGEN: A Generative Agent Framework for Modeling Malicious Software in Cybersecurity**

cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07586v1) [paper-pdf](http://arxiv.org/pdf/2506.07586v1)

**Authors**: Bikash Saha, Sandeep Kumar Shukla

**Abstract**: The dual use nature of Large Language Models (LLMs) presents a growing challenge in cybersecurity. While LLM enhances automation and reasoning for defenders, they also introduce new risks, particularly their potential to be misused for generating evasive, AI crafted malware. Despite this emerging threat, the research community currently lacks controlled and extensible tools that can simulate such behavior for testing and defense preparation. We present MalGEN, a multi agent framework that simulates coordinated adversarial behavior to generate diverse, activity driven malware samples. The agents work collaboratively to emulate attacker workflows, including payload planning, capability selection, and evasion strategies, within a controlled environment built for ethical and defensive research. Using MalGEN, we synthesized ten novel malware samples and evaluated them against leading antivirus and behavioral detection engines. Several samples exhibited stealthy and evasive characteristics that bypassed current defenses, validating MalGEN's ability to model sophisticated and new threats. By transforming the threat of LLM misuse into an opportunity for proactive defense, MalGEN offers a valuable framework for evaluating and strengthening cybersecurity systems. The framework addresses data scarcity, enables rigorous testing, and supports the development of resilient and future ready detection strategies.



## **11. When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment**

cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07452v1) [paper-pdf](http://arxiv.org/pdf/2506.07452v1)

**Authors**: Yuxin Xiao, Sana Tonekaboni, Walter Gerych, Vinith Suriyakumar, Marzyeh Ghassemi

**Abstract**: Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in jailbreak queries. Although these style patterns are semantically unrelated to the malicious intents behind jailbreak queries, their safety impact remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We evaluate 32 LLMs across seven jailbreak benchmarks, and find that malicious queries with style patterns inflate the attack success rate (ASR) for nearly all models. Notably, ASR inflation correlates with both the length of style patterns and the relative attention an LLM exhibits on them. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs and five fine-tuning style settings, SafeStyle consistently outperforms baselines in maintaining LLM safety.



## **12. A Red Teaming Roadmap Towards System-Level Safety**

cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.05376v2) [paper-pdf](http://arxiv.org/pdf/2506.05376v2)

**Authors**: Zifan Wang, Christina Q. Knight, Jeremy Kritz, Willow E. Primack, Julian Michael

**Abstract**: Large Language Model (LLM) safeguards, which implement request refusals, have become a widely adopted mitigation strategy against misuse. At the intersection of adversarial machine learning and AI safety, safeguard red teaming has effectively identified critical vulnerabilities in state-of-the-art refusal-trained LLMs. However, in our view the many conference submissions on LLM red teaming do not, in aggregate, prioritize the right research problems. First, testing against clear product safety specifications should take a higher priority than abstract social biases or ethical principles. Second, red teaming should prioritize realistic threat models that represent the expanding risk landscape and what real attackers might do. Finally, we contend that system-level safety is a necessary step to move red teaming research forward, as AI models present new threats as well as affordances for threat mitigation (e.g., detection and banning of malicious users) once placed in a deployment context. Adopting these priorities will be necessary in order for red teaming research to adequately address the slate of new threats that rapid AI advances present today and will present in the very near future.



## **13. Beyond Jailbreaks: Revealing Stealthier and Broader LLM Security Risks Stemming from Alignment Failures**

cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07402v1) [paper-pdf](http://arxiv.org/pdf/2506.07402v1)

**Authors**: Yukai Zhou, Sibei Yang, Wenjie Wang

**Abstract**: Large language models (LLMs) are increasingly deployed in real-world applications, raising concerns about their security. While jailbreak attacks highlight failures under overtly harmful queries, they overlook a critical risk: incorrectly answering harmless-looking inputs can be dangerous and cause real-world harm (Implicit Harm). We systematically reformulate the LLM risk landscape through a structured quadrant perspective based on output factuality and input harmlessness, uncovering an overlooked high-risk region. To investigate this gap, we propose JailFlipBench, a benchmark aims to capture implicit harm, spanning single-modal, multimodal, and factual extension scenarios with diverse evaluation metrics. We further develop initial JailFlip attack methodologies and conduct comprehensive evaluations across multiple open-source and black-box LLMs, show that implicit harm present immediate and urgent real-world risks, calling for broader LLM safety assessments and alignment beyond conventional jailbreak paradigms.



## **14. MrM: Black-Box Membership Inference Attacks against Multimodal RAG Systems**

cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07399v1) [paper-pdf](http://arxiv.org/pdf/2506.07399v1)

**Authors**: Peiru Yang, Jinhua Yin, Haoran Zheng, Xueying Bai, Huili Wang, Yufei Sun, Xintian Li, Shangguang Wang, Yongfeng Huang, Tao Qi

**Abstract**: Multimodal retrieval-augmented generation (RAG) systems enhance large vision-language models by integrating cross-modal knowledge, enabling their increasing adoption across real-world multimodal tasks. These knowledge databases may contain sensitive information that requires privacy protection. However, multimodal RAG systems inherently grant external users indirect access to such data, making them potentially vulnerable to privacy attacks, particularly membership inference attacks (MIAs). % Existing MIA methods targeting RAG systems predominantly focus on the textual modality, while the visual modality remains relatively underexplored. To bridge this gap, we propose MrM, the first black-box MIA framework targeted at multimodal RAG systems. It utilizes a multi-object data perturbation framework constrained by counterfactual attacks, which can concurrently induce the RAG systems to retrieve the target data and generate information that leaks the membership information. Our method first employs an object-aware data perturbation method to constrain the perturbation to key semantics and ensure successful retrieval. Building on this, we design a counterfact-informed mask selection strategy to prioritize the most informative masked regions, aiming to eliminate the interference of model self-knowledge and amplify attack efficacy. Finally, we perform statistical membership inference by modeling query trials to extract features that reflect the reconstruction of masked semantics from response patterns. Experiments on two visual datasets and eight mainstream commercial visual-language models (e.g., GPT-4o, Gemini-2) demonstrate that MrM achieves consistently strong performance across both sample-level and set-level evaluations, and remains robust under adaptive defenses.



## **15. Knowledge-to-Jailbreak: Investigating Knowledge-driven Jailbreaking Attacks for Large Language Models**

cs.CL

Accepted by KDD 2025 research track

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2406.11682v2) [paper-pdf](http://arxiv.org/pdf/2406.11682v2)

**Authors**: Shangqing Tu, Zhuoran Pan, Wenxuan Wang, Zhexin Zhang, Yuliang Sun, Jifan Yu, Hongning Wang, Lei Hou, Juanzi Li

**Abstract**: Large language models (LLMs) have been increasingly applied to various domains, which triggers increasing concerns about LLMs' safety on specialized domains, e.g. medicine. Despite prior explorations on general jailbreaking attacks, there are two challenges for applying existing attacks on testing the domain-specific safety of LLMs: (1) Lack of professional knowledge-driven attacks, (2) Insufficient coverage of domain knowledge. To bridge this gap, we propose a new task, knowledge-to-jailbreak, which aims to generate jailbreaking attacks from domain knowledge, requiring both attack effectiveness and knowledge relevance. We collect a large-scale dataset with 12,974 knowledge-jailbreak pairs and fine-tune a large language model as jailbreak-generator, to produce domain knowledge-specific jailbreaks. Experiments on 13 domains and 8 target LLMs demonstrate the effectiveness of jailbreak-generator in generating jailbreaks that are both threatening to the target LLMs and relevant to the given knowledge. We also apply our method to an out-of-domain knowledge base, showing that jailbreak-generator can generate jailbreaks that are comparable in harmfulness to those crafted by human experts. Data and code are available at: https://github.com/THU-KEG/Knowledge-to-Jailbreak/.



## **16. Backdoor Attack on Vision Language Models with Stealthy Semantic Manipulation**

cs.CV

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07214v1) [paper-pdf](http://arxiv.org/pdf/2506.07214v1)

**Authors**: Zhiyuan Zhong, Zhen Sun, Yepang Liu, Xinlei He, Guanhong Tao

**Abstract**: Vision Language Models (VLMs) have shown remarkable performance, but are also vulnerable to backdoor attacks whereby the adversary can manipulate the model's outputs through hidden triggers. Prior attacks primarily rely on single-modality triggers, leaving the crucial cross-modal fusion nature of VLMs largely unexplored. Unlike prior work, we identify a novel attack surface that leverages cross-modal semantic mismatches as implicit triggers. Based on this insight, we propose BadSem (Backdoor Attack with Semantic Manipulation), a data poisoning attack that injects stealthy backdoors by deliberately misaligning image-text pairs during training. To perform the attack, we construct SIMBad, a dataset tailored for semantic manipulation involving color and object attributes. Extensive experiments across four widely used VLMs show that BadSem achieves over 98% average ASR, generalizes well to out-of-distribution datasets, and can transfer across poisoning modalities. Our detailed analysis using attention visualization shows that backdoored models focus on semantically sensitive regions under mismatched conditions while maintaining normal behavior on clean inputs. To mitigate the attack, we try two defense strategies based on system prompt and supervised fine-tuning but find that both of them fail to mitigate the semantic backdoor. Our findings highlight the urgent need to address semantic vulnerabilities in VLMs for their safer deployment.



## **17. Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models**

cs.LG

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07121v1) [paper-pdf](http://arxiv.org/pdf/2506.07121v1)

**Authors**: Ren-Jian Wang, Ke Xue, Zeyu Qin, Ziniu Li, Sheng Tang, Hao-Tian Li, Shengcai Liu, Chao Qian

**Abstract**: Ensuring safety of large language models (LLMs) is important. Red teaming--a systematic approach to identifying adversarial prompts that elicit harmful responses from target LLMs--has emerged as a crucial safety evaluation method. Within this framework, the diversity of adversarial prompts is essential for comprehensive safety assessments. We find that previous approaches to red-teaming may suffer from two key limitations. First, they often pursue diversity through simplistic metrics like word frequency or sentence embedding similarity, which may not capture meaningful variation in attack strategies. Second, the common practice of training a single attacker model restricts coverage across potential attack styles and risk categories. This paper introduces Quality-Diversity Red-Teaming (QDRT), a new framework designed to address these limitations. QDRT achieves goal-driven diversity through behavior-conditioned training and implements a behavioral replay buffer in an open-ended manner. Additionally, it trains multiple specialized attackers capable of generating high-quality attacks across diverse styles and risk categories. Our empirical evaluation demonstrates that QDRT generates attacks that are both more diverse and more effective against a wide range of target LLMs, including GPT-2, Llama-3, Gemma-2, and Qwen2.5. This work advances the field of LLM safety by providing a systematic and effective approach to automated red-teaming, ultimately supporting the responsible deployment of LLMs.



## **18. HauntAttack: When Attack Follows Reasoning as a Shadow**

cs.CR

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07031v1) [paper-pdf](http://arxiv.org/pdf/2506.07031v1)

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Lei Sha, Zhifang Sui

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing exceptional capabilities. However, the enhancement of reasoning abilities and the exposure of their internal reasoning processes introduce new safety vulnerabilities. One intriguing concern is: when reasoning is strongly entangled with harmfulness, what safety-reasoning trade-off do LRMs exhibit? To address this issue, we introduce HauntAttack, a novel and general-purpose black-box attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we treat reasoning questions as carriers and substitute one of their original conditions with a harmful instruction. This process creates a reasoning pathway in which the model is guided step by step toward generating unsafe outputs. Based on HauntAttack, we conduct comprehensive experiments on multiple LRMs. Our results reveal that even the most advanced LRMs exhibit significant safety vulnerabilities. Additionally, we perform a detailed analysis of different models, various types of harmful instructions, and model output patterns, providing valuable insights into the security of LRMs.



## **19. sudo rm -rf agentic_security**

cs.CL

Accepted ACL 2025 Industry track

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2503.20279v3) [paper-pdf](http://arxiv.org/pdf/2503.20279v3)

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal-trained safeguards in commercial computer-use agents, such as Claude for Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24.41% (with no refinement), and up to 41.33% (by its iterative refinement) in Claude for Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs



## **20. Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text**

cs.CL

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07001v1) [paper-pdf](http://arxiv.org/pdf/2506.07001v1)

**Authors**: Yize Cheng, Vinu Sankar Sadasivan, Mehrdad Saberi, Shoumik Saha, Soheil Feizi

**Abstract**: The increasing capabilities of Large Language Models (LLMs) have raised concerns about their misuse in AI-generated plagiarism and social engineering. While various AI-generated text detectors have been proposed to mitigate these risks, many remain vulnerable to simple evasion techniques such as paraphrasing. However, recent detectors have shown greater robustness against such basic attacks. In this work, we introduce Adversarial Paraphrasing, a training-free attack framework that universally humanizes any AI-generated text to evade detection more effectively. Our approach leverages an off-the-shelf instruction-following LLM to paraphrase AI-generated content under the guidance of an AI text detector, producing adversarial examples that are specifically optimized to bypass detection. Extensive experiments show that our attack is both broadly effective and highly transferable across several detection systems. For instance, compared to simple paraphrasing attack--which, ironically, increases the true positive at 1% false positive (T@1%F) by 8.57% on RADAR and 15.03% on Fast-DetectGPT--adversarial paraphrasing, guided by OpenAI-RoBERTa-Large, reduces T@1%F by 64.49% on RADAR and a striking 98.96% on Fast-DetectGPT. Across a diverse set of detectors--including neural network-based, watermark-based, and zero-shot approaches--our attack achieves an average T@1%F reduction of 87.88% under the guidance of OpenAI-RoBERTa-Large. We also analyze the tradeoff between text quality and attack success to find that our method can significantly reduce detection rates, with mostly a slight degradation in text quality. Our adversarial setup highlights the need for more robust and resilient detection strategies in the light of increasingly sophisticated evasion techniques.



## **21. Robustifying Vision-Language Models via Dynamic Token Reweighting**

cs.CV

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2505.17132v2) [paper-pdf](http://arxiv.org/pdf/2505.17132v2)

**Authors**: Tanqiu Jiang, Jiacheng Liang, Rongyi Zhu, Jiawei Zhou, Fenglong Ma, Ting Wang

**Abstract**: Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. (warning: this paper contains potentially harmful content generated by VLMs.)



## **22. LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models**

cs.LG

Accepted as a regular paper at IEEE TITS 2025

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2501.15850v2) [paper-pdf](http://arxiv.org/pdf/2501.15850v2)

**Authors**: Yuewen Mei, Tong Nie, Jian Sun, Ye Tian

**Abstract**: Ensuring and improving the safety of autonomous driving systems (ADS) is crucial for the deployment of highly automated vehicles, especially in safety-critical events. To address the rarity issue, adversarial scenario generation methods are developed, in which behaviors of traffic participants are manipulated to induce safety-critical events. However, existing methods still face two limitations. First, identification of the adversarial participant directly impacts the effectiveness of the generation. However, the complexity of real-world scenarios, with numerous participants and diverse behaviors, makes identification challenging. Second, the potential of generated safety-critical scenarios to continuously improve ADS performance remains underexplored. To address these issues, we propose LLM-attacker: a closed-loop adversarial scenario generation framework leveraging large language models (LLMs). Specifically, multiple LLM agents are designed and coordinated to identify optimal attackers. Then, the trajectories of the attackers are optimized to generate adversarial scenarios. These scenarios are iteratively refined based on the performance of ADS, forming a feedback loop to improve ADS. Experimental results show that LLM-attacker can create more dangerous scenarios than other methods, and the ADS trained with it achieves a collision rate half that of training with normal scenarios. This indicates the ability of LLM-attacker to test and enhance the safety and robustness of ADS. Video demonstrations are provided at: https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view.



## **23. Refining Adaptive Zeroth-Order Optimization at Ease**

cs.LG

Published as a conference paper at ICML 2025

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2502.01014v2) [paper-pdf](http://arxiv.org/pdf/2502.01014v2)

**Authors**: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai

**Abstract**: Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (a) the first analysis to the variance reduction of first moment estimate in ZO optimization, (b) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (c) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (d) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges.



## **24. RED QUEEN: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking**

cs.CR

Accepted in ACL 2025 Findings

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2409.17458v2) [paper-pdf](http://arxiv.org/pdf/2409.17458v2)

**Authors**: Yifan Jiang, Kriti Aggarwal, Tanmay Laud, Kashif Munir, Jay Pujara, Subhabrata Mukherjee

**Abstract**: The rapid progress of Large Language Models (LLMs) has opened up new opportunities across various domains and applications; yet it also presents challenges related to potential misuse. To mitigate such risks, red teaming has been employed as a proactive security measure to probe language models for harmful outputs via jailbreak attacks. However, current jailbreak attack approaches are single-turn with explicit malicious queries that do not fully capture the complexity of real-world interactions. In reality, users can engage in multi-turn interactions with LLM-based chat assistants, allowing them to conceal their true intentions in a more covert manner. To bridge this gap, we, first, propose a new jailbreak approach, RED QUEEN ATTACK. This method constructs a multi-turn scenario, concealing the malicious intent under the guise of preventing harm. We craft 40 scenarios that vary in turns and select 14 harmful categories to generate 56k multi-turn attack data points. We conduct comprehensive experiments on the RED QUEEN ATTACK with four representative LLM families of different sizes. Our experiments reveal that all LLMs are vulnerable to RED QUEEN ATTACK, reaching 87.62% attack success rate on GPT-4o and 75.4% on Llama3-70B. Further analysis reveals that larger models are more susceptible to the RED QUEEN ATTACK, with multi-turn structures and concealment strategies contributing to its success. To prioritize safety, we introduce a straightforward mitigation strategy called RED QUEEN GUARD, which aligns LLMs to effectively counter adversarial attacks. This approach reduces the attack success rate to below 1% while maintaining the model's performance across standard benchmarks. Full implementation and dataset are publicly accessible at https://github.com/kriti-hippo/red_queen.



## **25. from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors**

cs.CL

arXiv admin note: substantial text overlap with arXiv:2412.12145

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2503.00038v3) [paper-pdf](http://arxiv.org/pdf/2503.00038v3)

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Jiangyu Lei, Qi Li

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs.



## **26. Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks: Theoretical and Empirical Evidence**

cs.LG

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2502.04204v2) [paper-pdf](http://arxiv.org/pdf/2502.04204v2)

**Authors**: Shaopeng Fu, Liang Ding, Jingfeng Zhang, Di Wang

**Abstract**: Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. While long-length adversarial prompts during AT might lead to strong LLM robustness, their synthesis however is very resource-consuming, which may limit the application of LLM AT. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the numbers of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix length during jailbreaking to the length during AT. Our findings show that it is practical to defend against ``long-length'' jailbreak attacks via efficient ``short-length'' AT. The code is available at https://github.com/fshp971/adv-icl.



## **27. GraphRAG under Fire**

cs.LG

13 pages

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2501.14050v3) [paper-pdf](http://arxiv.org/pdf/2501.14050v3)

**Authors**: Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, Ting Wang

**Abstract**: GraphRAG advances retrieval-augmented generation (RAG) by structuring external knowledge as multi-scale knowledge graphs, enabling language models to integrate both broad context and granular details in their generation. While GraphRAG has demonstrated success across domains, its security implications remain largely unexplored. To bridge this gap, this work examines GraphRAG's vulnerability to poisoning attacks, uncovering an intriguing security paradox: existing RAG poisoning attacks are less effective under GraphRAG than conventional RAG, due to GraphRAG's graph-based indexing and retrieval; yet, the same features also create new attack surfaces. We present GragPoison, a novel attack that exploits shared relations in the underlying knowledge graph to craft poisoning text capable of compromising multiple queries simultaneously. GragPoison employs three key strategies: (i) relation injection to introduce false knowledge, (ii) relation enhancement to amplify poisoning influence, and (iii) narrative generation to embed malicious content within coherent text. Empirical evaluation across diverse datasets and models shows that GragPoison substantially outperforms existing attacks in terms of effectiveness (up to 98% success rate) and scalability (using less than 68% poisoning text) on multiple variations of GraphRAG. We also explore potential defensive measures and their limitations, identifying promising directions for future research.



## **28. JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models**

cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2404.08793v2) [paper-pdf](http://arxiv.org/pdf/2404.08793v2)

**Authors**: Yingchaojie Feng, Zhizhang Chen, Zhining Kang, Sijia Wang, Haoyu Tian, Wei Zhang, Minfeng Zhu, Wei Chen

**Abstract**: The proliferation of large language models (LLMs) has underscored concerns regarding their security vulnerabilities, notably against jailbreak attacks, where adversaries design jailbreak prompts to circumvent safety mechanisms for potential misuse. Addressing these concerns necessitates a comprehensive analysis of jailbreak prompts to evaluate LLMs' defensive capabilities and identify potential weaknesses. However, the complexity of evaluating jailbreak performance and understanding prompt characteristics makes this analysis laborious. We collaborate with domain experts to characterize problems and propose an LLM-assisted framework to streamline the analysis process. It provides automatic jailbreak assessment to facilitate performance evaluation and support analysis of components and keywords in prompts. Based on the framework, we design JailbreakLens, a visual analysis system that enables users to explore the jailbreak performance against the target model, conduct multi-level analysis of prompt characteristics, and refine prompt instances to verify findings. Through a case study, technical evaluations, and expert interviews, we demonstrate our system's effectiveness in helping users evaluate model security and identify model weaknesses.



## **29. A Systematic Review of Poisoning Attacks Against Large Language Models**

cs.CR

28 Pages including number

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06518v1) [paper-pdf](http://arxiv.org/pdf/2506.06518v1)

**Authors**: Neil Fendley, Edward W. Staley, Joshua Carney, William Redman, Marie Chau, Nathan Drenkow

**Abstract**: With the widespread availability of pretrained Large Language Models (LLMs) and their training datasets, concerns about the security risks associated with their usage has increased significantly. One of these security risks is the threat of LLM poisoning attacks where an attacker modifies some part of the LLM training process to cause the LLM to behave in a malicious way. As an emerging area of research, the current frameworks and terminology for LLM poisoning attacks are derived from earlier classification poisoning literature and are not fully equipped for generative LLM settings. We conduct a systematic review of published LLM poisoning attacks to clarify the security implications and address inconsistencies in terminology across the literature. We propose a comprehensive poisoning threat model applicable to categorize a wide range of LLM poisoning attacks. The poisoning threat model includes four poisoning attack specifications that define the logistics and manipulation strategies of an attack as well as six poisoning metrics used to measure key characteristics of an attack. Under our proposed framework, we organize our discussion of published LLM poisoning literature along four critical dimensions of LLM poisoning attacks: concept poisons, stealthy poisons, persistent poisons, and poisons for unique tasks, to better understand the current landscape of security risks.



## **30. PROVSYN: Synthesizing Provenance Graphs for Data Augmentation in Intrusion Detection Systems**

cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06226v1) [paper-pdf](http://arxiv.org/pdf/2506.06226v1)

**Authors**: Yi Huang, Wajih UI Hassan, Yao Guo, Xiangqun Chen, Ding Li

**Abstract**: Provenance graph analysis plays a vital role in intrusion detection, particularly against Advanced Persistent Threats (APTs), by exposing complex attack patterns. While recent systems combine graph neural networks (GNNs) with natural language processing (NLP) to capture structural and semantic features, their effectiveness is limited by class imbalance in real-world data. To address this, we introduce PROVSYN, an automated framework that synthesizes provenance graphs through a three-phase pipeline: (1) heterogeneous graph structure synthesis with structural-semantic modeling, (2) rule-based topological refinement, and (3) context-aware textual attribute synthesis using large language models (LLMs). PROVSYN includes a comprehensive evaluation framework that integrates structural, textual, temporal, and embedding-based metrics, along with a semantic validation mechanism to assess the correctness of generated attack patterns and system behaviors. To demonstrate practical utility, we use the synthetic graphs to augment training datasets for downstream APT detection models. Experimental results show that PROVSYN produces high-fidelity graphs and improves detection performance through effective data augmentation.



## **31. Joint-GCG: Unified Gradient-Based Poisoning Attacks on Retrieval-Augmented Generation Systems**

cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06151v1) [paper-pdf](http://arxiv.org/pdf/2506.06151v1)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by retrieving relevant documents from external corpora before generating responses. This approach significantly expands LLM capabilities by leveraging vast, up-to-date external knowledge. However, this reliance on external knowledge makes RAG systems vulnerable to corpus poisoning attacks that manipulate generated outputs via poisoned document injection. Existing poisoning attack strategies typically treat the retrieval and generation stages as disjointed, limiting their effectiveness. We propose Joint-GCG, the first framework to unify gradient-based attacks across both retriever and generator models through three innovations: (1) Cross-Vocabulary Projection for aligning embedding spaces, (2) Gradient Tokenization Alignment for synchronizing token-level gradient signals, and (3) Adaptive Weighted Fusion for dynamically balancing attacking objectives. Evaluations demonstrate that Joint-GCG achieves at most 25% and an average of 5% higher attack success rate than previous methods across multiple retrievers and generators. While optimized under a white-box assumption, the generated poisons show unprecedented transferability to unseen models. Joint-GCG's innovative unification of gradient-based attacks across retrieval and generation stages fundamentally reshapes our understanding of vulnerabilities within RAG systems. Our code is available at https://github.com/NicerWang/Joint-GCG.



## **32. The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text**

cs.CL

42nd International Conference on Machine Learning (ICML 2025)

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2502.14921v2) [paper-pdf](http://arxiv.org/pdf/2502.14921v2)

**Authors**: Matthieu Meeus, Lukas Wutschitz, Santiago Zanella-Béguelin, Shruti Tople, Reza Shokri

**Abstract**: How much information about training samples can be leaked through synthetic data generated by Large Language Models (LLMs)? Overlooking the subtleties of information flow in synthetic data generation pipelines can lead to a false sense of privacy. In this paper, we assume an adversary has access to some synthetic data generated by a LLM. We design membership inference attacks (MIAs) that target the training data used to fine-tune the LLM that is then used to synthesize data. The significant performance of our MIA shows that synthetic data leak information about the training data. Further, we find that canaries crafted for model-based MIAs are sub-optimal for privacy auditing when only synthetic data is released. Such out-of-distribution canaries have limited influence on the model's output when prompted to generate useful, in-distribution synthetic data, which drastically reduces their effectiveness. To tackle this problem, we leverage the mechanics of auto-regressive models to design canaries with an in-distribution prefix and a high-perplexity suffix that leave detectable traces in synthetic data. This enhances the power of data-based MIAs and provides a better assessment of the privacy risks of releasing synthetic data generated by LLMs.



## **33. Simple Yet Effective: Extracting Private Data Across Clients in Federated Fine-Tuning of Large Language Models**

cs.CL

10 pages, 4 figures

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06060v1) [paper-pdf](http://arxiv.org/pdf/2506.06060v1)

**Authors**: Yingqi Hu, Zhuo Zhang, Jingyuan Zhang, Lizhen Qu, Zenglin Xu

**Abstract**: Federated fine-tuning of large language models (FedLLMs) presents a promising approach for achieving strong model performance while preserving data privacy in sensitive domains. However, the inherent memorization ability of LLMs makes them vulnerable to training data extraction attacks. To investigate this risk, we introduce simple yet effective extraction attack algorithms specifically designed for FedLLMs. In contrast to prior "verbatim" extraction attacks, which assume access to fragments from all training data, our approach operates under a more realistic threat model, where the attacker only has access to a single client's data and aims to extract previously unseen personally identifiable information (PII) from other clients. This requires leveraging contextual prefixes held by the attacker to generalize across clients. To evaluate the effectiveness of our approaches, we propose two rigorous metrics-coverage rate and efficiency-and extend a real-world legal dataset with PII annotations aligned with CPIS, GDPR, and CCPA standards, achieving 89.9% human-verified precision. Experimental results show that our method can extract up to 56.57% of victim-exclusive PII, with "Address," "Birthday," and "Name" being the most vulnerable categories. Our findings underscore the pressing need for robust defense strategies and contribute a new benchmark and evaluation framework for future research in privacy-preserving federated learning.



## **34. FDLLM: A Dedicated Detector for Black-Box LLMs Fingerprinting**

cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2501.16029v2) [paper-pdf](http://arxiv.org/pdf/2501.16029v2)

**Authors**: Zhiyuan Fu, Junfan Chen, Lan Zhang, Ting Yang, Jun Niu, Hongyu Sun, Ruidong Li, Peng Liu, Yuqing Zhang

**Abstract**: Large Language Models (LLMs) are rapidly transforming the landscape of digital content creation. However, the prevalent black-box Application Programming Interface (API) access to many LLMs introduces significant challenges in accountability, governance, and security. LLM fingerprinting, which aims to identify the source model by analyzing statistical and stylistic features of generated text, offers a potential solution. Current progress in this area is hindered by a lack of dedicated datasets and the need for efficient, practical methods that are robust against adversarial manipulations. To address these challenges, we introduce FD-Dataset, a comprehensive bilingual fingerprinting benchmark comprising 90,000 text samples from 20 famous proprietary and open-source LLMs. Furthermore, we present FDLLM, a novel fingerprinting method that leverages parameter-efficient Low-Rank Adaptation (LoRA) to fine-tune a foundation model. This approach enables LoRA to extract deep, persistent features that characterize each source LLM. Through our analysis, we find that LoRA adaptation promotes the aggregation of outputs from the same LLM in representation space while enhancing the separation between different LLMs. This mechanism explains why LoRA proves particularly effective for LLM fingerprinting. Extensive empirical evaluations on FD-Dataset demonstrate FDLLM's superiority, achieving a Macro F1 score 22.1% higher than the strongest baseline. FDLLM also exhibits strong generalization to newly released models, achieving an average accuracy of 95% on unseen models. Notably, FDLLM remains consistently robust under various adversarial attacks, including polishing, translation, and synonym substitution. Experimental results show that FDLLM reduces the average attack success rate from 49.2% (LM-D) to 23.9%.



## **35. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

cs.CV

31 pages, 8 figures

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.05982v1) [paper-pdf](http://arxiv.org/pdf/2506.05982v1)

**Authors**: Zonglin Wu, Yule Xue, Xin Wei, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.



## **36. UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning**

cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2503.01908v2) [paper-pdf](http://arxiv.org/pdf/2503.01908v2)

**Authors**: Jiawei Zhang, Shuang Yang, Bo Li

**Abstract**: Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements amplify the risks of adversarial attacks, especially when agents can access sensitive external functionalities. Nevertheless, manipulating LLM agents into performing targeted malicious actions or invoking specific tools remains challenging, as these agents extensively reason or plan before executing final actions. In this work, we present UDora, a unified red teaming framework designed for LLM agents that dynamically hijacks the agent's reasoning processes to compel malicious behavior. Specifically, UDora first generates the model's reasoning trace for the given task, then automatically identifies optimal points within this trace to insert targeted perturbations. The resulting perturbed reasoning is then used as a surrogate response for optimization. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets. The code is available at https://github.com/AI-secure/UDora.



## **37. PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning**

cs.CR

Accepted at ICML 2025. Tingchen Fu and Fazl Barez are core research  contributors

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2410.08811v2) [paper-pdf](http://arxiv.org/pdf/2410.08811v2)

**Authors**: Tingchen Fu, Mrinank Sharma, Philip Torr, Shay B. Cohen, David Krueger, Fazl Barez

**Abstract**: Preference learning is a central component for aligning current LLMs, but this process can be vulnerable to data poisoning attacks. To address this concern, we introduce PoisonBench, a benchmark for evaluating large language models' susceptibility to data poisoning during preference learning. Data poisoning attacks can manipulate large language model responses to include hidden malicious content or biases, potentially causing the model to generate harmful or unintended outputs while appearing to function normally. We deploy two distinct attack types across eight realistic scenarios, assessing 21 widely-used models. Our findings reveal concerning trends: (1) Scaling up parameter size does not inherently enhance resilience against poisoning attacks; (2) There exists a log-linear relationship between the effects of the attack and the data poison ratio; (3) The effect of data poisoning can generalize to extrapolated triggers that are not included in the poisoned data. These results expose weaknesses in current preference learning techniques, highlighting the urgent need for more robust defenses against malicious models and data manipulation.



## **38. Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models**

cs.SD

Accepted by ACL 2025 Findings

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2411.14842v2) [paper-pdf](http://arxiv.org/pdf/2411.14842v2)

**Authors**: Wanqi Yang, Yanda Li, Meng Fang, Yunchao Wei, Ling Chen

**Abstract**: Adversarial audio attacks pose a significant threat to the growing use of large audio-language models (LALMs) in voice-based human-machine interactions. While existing research focused on model-specific adversarial methods, real-world applications demand a more generalizable and universal approach to audio adversarial attacks. In this paper, we introduce the Chat-Audio Attacks (CAA) benchmark including four distinct types of audio attacks, which aims to explore the vulnerabilities of LALMs to these audio attacks in conversational scenarios. To evaluate the robustness of LALMs, we propose three evaluation strategies: Standard Evaluation, utilizing traditional metrics to quantify model performance under attacks; GPT-4o-Based Evaluation, which simulates real-world conversational complexities; and Human Evaluation, offering insights into user perception and trust. We evaluate six state-of-the-art LALMs with voice interaction capabilities, including Gemini-1.5-Pro, GPT-4o, and others, using three distinct evaluation methods on the CAA benchmark. Our comprehensive analysis reveals the impact of four types of audio attacks on the performance of these models, demonstrating that GPT-4o exhibits the highest level of resilience. Our data can be accessed via the following link: \href{https://github.com/crystraldo/CAA}{CAA}.



## **39. TracLLM: A Generic Framework for Attributing Long Context LLMs**

cs.CR

To appear in USENIX Security Symposium 2025. The code and data are  at: https://github.com/Wang-Yanting/TracLLM

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.04202v2) [paper-pdf](http://arxiv.org/pdf/2506.04202v2)

**Authors**: Yanting Wang, Wei Zou, Runpeng Geng, Jinyuan Jia

**Abstract**: Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: https://github.com/Wang-Yanting/TracLLM.



## **40. FedShield-LLM: A Secure and Scalable Federated Fine-Tuned Large Language Model**

cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.05640v1) [paper-pdf](http://arxiv.org/pdf/2506.05640v1)

**Authors**: Md Jueal Mia, M. Hadi Amini

**Abstract**: Federated Learning (FL) offers a decentralized framework for training and fine-tuning Large Language Models (LLMs) by leveraging computational resources across organizations while keeping sensitive data on local devices. It addresses privacy and security concerns while navigating challenges associated with the substantial computational demands of LLMs, which can be prohibitive for small and medium-sized organizations. FL supports the development of task-specific LLMs for cross-silo applications through fine-tuning but remains vulnerable to inference attacks, such as membership inference and gradient inversion, which threaten data privacy. Prior studies have utilized Differential Privacy (DP) in LLM fine-tuning, which, despite being effective at preserving privacy, can degrade model performance. To overcome these challenges, we propose a novel method, FedShield-LLM, that uses pruning with Fully Homomorphic Encryption (FHE) for Low-Rank Adaptation (LoRA) parameters, enabling secure computations on encrypted model updates while mitigating the attack surface by deactivating less important LoRA parameters. Furthermore, optimized federated algorithms for cross-silo environments enhance scalability and efficiency. Parameter-efficient fine-tuning techniques like LoRA substantially reduce computational and communication overhead, making FL feasible for resource-constrained clients. Experimental results show that the proposed method outperforms existing methods while maintaining robust privacy protection, enabling organizations to collaboratively train secure and efficient LLMs.   The code and data are available at, https://github.com/solidlabnetwork/fedshield-llm



## **41. Comprehensive Vulnerability Analysis is Necessary for Trustworthy LLM-MAS**

cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.01245v2) [paper-pdf](http://arxiv.org/pdf/2506.01245v2)

**Authors**: Pengfei He, Yue Xing, Shen Dong, Juanhui Li, Zhenwei Dai, Xianfeng Tang, Hui Liu, Han Xu, Zhen Xiang, Charu C. Aggarwal, Hui Liu

**Abstract**: This paper argues that a comprehensive vulnerability analysis is essential for building trustworthy Large Language Model-based Multi-Agent Systems (LLM-MAS). These systems, which consist of multiple LLM-powered agents working collaboratively, are increasingly deployed in high-stakes applications but face novel security threats due to their complex structures. While single-agent vulnerabilities are well-studied, LLM-MAS introduces unique attack surfaces through inter-agent communication, trust relationships, and tool integration that remain significantly underexplored. We present a systematic framework for vulnerability analysis of LLM-MAS that unifies diverse research. For each type of vulnerability, we define formal threat models grounded in practical attacker capabilities and illustrate them using real-world LLM-MAS applications. This formulation enables rigorous quantification of vulnerability across different architectures and provides a foundation for designing meaningful evaluation benchmarks. Our analysis reveals that LLM-MAS faces elevated risk due to compositional effects -- vulnerabilities in individual components can cascade through agent communication, creating threat models not present in single-agent systems. We conclude by identifying critical open challenges: (1) developing benchmarks specifically tailored to LLM-MAS vulnerability assessment, (2) considering new potential attacks specific to multi-agent architectures, and (3) implementing trust management systems that can enforce security in LLM-MAS. This research provides essential groundwork for future efforts to enhance LLM-MAS trustworthiness as these systems continue their expansion into critical applications.



## **42. SoK: Are Watermarks in LLMs Ready for Deployment?**

cs.CR

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.05594v1) [paper-pdf](http://arxiv.org/pdf/2506.05594v1)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah, My Thai

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs.   To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.



## **43. Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets**

cs.CR

Project Page: https://hsiung.cc/llm-similarity-risk/

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.05346v1) [paper-pdf](http://arxiv.org/pdf/2506.05346v1)

**Authors**: Lei Hsiung, Tianyu Pang, Yung-Chen Tang, Linyue Song, Tsung-Yi Ho, Pin-Yu Chen, Yaoqing Yang

**Abstract**: Recent advancements in large language models (LLMs) have underscored their vulnerability to safety alignment jailbreaks, particularly when subjected to downstream fine-tuning. However, existing mitigation strategies primarily focus on reactively addressing jailbreak incidents after safety guardrails have been compromised, removing harmful gradients during fine-tuning, or continuously reinforcing safety alignment throughout fine-tuning. As such, they tend to overlook a critical upstream factor: the role of the original safety-alignment data. This paper therefore investigates the degradation of safety guardrails through the lens of representation similarity between upstream alignment datasets and downstream fine-tuning tasks. Our experiments demonstrate that high similarity between these datasets significantly weakens safety guardrails, making models more susceptible to jailbreaks. Conversely, low similarity between these two types of datasets yields substantially more robust models and thus reduces harmfulness score by up to 10.33%. By highlighting the importance of upstream dataset design in the building of durable safety guardrails and reducing real-world vulnerability to jailbreak attacks, these findings offer actionable insights for fine-tuning service providers.



## **44. Beyond the Protocol: Unveiling Attack Vectors in the Model Context Protocol Ecosystem**

cs.CR

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.02040v2) [paper-pdf](http://arxiv.org/pdf/2506.02040v2)

**Authors**: Hao Song, Yiming Shen, Wenxuan Luo, Leixin Guo, Ting Chen, Jiashui Wang, Beibei Li, Xiaosong Zhang, Jiachi Chen

**Abstract**: The Model Context Protocol (MCP) is an emerging standard designed to enable seamless interaction between Large Language Model (LLM) applications and external tools or resources. Within a short period, thousands of MCP services have already been developed and deployed. However, the client-server integration architecture inherent in MCP may expand the attack surface against LLM Agent systems, introducing new vulnerabilities that allow attackers to exploit by designing malicious MCP servers. In this paper, we present the first systematic study of attack vectors targeting the MCP ecosystem. Our analysis identifies four categories of attacks, i.e., Tool Poisoning Attacks, Puppet Attacks, Rug Pull Attacks, and Exploitation via Malicious External Resources. To evaluate the feasibility of these attacks, we conduct experiments following the typical steps of launching an attack through malicious MCP servers: upload-download-attack. Specifically, we first construct malicious MCP servers and successfully upload them to three widely used MCP aggregation platforms. The results indicate that current audit mechanisms are insufficient to identify and prevent the proposed attack methods. Next, through a user study and interview with 20 participants, we demonstrate that users struggle to identify malicious MCP servers and often unknowingly install them from aggregator platforms. Finally, we demonstrate that these attacks can trigger harmful behaviors within the user's local environment-such as accessing private files or controlling devices to transfer digital assets-by deploying a proof-of-concept (PoC) framework against five leading LLMs. Additionally, based on interview results, we discuss four key challenges faced by the current security ecosystem surrounding MCP servers. These findings underscore the urgent need for robust security mechanisms to defend against malicious MCP servers.



## **45. DREAM: Disentangling Risks to Enhance Safety Alignment in Multimodal Large Language Models**

cs.CL

[NAACL 2025] The first four authors contribute equally, 23 pages,  repo at https://github.com/Kizna1ver/DREAM

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2504.18053v2) [paper-pdf](http://arxiv.org/pdf/2504.18053v2)

**Authors**: Jianyu Liu, Hangyu Guo, Ranjie Duan, Xingyuan Bu, Yancheng He, Shilong Li, Hui Huang, Jiaheng Liu, Yucheng Wang, Chenchen Jing, Xingwei Qu, Xiao Zhang, Yingshui Tan, Yanan Wu, Jihao Gu, Yangguang Li, Jianke Zhu

**Abstract**: Multimodal Large Language Models (MLLMs) pose unique safety challenges due to their integration of visual and textual data, thereby introducing new dimensions of potential attacks and complex risk combinations. In this paper, we begin with a detailed analysis aimed at disentangling risks through step-by-step reasoning within multimodal inputs. We find that systematic multimodal risk disentanglement substantially enhances the risk awareness of MLLMs. Via leveraging the strong discriminative abilities of multimodal risk disentanglement, we further introduce \textbf{DREAM} (\textit{\textbf{D}isentangling \textbf{R}isks to \textbf{E}nhance Safety \textbf{A}lignment in \textbf{M}LLMs}), a novel approach that enhances safety alignment in MLLMs through supervised fine-tuning and iterative Reinforcement Learning from AI Feedback (RLAIF). Experimental results show that DREAM significantly boosts safety during both inference and training phases without compromising performance on normal tasks (namely oversafety), achieving a 16.17\% improvement in the SIUO safe\&effective score compared to GPT-4V. The data and code are available at https://github.com/Kizna1ver/DREAM.



## **46. Membership Inference Attacks on Sequence Models**

cs.CR

Accepted to the 8th Deep Learning Security and Privacy Workshop  (DLSP) workshop (best paper award)

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.05126v1) [paper-pdf](http://arxiv.org/pdf/2506.05126v1)

**Authors**: Lorenzo Rossi, Michael Aerni, Jie Zhang, Florian Tramèr

**Abstract**: Sequence models, such as Large Language Models (LLMs) and autoregressive image generators, have a tendency to memorize and inadvertently leak sensitive information. While this tendency has critical legal implications, existing tools are insufficient to audit the resulting risks. We hypothesize that those tools' shortcomings are due to mismatched assumptions. Thus, we argue that effectively measuring privacy leakage in sequence models requires leveraging the correlations inherent in sequential generation. To illustrate this, we adapt a state-of-the-art membership inference attack to explicitly model within-sequence correlations, thereby demonstrating how a strong existing attack can be naturally extended to suit the structure of sequence models. Through a case study, we show that our adaptations consistently improve the effectiveness of memorization audits without introducing additional computational costs. Our work hence serves as an important stepping stone toward reliable memorization audits for large sequence models.



## **47. Seven Security Challenges That Must be Solved in Cross-domain Multi-agent LLM Systems**

cs.CR

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2505.23847v2) [paper-pdf](http://arxiv.org/pdf/2505.23847v2)

**Authors**: Ronny Ko, Jiseong Jeong, Shuyuan Zheng, Chuan Xiao, Tae-Wan Kim, Makoto Onizuka, Won-Yong Shin

**Abstract**: Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines.



## **48. Sentinel: SOTA model to protect against prompt injections**

cs.CR

6 pages, 2 tables

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.05446v1) [paper-pdf](http://arxiv.org/pdf/2506.05446v1)

**Authors**: Dror Ivry, Oran Nahum

**Abstract**: Large Language Models (LLMs) are increasingly powerful but remain vulnerable to prompt injection attacks, where malicious inputs cause the model to deviate from its intended instructions. This paper introduces Sentinel, a novel detection model, qualifire/prompt-injection-sentinel, based on the \answerdotai/ModernBERT-large architecture. By leveraging ModernBERT's advanced features and fine-tuning on an extensive and diverse dataset comprising a few open-source and private collections, Sentinel achieves state-of-the-art performance. This dataset amalgamates varied attack types, from role-playing and instruction hijacking to attempts to generate biased content, alongside a broad spectrum of benign instructions, with private datasets specifically targeting nuanced error correction and real-world misclassifications. On a comprehensive, unseen internal test set, Sentinel demonstrates an average accuracy of 0.987 and an F1-score of 0.980. Furthermore, when evaluated on public benchmarks, it consistently outperforms strong baselines like protectai/deberta-v3-base-prompt-injection-v2. This work details Sentinel's architecture, its meticulous dataset curation, its training methodology, and a thorough evaluation, highlighting its superior detection capabilities.



## **49. Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models**

cs.CL

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2505.23404v2) [paper-pdf](http://arxiv.org/pdf/2505.23404v2)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin

**Abstract**: Adversarial attacks on Large Language Models (LLMs) via jailbreaking techniques-methods that circumvent their built-in safety and ethical constraints-have emerged as a critical challenge in AI security. These attacks compromise the reliability of LLMs by exploiting inherent weaknesses in their comprehension capabilities. This paper investigates the efficacy of jailbreaking strategies that are specifically adapted to the diverse levels of understanding exhibited by different LLMs. We propose the Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models, a novel framework that classifies LLMs into Type I and Type II categories according to their semantic comprehension abilities. For each category, we design tailored jailbreaking strategies aimed at leveraging their vulnerabilities to facilitate successful attacks. Extensive experiments conducted on multiple LLMs demonstrate that our adaptive strategy markedly improves the success rate of jailbreaking. Notably, our approach achieves an exceptional 98.9% success rate in jailbreaking GPT-4o(29 May 2025 release)



## **50. On Automating Security Policies with Contemporary LLMs**

cs.CR

Short Paper. Accepted To Appear in IEEE SSE 2025 (part of SERVICES  2025)

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.04838v1) [paper-pdf](http://arxiv.org/pdf/2506.04838v1)

**Authors**: Pablo Fernández Saura, K. R. Jayaram, Vatche Isahagian, Jorge Bernal Bernabé, Antonio Skarmeta

**Abstract**: The complexity of modern computing environments and the growing sophistication of cyber threats necessitate a more robust, adaptive, and automated approach to security enforcement. In this paper, we present a framework leveraging large language models (LLMs) for automating attack mitigation policy compliance through an innovative combination of in-context learning and retrieval-augmented generation (RAG). We begin by describing how our system collects and manages both tool and API specifications, storing them in a vector database to enable efficient retrieval of relevant information. We then detail the architectural pipeline that first decomposes high-level mitigation policies into discrete tasks and subsequently translates each task into a set of actionable API calls. Our empirical evaluation, conducted using publicly available CTI policies in STIXv2 format and Windows API documentation, demonstrates significant improvements in precision, recall, and F1-score when employing RAG compared to a non-RAG baseline.



