# Latest Large Language Model Attack Papers
**update at 2025-03-15 11:01:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

cs.CV

Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10635v1) [paper-pdf](http://arxiv.org/pdf/2503.10635v1)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at https://github.com/VILA-Lab/M-Attack.



## **2. ASIDE: Architectural Separation of Instructions and Data in Language Models**

cs.LG

ICLR 2025 Workshop on Building Trust in Language Models and  Applications

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10566v1) [paper-pdf](http://arxiv.org/pdf/2503.10566v1)

**Authors**: Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Soroush Tabesh, Alexandra Volkova, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert

**Abstract**: Despite their remarkable performance, large language models lack elementary safety features, and this makes them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as a root cause for the success of prompt injection attacks. In this work, we propose an architectural change, ASIDE, that allows the model to clearly separate between instructions and data by using separate embeddings for them. Instead of training the embeddings from scratch, we propose a method to convert an existing model to ASIDE form by using two copies of the original model's embeddings layer, and applying an orthogonal rotation to one of them. We demonstrate the effectiveness of our method by showing (1) highly increased instruction-data separation scores without a loss in model capabilities and (2) competitive results on prompt injection benchmarks, even without dedicated safety training. Additionally, we study the working mechanism behind our method through an analysis of model representations.



## **3. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors**

cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.08708v2) [paper-pdf](http://arxiv.org/pdf/2503.08708v2)

**Authors**: Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, Xinlei He

**Abstract**: As Large Language Models (LLMs) advance, Machine-Generated Texts (MGTs) have become increasingly fluent, high-quality, and informative. Existing wide-range MGT detectors are designed to identify MGTs to prevent the spread of plagiarism and misinformation. However, adversaries attempt to humanize MGTs to evade detection (named evading attacks), which requires only minor modifications to bypass MGT detectors. Unfortunately, existing attacks generally lack a unified and comprehensive evaluation framework, as they are assessed using different experimental settings, model architectures, and datasets. To fill this gap, we introduce the Text-Humanization Benchmark (TH-Bench), the first comprehensive benchmark to evaluate evading attacks against MGT detectors. TH-Bench evaluates attacks across three key dimensions: evading effectiveness, text quality, and computational overhead. Our extensive experiments evaluate 6 state-of-the-art attacks against 13 MGT detectors across 6 datasets, spanning 19 domains and generated by 11 widely used LLMs. Our findings reveal that no single evading attack excels across all three dimensions. Through in-depth analysis, we highlight the strengths and limitations of different attacks. More importantly, we identify a trade-off among three dimensions and propose two optimization insights. Through preliminary experiments, we validate their correctness and effectiveness, offering potential directions for future research.



## **4. Prompt Inversion Attack against Collaborative Inference of Large Language Models**

cs.CR

To appear at IEEE Symposium on Security and Privacy 2025

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.09022v2) [paper-pdf](http://arxiv.org/pdf/2503.09022v2)

**Authors**: Wenjie Qu, Yuguang Zhou, Yongji Wu, Tingsong Xiao, Binhang Yuan, Yiming Li, Jiaheng Zhang

**Abstract**: Large language models (LLMs) have been widely applied for their remarkable capability of content generation. However, the practical use of open-source LLMs is hindered by high resource requirements, making deployment expensive and limiting widespread development. The collaborative inference is a promising solution for this problem, in which users collaborate by each hosting a subset of layers and transmitting intermediate activation. Many companies are building collaborative inference platforms to reduce LLM serving costs, leveraging users' underutilized GPUs. Despite widespread interest in collaborative inference within academia and industry, the privacy risks associated with LLM collaborative inference have not been well studied. This is largely because of the challenge posed by inverting LLM activation due to its strong non-linearity.   In this paper, to validate the severity of privacy threats in LLM collaborative inference, we introduce the concept of prompt inversion attack (PIA), where a malicious participant intends to recover the input prompt through the activation transmitted by its previous participant. Extensive experiments show that our PIA method substantially outperforms existing baselines. For example, our method achieves an 88.4\% token accuracy on the Skytrax dataset with the Llama-65B model when inverting the maximum number of transformer layers, while the best baseline method only achieves 22.8\% accuracy. The results verify the effectiveness of our PIA attack and highlights its practical threat to LLM collaborative inference systems.



## **5. ExtremeAIGC: Benchmarking LMM Vulnerability to AI-Generated Extremist Content**

cs.CR

Preprint

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.09964v1) [paper-pdf](http://arxiv.org/pdf/2503.09964v1)

**Authors**: Bhavik Chandna, Mariam Aboujenane, Usman Naseem

**Abstract**: Large Multimodal Models (LMMs) are increasingly vulnerable to AI-generated extremist content, including photorealistic images and text, which can be used to bypass safety mechanisms and generate harmful outputs. However, existing datasets for evaluating LMM robustness offer limited exploration of extremist content, often lacking AI-generated images, diverse image generation models, and comprehensive coverage of historical events, which hinders a complete assessment of model vulnerabilities. To fill this gap, we introduce ExtremeAIGC, a benchmark dataset and evaluation framework designed to assess LMM vulnerabilities against such content. ExtremeAIGC simulates real-world events and malicious use cases by curating diverse text- and image-based examples crafted using state-of-the-art image generation techniques. Our study reveals alarming weaknesses in LMMs, demonstrating that even cutting-edge safety measures fail to prevent the generation of extremist material. We systematically quantify the success rates of various attack strategies, exposing critical gaps in current defenses and emphasizing the need for more robust mitigation strategies.



## **6. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

cs.LG

AISTATS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2412.08099v4) [paper-pdf](http://arxiv.org/pdf/2412.08099v4)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like LLMTime with GPT-3.5, GPT-4, LLaMa, and Mistral, TimeGPT, and TimeLLM show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications. The code repository can be found at https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.



## **7. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

cs.CR

The paper is submitted to "The 48th International ACM SIGIR  Conference on Research and Development in Information Retrieval" and is  currently under review

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09334v1) [paper-pdf](http://arxiv.org/pdf/2503.09334v1)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. All scripts required to reproduce the dataset, along with examples and relevant resources for replicating our results, will be made available upon the paper's acceptance.



## **8. Prompt Inference Attack on Distributed Large Language Model Inference Frameworks**

cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09291v1) [paper-pdf](http://arxiv.org/pdf/2503.09291v1)

**Authors**: Xinjian Luo, Ting Yu, Xiaokui Xiao

**Abstract**: The inference process of modern large language models (LLMs) demands prohibitive computational resources, rendering them infeasible for deployment on consumer-grade devices. To address this limitation, recent studies propose distributed LLM inference frameworks, which employ split learning principles to enable collaborative LLM inference on resource-constrained hardware. However, distributing LLM layers across participants requires the transmission of intermediate outputs, which may introduce privacy risks to the original input prompts - a critical issue that has yet to be thoroughly explored in the literature.   In this paper, we rigorously examine the privacy vulnerabilities of distributed LLM inference frameworks by designing and evaluating three prompt inference attacks aimed at reconstructing input prompts from intermediate LLM outputs. These attacks are developed under various query and data constraints to reflect diverse real-world LLM service scenarios. Specifically, the first attack assumes an unlimited query budget and access to an auxiliary dataset sharing the same distribution as the target prompts. The second attack also leverages unlimited queries but uses an auxiliary dataset with a distribution differing from the target prompts. The third attack operates under the most restrictive scenario, with limited query budgets and no auxiliary dataset available. We evaluate these attacks on a range of LLMs, including state-of-the-art models such as Llama-3.2 and Phi-3.5, as well as widely-used models like GPT-2 and BERT for comparative analysis. Our experiments show that the first two attacks achieve reconstruction accuracies exceeding 90%, while the third achieves accuracies typically above 50%, even under stringent constraints. These findings highlight privacy risks in distributed LLM inference frameworks, issuing a strong alert on their deployment in real-world applications.



## **9. In-Context Defense in Computer Agents: An Empirical Study**

cs.AI

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09241v1) [paper-pdf](http://arxiv.org/pdf/2503.09241v1)

**Authors**: Pei Yang, Hai Ci, Mike Zheng Shou

**Abstract**: Computer agents powered by vision-language models (VLMs) have significantly advanced human-computer interaction, enabling users to perform complex tasks through natural language instructions. However, these agents are vulnerable to context deception attacks, an emerging threat where adversaries embed misleading content into the agent's operational environment, such as a pop-up window containing deceptive instructions. Existing defenses, such as instructing agents to ignore deceptive elements, have proven largely ineffective. As the first systematic study on protecting computer agents, we introduce textbf{in-context defense}, leveraging in-context learning and chain-of-thought (CoT) reasoning to counter such attacks. Our approach involves augmenting the agent's context with a small set of carefully curated exemplars containing both malicious environments and corresponding defensive responses. These exemplars guide the agent to first perform explicit defensive reasoning before action planning, reducing susceptibility to deceptive attacks. Experiments demonstrate the effectiveness of our method, reducing attack success rates by 91.2% on pop-up window attacks, 74.6% on average on environment injection attacks, while achieving 100% successful defenses against distracting advertisements. Our findings highlight that (1) defensive reasoning must precede action planning for optimal performance, and (2) a minimal number of exemplars (fewer than three) is sufficient to induce an agent's defensive behavior.



## **10. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2410.23746v3) [paper-pdf](http://arxiv.org/pdf/2410.23746v3)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.



## **11. A Survey on Trustworthy LLM Agents: Threats and Countermeasures**

cs.MA

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09648v1) [paper-pdf](http://arxiv.org/pdf/2503.09648v1)

**Authors**: Miao Yu, Fanci Meng, Xinyun Zhou, Shilong Wang, Junyuan Mao, Linsey Pang, Tianlong Chen, Kun Wang, Xinfeng Li, Yongfeng Zhang, Bo An, Qingsong Wen

**Abstract**: With the rapid evolution of Large Language Models (LLMs), LLM-based agents and Multi-agent Systems (MAS) have significantly expanded the capabilities of LLM ecosystems. This evolution stems from empowering LLMs with additional modules such as memory, tools, environment, and even other agents. However, this advancement has also introduced more complex issues of trustworthiness, which previous research focused solely on LLMs could not cover. In this survey, we propose the TrustAgent framework, a comprehensive study on the trustworthiness of agents, characterized by modular taxonomy, multi-dimensional connotations, and technical implementation. By thoroughly investigating and summarizing newly emerged attacks, defenses, and evaluation methods for agents and MAS, we extend the concept of Trustworthy LLM to the emerging paradigm of Trustworthy Agent. In TrustAgent, we begin by deconstructing and introducing various components of the Agent and MAS. Then, we categorize their trustworthiness into intrinsic (brain, memory, and tool) and extrinsic (user, agent, and environment) aspects. Subsequently, we delineate the multifaceted meanings of trustworthiness and elaborate on the implementation techniques of existing research related to these internal and external modules. Finally, we present our insights and outlook on this domain, aiming to provide guidance for future endeavors.



## **12. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

cs.LG

4 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09066v1) [paper-pdf](http://arxiv.org/pdf/2503.09066v1)

**Authors**: Xin Wei Chia, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.



## **13. JBFuzz: Jailbreaking LLMs Efficiently and Effectively Using Fuzzing**

cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.08990v1) [paper-pdf](http://arxiv.org/pdf/2503.08990v1)

**Authors**: Vasudev Gohil

**Abstract**: Large language models (LLMs) have shown great promise as language understanding and decision making tools, and they have permeated various aspects of our everyday life. However, their widespread availability also comes with novel risks, such as generating harmful, unethical, or offensive content, via an attack called jailbreaking. Despite extensive efforts from LLM developers to align LLMs using human feedback, they are still susceptible to jailbreak attacks. To tackle this issue, researchers often employ red-teaming to understand and investigate jailbreak prompts. However, existing red-teaming approaches lack effectiveness, scalability, or both. To address these issues, we propose JBFuzz, a novel effective, automated, and scalable red-teaming technique for jailbreaking LLMs.   JBFuzz is inspired by the success of fuzzing for detecting bugs/vulnerabilities in software. We overcome three challenges related to effectiveness and scalability by devising novel seed prompts, a lightweight mutation engine, and a lightweight and accurate evaluator for guiding the fuzzer. Assimilating all three solutions results in a potent fuzzer that only requires black-box access to the target LLM. We perform extensive experimental evaluation of JBFuzz using nine popular and widely-used LLMs. We find that JBFuzz successfully jailbreaks all LLMs for various harmful/unethical questions, with an average attack success rate of 99%. We also find that JBFuzz is extremely efficient as it jailbreaks a given LLM for a given question in 60 seconds on average. Our work highlights the susceptibility of the state-of-the-art LLMs to jailbreak attacks even after safety alignment, and serves as a valuable red-teaming tool for LLM developers.



## **14. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

cs.CL

Accepted to NAACL 2025 Main (oral)

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2410.18469v3) [paper-pdf](http://arxiv.org/pdf/2410.18469v3)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.



## **15. Backtracking for Safety**

cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08919v1) [paper-pdf](http://arxiv.org/pdf/2503.08919v1)

**Authors**: Bilgehan Sel, Dingcheng Li, Phillip Wallis, Vaishakh Keshava, Ming Jin, Siddhartha Reddy Jonnalagadda

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various tasks, but ensuring their safety and alignment with human values remains crucial. Current safety alignment methods, such as supervised fine-tuning and reinforcement learning-based approaches, can exhibit vulnerabilities to adversarial attacks and often result in shallow safety alignment, primarily focusing on preventing harmful content in the initial tokens of the generated output. While methods like resetting can help recover from unsafe generations by discarding previous tokens and restarting the generation process, they are not well-suited for addressing nuanced safety violations like toxicity that may arise within otherwise benign and lengthy generations. In this paper, we propose a novel backtracking method designed to address these limitations. Our method allows the model to revert to a safer generation state, not necessarily at the beginning, when safety violations occur during generation. This approach enables targeted correction of problematic segments without discarding the entire generated text, thereby preserving efficiency. We demonstrate that our method dramatically reduces toxicity appearing through the generation process with minimal impact to efficiency.



## **16. Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context**

cs.CL

arXiv admin note: text overlap with arXiv:2407.14644

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.16359v2) [paper-pdf](http://arxiv.org/pdf/2412.16359v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous studies that uncovered vulnerabilities in large language models (LLMs) frequently employed nonsensical adversarial prompts. However, such prompts can now be readily identified using automated detection techniques. To further strengthen adversarial attacks, we focus on human-readable adversarial prompts, which are more realistic and potent threats. Our key contributions are (1) situation-driven attacks leveraging movie scripts as context to create human-readable prompts that successfully deceive LLMs, (2) adversarial suffix conversion to transform nonsensical adversarial suffixes into independent meaningful text, and (3) AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B.



## **17. Proactive Privacy Amnesia for Large Language Models: Safeguarding PII with Negligible Impact on Model Utility**

cs.CL

ICLR'25 Poster. Project page and code is available at  https://ppa-iclr2025.my.canva.site/

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.17591v2) [paper-pdf](http://arxiv.org/pdf/2502.17591v2)

**Authors**: Martin Kuo, Jingyang Zhang, Jianyi Zhang, Minxue Tang, Louis DiValentin, Aolin Ding, Jingwei Sun, William Chen, Amin Hass, Tianlong Chen, Yiran Chen, Hai Li

**Abstract**: With the rise of large language models (LLMs), increasing research has recognized their risk of leaking personally identifiable information (PII) under malicious attacks. Although efforts have been made to protect PII in LLMs, existing methods struggle to balance privacy protection with maintaining model utility. In this paper, inspired by studies of amnesia in cognitive science, we propose a novel approach, Proactive Privacy Amnesia (PPA), to safeguard PII in LLMs while preserving their utility. This mechanism works by actively identifying and forgetting key memories most closely associated with PII in sequences, followed by a memory implanting using suitable substitute memories to maintain the LLM's functionality. We conduct evaluations across multiple models to protect common PII, such as phone numbers and physical addresses, against prevalent PII-targeted attacks, demonstrating the superiority of our method compared with other existing defensive techniques. The results show that our PPA method completely eliminates the risk of phone number exposure by 100% and significantly reduces the risk of physical address exposure by 9.8% - 87.6%, all while maintaining comparable model utility performance.



## **18. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.16750v3) [paper-pdf](http://arxiv.org/pdf/2502.16750v3)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **19. Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation**

cs.CL

17 pages, 10 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08195v1) [paper-pdf](http://arxiv.org/pdf/2503.08195v1)

**Authors**: Wenlong Meng, Fan Zhang, Wendao Yao, Zhenyuan Guo, Yuwei Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness.



## **20. Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models**

cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.11054v4) [paper-pdf](http://arxiv.org/pdf/2502.11054v4)

**Authors**: Zonghao Ying, Deyue Zhang, Zonglei Jing, Yisong Xiao, Quanchen Zou, Aishan Liu, Siyuan Liang, Xiangzheng Zhang, Xianglong Liu, Dacheng Tao

**Abstract**: Multi-turn jailbreak attacks simulate real-world human interactions by engaging large language models (LLMs) in iterative dialogues, exposing critical safety vulnerabilities. However, existing methods often struggle to balance semantic coherence with attack effectiveness, resulting in either benign semantic drift or ineffective detection evasion. To address this challenge, we propose Reasoning-Augmented Conversation, a novel multi-turn jailbreak framework that reformulates harmful queries into benign reasoning tasks and leverages LLMs' strong reasoning capabilities to compromise safety alignment. Specifically, we introduce an attack state machine framework to systematically model problem translation and iterative reasoning, ensuring coherent query generation across multiple turns. Building on this framework, we design gain-guided exploration, self-play, and rejection feedback modules to preserve attack semantics, enhance effectiveness, and sustain reasoning-driven attack progression. Extensive experiments on multiple LLMs demonstrate that RACE achieves state-of-the-art attack effectiveness in complex conversational scenarios, with attack success rates (ASRs) increasing by up to 96%. Notably, our approach achieves ASRs of 82% and 92% against leading commercial models, OpenAI o1 and DeepSeek R1, underscoring its potency. We release our code at https://github.com/NY1024/RACE to facilitate further research in this critical domain.



## **21. Safety Guardrails for LLM-Enabled Robots**

cs.RO

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07885v1) [paper-pdf](http://arxiv.org/pdf/2503.07885v1)

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots.



## **22. PoisonedParrot: Subtle Data Poisoning Attacks to Elicit Copyright-Infringing Content from Large Language Models**

cs.LG

18 pages, 18 figures. Accepted at NAACL 2025

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07697v1) [paper-pdf](http://arxiv.org/pdf/2503.07697v1)

**Authors**: Michael-Andrei Panaitescu-Liess, Pankayaraj Pathmanathan, Yigitcan Kaya, Zora Che, Bang An, Sicheng Zhu, Aakriti Agrawal, Furong Huang

**Abstract**: As the capabilities of large language models (LLMs) continue to expand, their usage has become increasingly prevalent. However, as reflected in numerous ongoing lawsuits regarding LLM-generated content, addressing copyright infringement remains a significant challenge. In this paper, we introduce PoisonedParrot: the first stealthy data poisoning attack that induces an LLM to generate copyrighted content even when the model has not been directly trained on the specific copyrighted material. PoisonedParrot integrates small fragments of copyrighted text into the poison samples using an off-the-shelf LLM. Despite its simplicity, evaluated in a wide range of experiments, PoisonedParrot is surprisingly effective at priming the model to generate copyrighted content with no discernible side effects. Moreover, we discover that existing defenses are largely ineffective against our attack. Finally, we make the first attempt at mitigating copyright-infringement poisoning attacks by proposing a defense: ParrotTrap. We encourage the community to explore this emerging threat model further.



## **23. The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective**

cs.LG

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2405.16918v2) [paper-pdf](http://arxiv.org/pdf/2405.16918v2)

**Authors**: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

**Abstract**: Flatness of the loss surface not only correlates positively with generalization, but is also related to adversarial robustness since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer. We observe a peculiar property of adversarial examples in the context of relative flatness: during an iterative first-order white-box attack, the flatness of the loss surface measured around the adversarial example first becomes sharper until the label is flipped, but if we keep the attack running, it runs into a flat uncanny valley where the label remains flipped. In extensive experiments, we observe this phenomenon across various model architectures and datasets, even for adversarially trained models. Our results also extend to large language models (LLMs), but due to the discrete nature of the input space and comparatively weak attacks, adversarial examples rarely reach truly flat regions. Most importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also guarantee the behavior of the function around the examples. We, therefore theoretically connect relative flatness to adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in combination with a low global Lipschitz constant for a robust model.



## **24. Utilizing Jailbreak Probability to Attack and Safeguard Multimodal LLMs**

cs.CR

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06989v1) [paper-pdf](http://arxiv.org/pdf/2503.06989v1)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Deyue Zhang, Dongdong Yang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal contents. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on inputs to maximize jailbreak probability. To counteract attacks, we also propose two defensive methods: Jailbreak-Probability-based Finetuning (JPF) and Jailbreak-Probability-based Defensive Noise (JPDN), which minimizes jailbreak probability in the MLLM parameters and input space, respectively. Extensive experiments show that (1) JPA yields improvements (up to 28.38\%) under both white and black box settings compared to previous methods with small perturbation bounds and few iterations. (2) JPF and JPDN significantly reduce jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.



## **25. InferDPT: Privacy-Preserving Inference for Black-box Large Language Model**

cs.CR

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2310.12214v7) [paper-pdf](http://arxiv.org/pdf/2310.12214v7)

**Authors**: Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Zhikun Zhang

**Abstract**: Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.



## **26. Stepwise Reasoning Error Disruption Attack of LLMs**

cs.AI

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2412.11934v3) [paper-pdf](http://arxiv.org/pdf/2412.11934v3)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications.



## **27. Can Watermarking Large Language Models Prevent Copyrighted Text Generation and Hide Training Data?**

cs.LG

19 pages, 7 figures. Published at AAAI 2025. Code will be available  at https://github.com/michael-panaitescu/watermark_copyright_aaai25

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2407.17417v2) [paper-pdf](http://arxiv.org/pdf/2407.17417v2)

**Authors**: Michael-Andrei Panaitescu-Liess, Zora Che, Bang An, Yuancheng Xu, Pankayaraj Pathmanathan, Souradip Chakraborty, Sicheng Zhu, Tom Goldstein, Furong Huang

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in generating diverse and contextually rich text. However, concerns regarding copyright infringement arise as LLMs may inadvertently produce copyrighted material. In this paper, we first investigate the effectiveness of watermarking LLMs as a deterrent against the generation of copyrighted texts. Through theoretical analysis and empirical evaluation, we demonstrate that incorporating watermarks into LLMs significantly reduces the likelihood of generating copyrighted content, thereby addressing a critical concern in the deployment of LLMs. However, we also find that watermarking can have unintended consequences on Membership Inference Attacks (MIAs), which aim to discern whether a sample was part of the pretraining dataset and may be used to detect copyright violations. Surprisingly, we find that watermarking adversely affects the success rate of MIAs, complicating the task of detecting copyrighted text in the pretraining dataset. These results reveal the complex interplay between different regulatory measures, which may impact each other in unforeseen ways. Finally, we propose an adaptive technique to improve the success rate of a recent MIA under watermarking. Our findings underscore the importance of developing adaptive methods to study critical problems in LLMs with potential legal implications.



## **28. CtrlRAG: Black-box Adversarial Attacks Based on Masked Language Models in Retrieval-Augmented Language Generation**

cs.CL

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06950v1) [paper-pdf](http://arxiv.org/pdf/2503.06950v1)

**Authors**: Runqi Sui

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by integrating external knowledge bases. However, this integration introduces a new security threat: adversaries can exploit the retrieval mechanism to inject malicious content into the knowledge base, thereby influencing the generated responses. Based on this attack vector, we propose CtrlRAG, a novel attack method designed for RAG system in the black-box setting, which aligns with real-world scenarios. Unlike existing attack methods, CtrlRAG introduces a perturbation mechanism using Masked Language Model (MLM) to dynamically optimize malicious content in response to changes in the retrieved context. Experimental results demonstrate that CtrlRAG outperforms three baseline methods in both Emotional Manipulation and Hallucination Amplification objectives. Furthermore, we evaluate three existing defense mechanisms, revealing their limited effectiveness against CtrlRAG and underscoring the urgent need for more robust defenses.



## **29. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

cs.RO

Github: https://github.com/William-wAng618/roboticAttack Homepage:  https://vlaattacker.github.io/

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2411.13587v3) [paper-pdf](http://arxiv.org/pdf/2411.13587v3)

**Authors**: Taowen Wang, Cheng Han, James Chenhao Liang, Wenhao Yang, Dongfang Liu, Luna Xinyu Zhang, Qifan Wang, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce two untargeted attack objectives that leverage spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, we advance both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for continuously developing robust defense strategies prior to physical-world deployments.



## **30. Privacy Auditing of Large Language Models**

cs.CR

ICLR 2025

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.06808v1) [paper-pdf](http://arxiv.org/pdf/2503.06808v1)

**Authors**: Ashwinee Panda, Xinyu Tang, Milad Nasr, Christopher A. Choquette-Choo, Prateek Mittal

**Abstract**: Current techniques for privacy auditing of large language models (LLMs) have limited efficacy -- they rely on basic approaches to generate canaries which leads to weak membership inference attacks that in turn give loose lower bounds on the empirical privacy leakage. We develop canaries that are far more effective than those used in prior work under threat models that cover a range of realistic settings. We demonstrate through extensive experiments on multiple families of fine-tuned LLMs that our approach sets a new standard for detection of privacy leakage. For measuring the memorization rate of non-privately trained LLMs, our designed canaries surpass prior approaches. For example, on the Qwen2.5-0.5B model, our designed canaries achieve $49.6\%$ TPR at $1\%$ FPR, vastly surpassing the prior approach's $4.2\%$ TPR at $1\%$ FPR. Our method can be used to provide a privacy audit of $\varepsilon \approx 1$ for a model trained with theoretical $\varepsilon$ of 4. To the best of our knowledge, this is the first time that a privacy audit of LLM training has achieved nontrivial auditing success in the setting where the attacker cannot train shadow models, insert gradient canaries, or access the model at every iteration.



## **31. Can Small Language Models Reliably Resist Jailbreak Attacks? A Comprehensive Evaluation**

cs.CR

19 pages, 12 figures

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.06519v1) [paper-pdf](http://arxiv.org/pdf/2503.06519v1)

**Authors**: Wenhui Zhang, Huiyu Xu, Zhibo Wang, Zeqing He, Ziqi Zhu, Kui Ren

**Abstract**: Small language models (SLMs) have emerged as promising alternatives to large language models (LLMs) due to their low computational demands, enhanced privacy guarantees and comparable performance in specific domains through light-weight fine-tuning. Deploying SLMs on edge devices, such as smartphones and smart vehicles, has become a growing trend. However, the security implications of SLMs have received less attention than LLMs, particularly regarding jailbreak attacks, which is recognized as one of the top threats of LLMs by the OWASP. In this paper, we conduct the first large-scale empirical study of SLMs' vulnerabilities to jailbreak attacks. Through systematically evaluation on 63 SLMs from 15 mainstream SLM families against 8 state-of-the-art jailbreak methods, we demonstrate that 47.6% of evaluated SLMs show high susceptibility to jailbreak attacks (ASR > 40%) and 38.1% of them can not even resist direct harmful query (ASR > 50%). We further analyze the reasons behind the vulnerabilities and identify four key factors: model size, model architecture, training datasets and training techniques. Moreover, we assess the effectiveness of three prompt-level defense methods and find that none of them achieve perfect performance, with detection accuracy varying across different SLMs and attack methods. Notably, we point out that the inherent security awareness play a critical role in SLM security, and models with strong security awareness could timely terminate unsafe response with little reminder. Building upon the findings, we highlight the urgent need for security-by-design approaches in SLM development and provide valuable insights for building more trustworthy SLM ecosystem.



## **32. Life-Cycle Routing Vulnerabilities of LLM Router**

cs.CR

Work in progress

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.08704v1) [paper-pdf](http://arxiv.org/pdf/2503.08704v1)

**Authors**: Qiqi Lin, Xiaoyang Ji, Shengfang Zhai, Qingni Shen, Zhi Zhang, Yuejian Fang, Yansong Gao

**Abstract**: Large language models (LLMs) have achieved remarkable success in natural language processing, yet their performance and computational costs vary significantly. LLM routers play a crucial role in dynamically balancing these trade-offs. While previous studies have primarily focused on routing efficiency, security vulnerabilities throughout the entire LLM router life cycle, from training to inference, remain largely unexplored. In this paper, we present a comprehensive investigation into the life-cycle routing vulnerabilities of LLM routers. We evaluate both white-box and black-box adversarial robustness, as well as backdoor robustness, across several representative routing models under extensive experimental settings. Our experiments uncover several key findings: 1) Mainstream DNN-based routers tend to exhibit the weakest adversarial and backdoor robustness, largely due to their strong feature extraction capabilities that amplify vulnerabilities during both training and inference; 2) Training-free routers demonstrate the strongest robustness across different attack types, benefiting from the absence of learnable parameters that can be manipulated. These findings highlight critical security risks spanning the entire life cycle of LLM routers and provide insights for developing more robust models.



## **33. MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks**

cs.LG

Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2502.17832v2) [paper-pdf](http://arxiv.org/pdf/2502.17832v2)

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-Wei Chang, Daniel Kang, Heng Ji

**Abstract**: Multimodal large language models (MLLMs) equipped with Retrieval Augmented Generation (RAG) leverage both their rich parametric knowledge and the dynamic, external knowledge to excel in tasks such as Question Answering. While RAG enhances MLLMs by grounding responses in query-relevant external knowledge, this reliance poses a critical yet underexplored safety risk: knowledge poisoning attacks, where misinformation or irrelevant knowledge is intentionally injected into external knowledge bases to manipulate model outputs to be incorrect and even harmful. To expose such vulnerabilities in multimodal RAG, we propose MM-PoisonRAG, a novel knowledge poisoning attack framework with two attack strategies: Localized Poisoning Attack (LPA), which injects query-specific misinformation in both text and images for targeted manipulation, and Globalized Poisoning Attack (GPA) to provide false guidance during MLLM generation to elicit nonsensical responses across all queries. We evaluate our attacks across multiple tasks, models, and access settings, demonstrating that LPA successfully manipulates the MLLM to generate attacker-controlled answers, with a success rate of up to 56% on MultiModalQA. Moreover, GPA completely disrupts model generation to 0% accuracy with just a single irrelevant knowledge injection. Our results highlight the urgent need for robust defenses against knowledge poisoning to safeguard multimodal RAG frameworks.



## **34. Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions**

cs.CL

3 tables and 1 figures in the main text. This paper is accepted by  NAACL 2025 findings

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2410.18966v2) [paper-pdf](http://arxiv.org/pdf/2410.18966v2)

**Authors**: Yujuan Fu, Ozlem Uzuner, Meliha Yetisgen, Fei Xia

**Abstract**: Large language models (LLMs) have demonstrated great performance across various benchmarks, showing potential as general-purpose task solvers. However, as LLMs are typically trained on vast amounts of data, a significant concern in their evaluation is data contamination, where overlap between training data and evaluation datasets inflates performance assessments. Multiple approaches have been developed to identify data contamination. These approaches rely on specific assumptions that may not hold universally across different settings. To bridge this gap, we systematically review 50 papers on data contamination detection, categorize the underlying assumptions, and assess whether they have been rigorously validated. We identify and analyze eight categories of assumptions and test three of them as case studies. Our case studies focus on detecting direct, instance-level data contamination, which is also referred to as Membership Inference Attacks (MIA). Our analysis reveals that MIA approaches based on these three assumptions can have similar performance to random guessing, on datasets used in LLM pretraining, suggesting that current LLMs might learn data distributions rather than memorizing individual instances. Meanwhile, MIA can easily fail when there are data distribution shifts between the seen and unseen instances.



## **35. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2411.00827v3) [paper-pdf](http://arxiv.org/pdf/2411.00827v3)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLBreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.



## **36. Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models**

cs.LG

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06269v1) [paper-pdf](http://arxiv.org/pdf/2503.06269v1)

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at https://github.com/Sckathach/subspace-rerouting.



## **37. Reinforced Diffuser for Red Teaming Large Vision-Language Models**

cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06223v1) [paper-pdf](http://arxiv.org/pdf/2503.06223v1)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The rapid advancement of large Vision-Language Models (VLMs) has raised significant safety concerns, particularly regarding their vulnerability to jailbreak attacks. While existing research primarily focuses on VLMs' susceptibility to harmful instructions, this work identifies a critical yet overlooked vulnerability: current alignment mechanisms often fail to address the risks posed by toxic text continuation tasks. To investigate this issue, we propose a novel Red Team Diffuser (RTD) framework, which leverages reinforcement learning to generate red team images that effectively induce highly toxic continuations from target black-box VLMs. The RTD pipeline begins with a greedy search for high-quality image prompts that maximize the toxicity of VLM-generated sentence continuations, guided by a Large Language Model (LLM). These prompts are then used as input for the reinforcement fine-tuning of a diffusion model, which employs toxicity and alignment rewards to further amplify harmful outputs. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% on the original attack set and 8.91% on a hold-out set. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. These findings reveal significant deficiencies in existing alignment strategies, particularly their inability to prevent harmful continuations. Our work underscores the urgent need for more robust and adaptive alignment mechanisms to ensure the safe deployment of VLMs in real-world applications.



## **38. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

cs.CR

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2410.02644v2) [paper-pdf](http://arxiv.org/pdf/2410.02644v2)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.



## **39. Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks**

cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05445v1) [paper-pdf](http://arxiv.org/pdf/2503.05445v1)

**Authors**: Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang

**Abstract**: Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.



## **40. A Practical Memory Injection Attack against LLM Agents**

cs.LG

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.03704v2) [paper-pdf](http://arxiv.org/pdf/2503.03704v2)

**Authors**: Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang

**Abstract**: Agents based on large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, that enables the injection of malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps leading to undesirable agent actions when executing the victim user's query. Specifically, we introduce a sequence of bridging steps to link the victim query to the malicious reasoning steps. During the injection of the malicious record, we propose an indication prompt to guide the agent to autonomously generate our designed bridging steps. We also propose a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing the victim query comes after. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting practical risks of LLM agents.



## **41. Double Backdoored: Converting Code Large Language Model Backdoors to Traditional Malware via Adversarial Instruction Tuning Attacks**

cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2404.18567v2) [paper-pdf](http://arxiv.org/pdf/2404.18567v2)

**Authors**: Md Imran Hossen, Sai Venkatesh Chilukoti, Liqun Shan, Sheng Chen, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Large Language Models designed for coding tasks are increasingly employed as AI coding assistants. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. This work investigates novel techniques for transitioning backdoors from the AI/ML domain to traditional computer malware, shedding light on the critical intersection of AI and cyber/software security. To explore this intersection, we present MalInstructCoder, a framework designed to comprehensively assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs. MalInstructCoder introduces an automated data poisoning pipeline to inject malicious code snippets into benign code, poisoning instruction fine-tuning data while maintaining functional validity. It presents two practical adversarial instruction tuning attacks with real-world security implications: the clean prompt poisoning attack and the backdoor attack. These attacks aim to manipulate Code LLMs to generate code incorporating malicious or harmful functionality under specific attack scenarios while preserving intended functionality. We conduct a comprehensive investigation into the exploitability of the code-specific instruction tuning process involving three state-of-the-art Code LLMs: CodeLlama, DeepSeek-Coder, and StarCoder2. Our findings reveal that these models are highly vulnerable to our attacks. Specifically, the clean prompt poisoning attack achieves the ASR@1 ranging from over 75% to 86% by poisoning only 1% (162 samples) of the instruction fine-tuning dataset. Similarly, the backdoor attack achieves the ASR@1 ranging from 76% to 86% with a 0.5% poisoning rate. Our study sheds light on the critical cybersecurity risks posed by instruction-tuned Code LLMs and highlights the urgent need for robust defense mechanisms.



## **42. Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety**

cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.05021v1) [paper-pdf](http://arxiv.org/pdf/2503.05021v1)

**Authors**: Yuyou Zhang, Miao Li, William Han, Yihang Yao, Zhepeng Cen, Ding Zhao

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios.



## **43. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

cs.CR

ICLR 2025 camera-ready version

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.06186v4) [paper-pdf](http://arxiv.org/pdf/2410.06186v4)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.



## **44. Get my drift? Catching LLM Task Drift with Activation Deltas**

cs.CR

SaTML 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2406.00799v6) [paper-pdf](http://arxiv.org/pdf/2406.00799v6)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: LLMs are commonly used in retrieval-augmented applications to execute user instructions based on data from external sources. For example, modern search engines use LLMs to answer queries based on relevant search results; email plugins summarize emails by processing their content through an LLM. However, the potentially untrusted provenance of these data sources can lead to prompt injection attacks, where the LLM is manipulated by natural language instructions embedded in the external data, causing it to deviate from the user's original instruction(s). We define this deviation as task drift. Task drift is a significant concern as it allows attackers to exfiltrate data or influence the LLM's output for other users. We study LLM activations as a solution to detect task drift, showing that activation deltas - the difference in activations before and after processing external data - are strongly correlated with this phenomenon. Through two probing methods, we demonstrate that a simple linear classifier can detect drift with near-perfect ROC AUC on an out-of-distribution test set. We evaluate these methods by making minimal assumptions about how users' tasks, system prompts, and attacks can be phrased. We observe that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Interestingly, the fact that this solution does not require any modifications to the LLM (e.g., fine-tuning), as well as its compatibility with existing meta-prompting solutions, makes it cost-efficient and easy to deploy. To encourage further research on activation-based task inspection, decoding, and interpretability, we release our large-scale TaskTracker toolkit, featuring a dataset of over 500K instances, representations from six SoTA language models, and a suite of inspection tools.



## **45. Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges**

cs.LG

Accepted to the ICBINB Workshop at ICLR'25

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04474v1) [paper-pdf](http://arxiv.org/pdf/2503.04474v1)

**Authors**: Francisco Eiras, Eliott Zemour, Eric Lin, Vaikkunth Mugunthan

**Abstract**: Large Language Model (LLM) based judges form the underpinnings of key safety evaluation processes such as offline benchmarking, automated red-teaming, and online guardrailing. This widespread requirement raises the crucial question: can we trust the evaluations of these evaluators? In this paper, we highlight two critical challenges that are typically overlooked: (i) evaluations in the wild where factors like prompt sensitivity and distribution shifts can affect performance and (ii) adversarial attacks that target the judge. We highlight the importance of these through a study of commonly used safety judges, showing that small changes such as the style of the model output can lead to jumps of up to 0.24 in the false negative rate on the same dataset, whereas adversarial attacks on the model generation can fool some judges into misclassifying 100% of harmful generations as safe ones. These findings reveal gaps in commonly used meta-evaluation benchmarks and weaknesses in the robustness of current LLM judges, indicating that low attack success under certain judges could create a false sense of security.



## **46. Stealthy Jailbreak Attacks on Large Language Models via Benign Data Mirroring**

cs.CL

Accepted by NAACL 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.21083v2) [paper-pdf](http://arxiv.org/pdf/2410.21083v2)

**Authors**: Honglin Mu, Han He, Yuxin Zhou, Yunlong Feng, Yang Xu, Libo Qin, Xiaoming Shi, Zeming Liu, Xudong Han, Qi Shi, Qingfu Zhu, Wanxiang Che

**Abstract**: Large language model (LLM) safety is a critical issue, with numerous studies employing red team testing to enhance model security. Among these, jailbreak methods explore potential vulnerabilities by crafting malicious prompts that induce model outputs contrary to safety alignments. Existing black-box jailbreak methods often rely on model feedback, repeatedly submitting queries with detectable malicious instructions during the attack search process. Although these approaches are effective, the attacks may be intercepted by content moderators during the search process. We propose an improved transfer attack method that guides malicious prompt construction by locally training a mirror model of the target black-box model through benign data distillation. This method offers enhanced stealth, as it does not involve submitting identifiable malicious instructions to the target model during the search phase. Our approach achieved a maximum attack success rate of 92%, or a balanced value of 80% with an average of 1.5 detectable jailbreak queries per sample against GPT-3.5 Turbo on a subset of AdvBench. These results underscore the need for more robust defense mechanisms.



## **47. Exploring the Multilingual NLG Evaluation Abilities of LLM-Based Evaluators**

cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04360v1) [paper-pdf](http://arxiv.org/pdf/2503.04360v1)

**Authors**: Jiayi Chang, Mingqi Gao, Xinyu Hu, Xiaojun Wan

**Abstract**: Previous research has shown that LLMs have potential in multilingual NLG evaluation tasks. However, existing research has not fully explored the differences in the evaluation capabilities of LLMs across different languages. To this end, this study provides a comprehensive analysis of the multilingual evaluation performance of 10 recent LLMs, spanning high-resource and low-resource languages through correlation analysis, perturbation attacks, and fine-tuning. We found that 1) excluding the reference answer from the prompt and using large-parameter LLM-based evaluators leads to better performance across various languages; 2) most LLM-based evaluators show a higher correlation with human judgments in high-resource languages than in low-resource languages; 3) in the languages where they are most sensitive to such attacks, they also tend to exhibit the highest correlation with human judgments; and 4) fine-tuning with data from a particular language yields a broadly consistent enhancement in the model's evaluation performance across diverse languages. Our findings highlight the imbalance in LLMs'evaluation capabilities across different languages and suggest that low-resource language scenarios deserve more attention.



## **48. Malware Detection at the Edge with Lightweight LLMs: A Performance Evaluation**

cs.CR

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04302v1) [paper-pdf](http://arxiv.org/pdf/2503.04302v1)

**Authors**: Christian Rondanini, Barbara Carminati, Elena Ferrari, Antonio Gaudiano, Ashish Kundu

**Abstract**: The rapid evolution of malware attacks calls for the development of innovative detection methods, especially in resource-constrained edge computing. Traditional detection techniques struggle to keep up with modern malware's sophistication and adaptability, prompting a shift towards advanced methodologies like those leveraging Large Language Models (LLMs) for enhanced malware detection. However, deploying LLMs for malware detection directly at edge devices raises several challenges, including ensuring accuracy in constrained environments and addressing edge devices' energy and computational limits. To tackle these challenges, this paper proposes an architecture leveraging lightweight LLMs' strengths while addressing limitations like reduced accuracy and insufficient computational power. To evaluate the effectiveness of the proposed lightweight LLM-based approach for edge computing, we perform an extensive experimental evaluation using several state-of-the-art lightweight LLMs. We test them with several publicly available datasets specifically designed for edge and IoT scenarios and different edge nodes with varying computational power and characteristics.



## **49. One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs**

cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04856v1) [paper-pdf](http://arxiv.org/pdf/2503.04856v1)

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim

**Abstract**: Despite extensive safety enhancements in large language models (LLMs), multi-turn "jailbreak" conversations crafted by skilled human adversaries can still breach even the most sophisticated guardrails. However, these multi-turn attacks demand considerable manual effort, limiting their scalability. In this work, we introduce a novel approach called Multi-turn-to-Single-turn (M2S) that systematically converts multi-turn jailbreak prompts into single-turn attacks. Specifically, we propose three conversion strategies - Hyphenize, Numberize, and Pythonize - each preserving sequential context yet packaging it in a single query. Our experiments on the Multi-turn Human Jailbreak (MHJ) dataset show that M2S often increases or maintains high Attack Success Rates (ASRs) compared to original multi-turn conversations. Notably, using a StrongREJECT-based evaluation of harmfulness, M2S achieves up to 95.9% ASR on Mistral-7B and outperforms original multi-turn prompts by as much as 17.5% in absolute improvement on GPT-4o. Further analysis reveals that certain adversarial tactics, when consolidated into a single prompt, exploit structural formatting cues to evade standard policy checks. These findings underscore that single-turn attacks - despite being simpler and cheaper to conduct - can be just as potent, if not more, than their multi-turn counterparts. Our findings underscore the urgent need to reevaluate and reinforce LLM safety strategies, given how adversarial queries can be compacted into a single prompt while still retaining sufficient complexity to bypass existing safety measures.



## **50. The VLLM Safety Paradox: Dual Ease in Jailbreak Attack and Defense**

cs.CR

Logic smoothing and language polishing

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2411.08410v2) [paper-pdf](http://arxiv.org/pdf/2411.08410v2)

**Authors**: Yangyang Guo, Fangkai Jiao, Liqiang Nie, Mohan Kankanhalli

**Abstract**: The vulnerability of Vision Large Language Models (VLLMs) to jailbreak attacks appears as no surprise. However, recent defense mechanisms against these attacks have reached near-saturation performance on benchmark evaluations, often with minimal effort. This \emph{dual high performance} in both attack and defense raises a fundamental and perplexing paradox. To gain a deep understanding of this issue and thus further help strengthen the trustworthiness of VLLMs, this paper makes three key contributions: i) One tentative explanation for VLLMs being prone to jailbreak attacks--\textbf{inclusion of vision inputs}, as well as its in-depth analysis. ii) The recognition of a largely ignored problem in existing defense mechanisms--\textbf{over-prudence}. The problem causes these defense methods to exhibit unintended abstention, even in the presence of benign inputs, thereby undermining their reliability in faithfully defending against attacks. iii) A simple safety-aware method--\textbf{LLM-Pipeline}. Our method repurposes the more advanced guardrails of LLMs on the shelf, serving as an effective alternative detector prior to VLLM response. Last but not least, we find that the two representative evaluation methods for jailbreak often exhibit chance agreement. This limitation makes it potentially misleading when evaluating attack strategies or defense mechanisms. We believe the findings from this paper offer useful insights to rethink the foundational development of VLLM safety with respect to benchmark datasets, defense strategies, and evaluation methods.



