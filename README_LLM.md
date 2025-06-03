# Latest Large Language Model Attack Papers
**update at 2025-06-03 09:49:02**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. A Large Language Model-Supported Threat Modeling Framework for Transportation Cyber-Physical Systems**

cs.CR

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2506.00831v1) [paper-pdf](http://arxiv.org/pdf/2506.00831v1)

**Authors**: M Sabbir Salek, Mashrur Chowdhury, Muhaimin Bin Munir, Yuchen Cai, Mohammad Imtiaz Hasan, Jean-Michel Tine, Latifur Khan, Mizanur Rahman

**Abstract**: Modern transportation systems rely on cyber-physical systems (CPS), where cyber systems interact seamlessly with physical systems like transportation-related sensors and actuators to enhance safety, mobility, and energy efficiency. However, growing automation and connectivity increase exposure to cyber vulnerabilities. Existing threat modeling frameworks for transportation CPS are often limited in scope, resource-intensive, and dependent on significant cybersecurity expertise. To address these gaps, we present TraCR-TMF (Transportation Cybersecurity and Resiliency Threat Modeling Framework), a large language model (LLM)-based framework that minimizes expert intervention. TraCR-TMF identifies threats, potential attack techniques, and corresponding countermeasures by leveraging the MITRE ATT&CK matrix through three LLM-based approaches: (i) a retrieval-augmented generation (RAG) method requiring no expert input, (ii) an in-context learning approach requiring low expert input, and (iii) a supervised fine-tuning method requiring moderate expert input. TraCR-TMF also maps attack paths to critical assets by analyzing vulnerabilities using a customized LLM. The framework was evaluated in two scenarios. First, it identified relevant attack techniques across transportation CPS applications, with 90% precision as validated by experts. Second, using a fine-tuned LLM, it successfully predicted multiple exploitations including lateral movement, data exfiltration, and ransomware-related encryption that occurred during a major real-world cyberattack incident. These results demonstrate TraCR-TMF's effectiveness in CPS threat modeling, its reduced reliance on cybersecurity expertise, and its adaptability across CPS domains.



## **2. Jailbreak-R1: Exploring the Jailbreak Capabilities of LLMs via Reinforcement Learning**

cs.AI

21 pages, 8 figures

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2506.00782v1) [paper-pdf](http://arxiv.org/pdf/2506.00782v1)

**Authors**: Weiyang Guo, Zesheng Shi, Zhuo Li, Yequan Wang, Xuebo Liu, Wenya Wang, Fangming Liu, Min Zhang, Jing Li

**Abstract**: As large language models (LLMs) grow in power and influence, ensuring their safety and preventing harmful output becomes critical. Automated red teaming serves as a tool to detect security vulnerabilities in LLMs without manual labor. However, most existing methods struggle to balance the effectiveness and diversity of red-team generated attack prompts. To address this challenge, we propose \ourapproach, a novel automated red teaming training framework that utilizes reinforcement learning to explore and generate more effective attack prompts while balancing their diversity. Specifically, it consists of three training stages: (1) Cold Start: The red team model is supervised and fine-tuned on a jailbreak dataset obtained through imitation learning. (2) Warm-up Exploration: The model is trained in jailbreak instruction following and exploration, using diversity and consistency as reward signals. (3) Enhanced Jailbreak: Progressive jailbreak rewards are introduced to gradually enhance the jailbreak performance of the red-team model. Extensive experiments on a variety of LLMs show that \ourapproach effectively balances the diversity and effectiveness of jailbreak prompts compared to existing methods. Our work significantly improves the efficiency of red team exploration and provides a new perspective on automated red teaming.



## **3. CoP: Agentic Red-teaming for Large Language Models using Composition of Principles**

cs.AI

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2506.00781v1) [paper-pdf](http://arxiv.org/pdf/2506.00781v1)

**Authors**: Chen Xiong, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Recent advances in Large Language Models (LLMs) have spurred transformative applications in various domains, ranging from open-source to proprietary LLMs. However, jailbreak attacks, which aim to break safety alignment and user compliance by tricking the target LLMs into answering harmful and risky responses, are becoming an urgent concern. The practice of red-teaming for LLMs is to proactively explore potential risks and error-prone instances before the release of frontier AI technology. This paper proposes an agentic workflow to automate and scale the red-teaming process of LLMs through the Composition-of-Principles (CoP) framework, where human users provide a set of red-teaming principles as instructions to an AI agent to automatically orchestrate effective red-teaming strategies and generate jailbreak prompts. Distinct from existing red-teaming methods, our CoP framework provides a unified and extensible framework to encompass and orchestrate human-provided red-teaming principles to enable the automated discovery of new red-teaming strategies. When tested against leading LLMs, CoP reveals unprecedented safety risks by finding novel jailbreak prompts and improving the best-known single-turn attack success rate by up to 19.0 times.



## **4. Underestimated Privacy Risks for Minority Populations in Large Language Model Unlearning**

cs.LG

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2412.08559v3) [paper-pdf](http://arxiv.org/pdf/2412.08559v3)

**Authors**: Rongzhe Wei, Mufei Li, Mohsen Ghassemi, Eleonora Kreačić, Yifan Li, Xiang Yue, Bo Li, Vamsi K. Potluru, Pan Li, Eli Chien

**Abstract**: Large Language Models (LLMs) embed sensitive, human-generated data, prompting the need for unlearning methods. Although certified unlearning offers strong privacy guarantees, its restrictive assumptions make it unsuitable for LLMs, giving rise to various heuristic approaches typically assessed through empirical evaluations. These standard evaluations randomly select data for removal, apply unlearning techniques, and use membership inference attacks (MIAs) to compare unlearned models against models retrained without the removed data. However, to ensure robust privacy protections for every data point, it is essential to account for scenarios in which certain data subsets face elevated risks. Prior research suggests that outliers, particularly including data tied to minority groups, often exhibit higher memorization propensity which indicates they may be more difficult to unlearn. Building on these insights, we introduce a complementary, minority-aware evaluation framework to highlight blind spots in existing frameworks. We substantiate our findings with carefully designed experiments, using canaries with personally identifiable information (PII) to represent these minority subsets and demonstrate that they suffer at least 20% higher privacy leakage across various unlearning methods, MIAs, datasets, and LLM scales. Our proposed minority-aware evaluation framework marks an essential step toward more equitable and comprehensive assessments of LLM unlearning efficacy.



## **5. Tokens for Learning, Tokens for Unlearning: Mitigating Membership Inference Attacks in Large Language Models via Dual-Purpose Training**

cs.LG

ACL'25 (Findings)

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2502.19726v2) [paper-pdf](http://arxiv.org/pdf/2502.19726v2)

**Authors**: Toan Tran, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) have become the backbone of modern natural language processing but pose privacy concerns about leaking sensitive training data. Membership inference attacks (MIAs), which aim to infer whether a sample is included in a model's training dataset, can serve as a foundation for broader privacy threats. Existing defenses designed for traditional classification models do not account for the sequential nature of text data. As a result, they either require significant computational resources or fail to effectively mitigate privacy risks in LLMs. In this work, we propose \methodname, a lightweight yet effective empirical privacy defense for protecting training data of language models by leveraging token-specific characteristics. By analyzing token dynamics during training, we propose a token selection strategy that categorizes tokens into hard tokens for learning and memorized tokens for unlearning. Subsequently, our training-phase defense optimizes a novel dual-purpose token-level loss to achieve a Pareto-optimal balance between utility and privacy. Extensive experiments demonstrate that our approach not only provides strong protection against MIAs but also improves language modeling performance by around 10\% across various LLM architectures and datasets compared to the baselines.



## **6. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

cs.CR

Accepted to CVPR 2025

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2411.18688v4) [paper-pdf](http://arxiv.org/pdf/2411.18688v4)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Alvaro Velasquez, Ahmad Beirami, Furong Huang, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks. In this work, we first highlight an important safety gap to describe that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model through controlled decoding to defend against jailbreak attacks. Additionally, we provide a mathematical characterization of Immune, offering insights on why it improves safety against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.



## **7. Security Concerns for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2505.18889v2) [paper-pdf](http://arxiv.org/pdf/2505.18889v2)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as GPT-4 and its recent iterations, Google's Gemini, Anthropic's Claude 3 models, and xAI's Grok have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. In this survey, we provide a comprehensive overview of the emerging security concerns around LLMs, categorizing threats into prompt injection and jailbreaking, adversarial attacks such as input perturbations and data poisoning, misuse by malicious actors for purposes such as generating disinformation, phishing emails, and malware, and worrisome risks inherent in autonomous LLM agents. A significant focus has been recently placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.



## **8. SafeTuneBed: A Toolkit for Benchmarking LLM Safety Alignment in Fine-Tuning**

cs.LG

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00676v1) [paper-pdf](http://arxiv.org/pdf/2506.00676v1)

**Authors**: Saad Hossain, Samanvay Vajpayee, Sirisha Rambhatla

**Abstract**: As large language models (LLMs) become ubiquitous, parameter-efficient fine-tuning methods and safety-first defenses have proliferated rapidly. However, the number of approaches and their recent increase have resulted in diverse evaluations-varied datasets, metrics, and inconsistent threat settings-making it difficult to fairly compare safety, utility, and robustness across methods. To address this, we introduce SafeTuneBed, a benchmark and toolkit unifying fine-tuning and defense evaluation. SafeTuneBed (i) curates a diverse repository of multiple fine-tuning datasets spanning sentiment analysis, question-answering, multi-step reasoning, and open-ended instruction tasks, and allows for the generation of harmful-variant splits; (ii) enables integration of state-of-the-art defenses, including alignment-stage immunization, in-training safeguards, and post-tuning repair; and (iii) provides evaluators for safety (attack success rate, refusal consistency) and utility. Built on Python-first, dataclass-driven configs and plugins, SafeTuneBed requires minimal additional code to specify any fine-tuning regime, defense method, and metric suite, while ensuring end-to-end reproducibility. We showcase its value by benchmarking representative defenses across varied poisoning scenarios and tasks. By standardizing data, code, and metrics, SafeTuneBed is the first focused toolkit of its kind to accelerate rigorous and comparable research in safe LLM fine-tuning. Code is available at: https://github.com/criticalml-uw/SafeTuneBed



## **9. SafeTy Reasoning Elicitation Alignment for Multi-Turn Dialogues**

cs.CL

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00668v1) [paper-pdf](http://arxiv.org/pdf/2506.00668v1)

**Authors**: Martin Kuo, Jianyi Zhang, Aolin Ding, Louis DiValentin, Amin Hass, Benjamin F Morris, Isaac Jacobson, Randolph Linderman, James Kiessling, Nicolas Ramos, Bhavna Gopal, Maziyar Baran Pouyan, Changwei Liu, Hai Li, Yiran Chen

**Abstract**: Malicious attackers can exploit large language models (LLMs) by engaging them in multi-turn dialogues to achieve harmful objectives, posing significant safety risks to society. To address this challenge, we propose a novel defense mechanism: SafeTy Reasoning Elicitation Alignment for Multi-Turn Dialogues (STREAM). STREAM defends LLMs against multi-turn attacks while preserving their functional capabilities. Our approach involves constructing a human-annotated dataset, the Safety Reasoning Multi-turn Dialogues dataset, which is used to fine-tune a plug-and-play safety reasoning moderator. This model is designed to identify malicious intent hidden within multi-turn conversations and alert the target LLM of potential risks. We evaluate STREAM across multiple LLMs against prevalent multi-turn attack strategies. Experimental results demonstrate that our method significantly outperforms existing defense techniques, reducing the Attack Success Rate (ASR) by 51.2%, all while maintaining comparable LLM capability.



## **10. Stepwise Reasoning Error Disruption Attack of LLMs**

cs.AI

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2412.11934v4) [paper-pdf](http://arxiv.org/pdf/2412.11934v4)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications. Our code is available at: https://github.com/Applied-Machine-Learning-Lab/SEED-Attack.



## **11. Goal-Aware Identification and Rectification of Misinformation in Multi-Agent Systems**

cs.CL

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00509v1) [paper-pdf](http://arxiv.org/pdf/2506.00509v1)

**Authors**: Zherui Li, Yan Mi, Zhenhong Zhou, Houcheng Jiang, Guibin Zhang, Kun Wang, Junfeng Fang

**Abstract**: Large Language Model-based Multi-Agent Systems (MASs) have demonstrated strong advantages in addressing complex real-world tasks. However, due to the introduction of additional attack surfaces, MASs are particularly vulnerable to misinformation injection. To facilitate a deeper understanding of misinformation propagation dynamics within these systems, we introduce MisinfoTask, a novel dataset featuring complex, realistic tasks designed to evaluate MAS robustness against such threats. Building upon this, we propose ARGUS, a two-stage, training-free defense framework leveraging goal-aware reasoning for precise misinformation rectification within information flows. Our experiments demonstrate that in challenging misinformation scenarios, ARGUS exhibits significant efficacy across various injection attacks, achieving an average reduction in misinformation toxicity of approximately 28.17% and improving task success rates under attack by approximately 10.33%. Our code and dataset is available at: https://github.com/zhrli324/ARGUS.



## **12. WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermarks**

cs.CR

Accepted to ACL 2025 (Main Proceedings)

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2409.04459v2) [paper-pdf](http://arxiv.org/pdf/2409.04459v2)

**Authors**: Anudeex Shetty, Qiongkai Xu, Jey Han Lau

**Abstract**: Embeddings-as-a-Service (EaaS) is a service offered by large language model (LLM) developers to supply embeddings generated by LLMs. Previous research suggests that EaaS is prone to imitation attacks -- attacks that clone the underlying EaaS model by training another model on the queried embeddings. As a result, EaaS watermarks are introduced to protect the intellectual property of EaaS providers. In this paper, we first show that existing EaaS watermarks can be removed by paraphrasing when attackers clone the model. Subsequently, we propose a novel watermarking technique that involves linearly transforming the embeddings, and show that it is empirically and theoretically robust against paraphrasing.



## **13. Spectral Insights into Data-Oblivious Critical Layers in Large Language Models**

cs.LG

Accepted by Findings of ACL2025

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00382v1) [paper-pdf](http://arxiv.org/pdf/2506.00382v1)

**Authors**: Xuyuan Liu, Lei Hsiung, Yaoqing Yang, Yujun Yan

**Abstract**: Understanding how feature representations evolve across layers in large language models (LLMs) is key to improving their interpretability and robustness. While recent studies have identified critical layers linked to specific functions or behaviors, these efforts typically rely on data-dependent analyses of fine-tuned models, limiting their use to post-hoc settings. In contrast, we introduce a data-oblivious approach to identify intrinsic critical layers in pre-fine-tuned LLMs by analyzing representation dynamics via Centered Kernel Alignment(CKA). We show that layers with significant shifts in representation space are also those most affected during fine-tuning--a pattern that holds consistently across tasks for a given model. Our spectral analysis further reveals that these shifts are driven by changes in the top principal components, which encode semantic transitions from rationales to conclusions. We further apply these findings to two practical scenarios: efficient domain adaptation, where fine-tuning critical layers leads to greater loss reduction compared to non-critical layers; and backdoor defense, where freezing them reduces attack success rates by up to 40%.



## **14. Keeping an Eye on LLM Unlearning: The Hidden Risk and Remedy**

cs.CR

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00359v1) [paper-pdf](http://arxiv.org/pdf/2506.00359v1)

**Authors**: Jie Ren, Zhenwei Dai, Xianfeng Tang, Yue Xing, Shenglai Zeng, Hui Liu, Jingying Zeng, Qiankun Peng, Samarth Varshney, Suhang Wang, Qi He, Charu C. Aggarwal, Hui Liu

**Abstract**: Although Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of tasks, growing concerns have emerged over the misuse of sensitive, copyrighted, or harmful data during training. To address these concerns, unlearning techniques have been developed to remove the influence of specific data without retraining from scratch. However, this paper reveals a critical vulnerability in fine-tuning-based unlearning: a malicious user can craft a manipulated forgetting request that stealthily degrades the model's utility for benign users. We demonstrate this risk through a red-teaming Stealthy Attack (SA), which is inspired by two key limitations of existing unlearning (the inability to constrain the scope of unlearning effect and the failure to distinguish benign tokens from unlearning signals). Prior work has shown that unlearned models tend to memorize forgetting data as unlearning signals, and respond with hallucinations or feigned ignorance when unlearning signals appear in the input. By subtly increasing the presence of common benign tokens in the forgetting data, SA enhances the connection between benign tokens and unlearning signals. As a result, when normal users include such tokens in their prompts, the model exhibits unlearning behaviors, leading to unintended utility degradation. To address this vulnerability, we propose Scope-aware Unlearning (SU), a lightweight enhancement that introduces a scope term into the unlearning objective, encouraging the model to localize the forgetting effect. Our method requires no additional data processing, integrates seamlessly with existing fine-tuning frameworks, and significantly improves robustness against SA. Extensive experiments validate the effectiveness of both SA and SU.



## **15. Adversarial Threat Vectors and Risk Mitigation for Retrieval-Augmented Generation Systems**

cs.CR

SPIE DCS: Proceedings Volume Assurance and Security for AI-enabled  Systems 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2506.00281v1) [paper-pdf](http://arxiv.org/pdf/2506.00281v1)

**Authors**: Chris M. Ward, Josh Harguess

**Abstract**: Retrieval-Augmented Generation (RAG) systems, which integrate Large Language Models (LLMs) with external knowledge sources, are vulnerable to a range of adversarial attack vectors. This paper examines the importance of RAG systems through recent industry adoption trends and identifies the prominent attack vectors for RAG: prompt injection, data poisoning, and adversarial query manipulation. We analyze these threats under risk management lens, and propose robust prioritized control list that includes risk-mitigating actions like input validation, adversarial training, and real-time monitoring.



## **16. Differential privacy enables fair and accurate AI-based analysis of speech disorders while protecting patient data**

cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2409.19078v3) [paper-pdf](http://arxiv.org/pdf/2409.19078v3)

**Authors**: Soroosh Tayebi Arasteh, Mahshad Lotfinia, Paula Andrea Perez-Toro, Tomas Arias-Vergara, Mahtab Ranji, Juan Rafael Orozco-Arroyave, Maria Schuster, Andreas Maier, Seung Hee Yang

**Abstract**: Speech pathology has impacts on communication abilities and quality of life. While deep learning-based models have shown potential in diagnosing these disorders, the use of sensitive data raises critical privacy concerns. Although differential privacy (DP) has been explored in the medical imaging domain, its application in pathological speech analysis remains largely unexplored despite the equally critical privacy concerns. To the best of our knowledge, this study is the first to investigate DP's impact on pathological speech data, focusing on the trade-offs between privacy, diagnostic accuracy, and fairness. Using a large, real-world dataset of 200 hours of recordings from 2,839 German-speaking participants, we observed a maximum accuracy reduction of 3.85% when training with DP with high privacy levels. To highlight real-world privacy risks, we demonstrated the vulnerability of non-private models to gradient inversion attacks, reconstructing identifiable speech samples and showcasing DP's effectiveness in mitigating these risks. To explore the potential generalizability across languages and disorders, we validated our approach on a dataset of Spanish-speaking Parkinson's disease patients, leveraging pretrained models from healthy English-speaking datasets, and demonstrated that careful pretraining on large-scale task-specific datasets can maintain favorable accuracy under DP constraints. A comprehensive fairness analysis revealed minimal gender bias at reasonable privacy levels but underscored the need for addressing age-related disparities. Our results establish that DP can balance privacy and utility in speech disorder detection, while highlighting unique challenges in privacy-fairness trade-offs for speech data. This provides a foundation for refining DP methodologies and improving fairness across diverse patient groups in real-world deployments.



## **17. TRIDENT: Enhancing Large Language Model Safety with Tri-Dimensional Diversified Red-Teaming Data Synthesis**

cs.CL

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24672v1) [paper-pdf](http://arxiv.org/pdf/2505.24672v1)

**Authors**: Xiaorui Wu, Xiaofeng Mao, Fei Li, Xin Zhang, Xuanhong Li, Chong Teng, Donghong Ji, Zhuang Li

**Abstract**: Large Language Models (LLMs) excel in various natural language processing tasks but remain vulnerable to generating harmful content or being exploited for malicious purposes. Although safety alignment datasets have been introduced to mitigate such risks through supervised fine-tuning (SFT), these datasets often lack comprehensive risk coverage. Most existing datasets focus primarily on lexical diversity while neglecting other critical dimensions. To address this limitation, we propose a novel analysis framework to systematically measure the risk coverage of alignment datasets across three essential dimensions: Lexical Diversity, Malicious Intent, and Jailbreak Tactics. We further introduce TRIDENT, an automated pipeline that leverages persona-based, zero-shot LLM generation to produce diverse and comprehensive instructions spanning these dimensions. Each harmful instruction is paired with an ethically aligned response, resulting in two datasets: TRIDENT-Core, comprising 26,311 examples, and TRIDENT-Edge, with 18,773 examples. Fine-tuning Llama 3.1-8B on TRIDENT-Edge demonstrates substantial improvements, achieving an average 14.29% reduction in Harm Score, and a 20% decrease in Attack Success Rate compared to the best-performing baseline model fine-tuned on the WildBreak dataset.



## **18. Benchmarking Large Language Models for Cryptanalysis and Mismatched-Generalization**

cs.CL

Preprint

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24621v1) [paper-pdf](http://arxiv.org/pdf/2505.24621v1)

**Authors**: Utsav Maskey, Chencheng Zhu, Usman Naseem

**Abstract**: Recent advancements in Large Language Models (LLMs) have transformed natural language understanding and generation, leading to extensive benchmarking across diverse tasks. However, cryptanalysis a critical area for data security and encryption has not yet been thoroughly explored in LLM evaluations. To address this gap, we evaluate cryptanalytic potential of state of the art LLMs on encrypted texts generated using a range of cryptographic algorithms. We introduce a novel benchmark dataset comprising diverse plain texts spanning various domains, lengths, writing styles, and topics paired with their encrypted versions. Using zero-shot and few shot settings, we assess multiple LLMs for decryption accuracy and semantic comprehension across different encryption schemes. Our findings reveal key insights into the strengths and limitations of LLMs in side-channel communication while raising concerns about their susceptibility to jailbreaking attacks. This research highlights the dual-use nature of LLMs in security contexts and contributes to the ongoing discussion on AI safety and security.



## **19. BaxBench: Can LLMs Generate Correct and Secure Backends?**

cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2502.11844v3) [paper-pdf](http://arxiv.org/pdf/2502.11844v3)

**Authors**: Mark Vero, Niels Mündler, Victor Chibotaru, Veselin Raychev, Maximilian Baader, Nikola Jovanović, Jingxuan He, Martin Vechev

**Abstract**: Automatic program generation has long been a fundamental challenge in computer science. Recent benchmarks have shown that large language models (LLMs) can effectively generate code at the function level, make code edits, and solve algorithmic coding tasks. However, to achieve full automation, LLMs should be able to generate production-quality, self-contained application modules. To evaluate the capabilities of LLMs in solving this challenge, we introduce BaxBench, a novel evaluation benchmark consisting of 392 tasks for the generation of backend applications. We focus on backends for three critical reasons: (i) they are practically relevant, building the core components of most modern web and cloud software, (ii) they are difficult to get right, requiring multiple functions and files to achieve the desired functionality, and (iii) they are security-critical, as they are exposed to untrusted third-parties, making secure solutions that prevent deployment-time attacks an imperative. BaxBench validates the functionality of the generated applications with comprehensive test cases, and assesses their security exposure by executing end-to-end exploits. Our experiments reveal key limitations of current LLMs in both functionality and security: (i) even the best model, OpenAI o1, achieves a mere 62% on code correctness; (ii) on average, we could successfully execute security exploits on around half of the correct programs generated by each LLM; and (iii) in less popular backend frameworks, models further struggle to generate correct and secure applications. Progress on BaxBench signifies important steps towards autonomous and secure software development with LLMs.



## **20. Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors**

cs.CL

Accepted at Findings of ACL 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24523v1) [paper-pdf](http://arxiv.org/pdf/2505.24523v1)

**Authors**: Andrea Pedrotti, Michele Papucci, Cristiano Ciaccio, Alessio Miaschi, Giovanni Puccetti, Felice Dell'Orletta, Andrea Esuli

**Abstract**: Recent advancements in Generative AI and Large Language Models (LLMs) have enabled the creation of highly realistic synthetic content, raising concerns about the potential for malicious use, such as misinformation and manipulation. Moreover, detecting Machine-Generated Text (MGT) remains challenging due to the lack of robust benchmarks that assess generalization to real-world scenarios. In this work, we present a pipeline to test the resilience of state-of-the-art MGT detectors (e.g., Mage, Radar, LLM-DetectAIve) to linguistically informed adversarial attacks. To challenge the detectors, we fine-tune language models using Direct Preference Optimization (DPO) to shift the MGT style toward human-written text (HWT). This exploits the detectors' reliance on stylistic clues, making new generations more challenging to detect. Additionally, we analyze the linguistic shifts induced by the alignment and which features are used by detectors to detect MGT texts. Our results show that detectors can be easily fooled with relatively few examples, resulting in a significant drop in detection performance. This highlights the importance of improving detection methods and making them robust to unseen in-domain texts.



## **21. An Interpretable N-gram Perplexity Threat Model for Large Language Model Jailbreaks**

cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2410.16222v2) [paper-pdf](http://arxiv.org/pdf/2410.16222v2)

**Authors**: Valentyn Boreiko, Alexander Panfilov, Vaclav Voracek, Matthias Hein, Jonas Geiping

**Abstract**: A plethora of jailbreaking attacks have been proposed to obtain harmful responses from safety-tuned LLMs. These methods largely succeed in coercing the target output in their original settings, but their attacks vary substantially in fluency and computational effort. In this work, we propose a unified threat model for the principled comparison of these methods. Our threat model checks if a given jailbreak is likely to occur in the distribution of text. For this, we build an N-gram language model on 1T tokens, which, unlike model-based perplexity, allows for an LLM-agnostic, nonparametric, and inherently interpretable evaluation. We adapt popular attacks to this threat model, and, for the first time, benchmark these attacks on equal footing with it. After an extensive comparison, we find attack success rates against safety-tuned modern models to be lower than previously presented and that attacks based on discrete optimization significantly outperform recent LLM-based attacks. Being inherently interpretable, our threat model allows for a comprehensive analysis and comparison of jailbreak attacks. We find that effective attacks exploit and abuse infrequent bigrams, either selecting the ones absent from real-world text or rare ones, e.g., specific to Reddit or code datasets.



## **22. SEAR: A Multimodal Dataset for Analyzing AR-LLM-Driven Social Engineering Behaviors**

cs.AI

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24458v1) [paper-pdf](http://arxiv.org/pdf/2505.24458v1)

**Authors**: Tianlong Yu, Chenghang Ye, Zheyu Yang, Ziyi Zhou, Cui Tang, Zui Tao, Jun Zhang, Kailong Wang, Liting Zhou, Yang Yang, Ting Bi

**Abstract**: The SEAR Dataset is a novel multimodal resource designed to study the emerging threat of social engineering (SE) attacks orchestrated through augmented reality (AR) and multimodal large language models (LLMs). This dataset captures 180 annotated conversations across 60 participants in simulated adversarial scenarios, including meetings, classes and networking events. It comprises synchronized AR-captured visual/audio cues (e.g., facial expressions, vocal tones), environmental context, and curated social media profiles, alongside subjective metrics such as trust ratings and susceptibility assessments. Key findings reveal SEAR's alarming efficacy in eliciting compliance (e.g., 93.3% phishing link clicks, 85% call acceptance) and hijacking trust (76.7% post-interaction trust surge). The dataset supports research in detecting AR-driven SE attacks, designing defensive frameworks, and understanding multimodal adversarial manipulation. Rigorous ethical safeguards, including anonymization and IRB compliance, ensure responsible use. The SEAR dataset is available at https://github.com/INSLabCN/SEAR-Dataset.



## **23. Learning Safety Constraints for Large Language Models**

cs.LG

ICML 2025 (Spotlight)

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24445v1) [paper-pdf](http://arxiv.org/pdf/2505.24445v1)

**Authors**: Xin Chen, Yarden As, Andreas Krause

**Abstract**: Large language models (LLMs) have emerged as powerful tools but pose significant safety risks through harmful outputs and vulnerability to adversarial attacks. We propose SaP, short for Safety Polytope, a geometric approach to LLM safety that learns and enforces multiple safety constraints directly in the model's representation space. We develop a framework that identifies safe and unsafe regions via the polytope's facets, enabling both detection and correction of unsafe outputs through geometric steering. Unlike existing approaches that modify model weights, SaP operates post-hoc in the representation space, preserving model capabilities while enforcing safety constraints. Experiments across multiple LLMs demonstrate that our method can effectively detect unethical inputs, reduce adversarial attack success rates while maintaining performance on standard tasks, thus highlighting the importance of having an explicit geometric model for safety. Analysis of the learned polytope facets reveals emergence of specialization in detecting different semantic notions of safety, providing interpretable insights into how safety is captured in LLMs' representation space.



## **24. Breaking the Gold Standard: Extracting Forgotten Data under Exact Unlearning in Large Language Models**

cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24379v1) [paper-pdf](http://arxiv.org/pdf/2505.24379v1)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large language models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard, believed to be robust against privacy-related attacks. In this paper, we challenge this assumption by introducing a novel data extraction attack that compromises even exact unlearning. Our method leverages both the pre- and post-unlearning models: by guiding the post-unlearning model using signals from the pre-unlearning model, we uncover patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints.



## **25. Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction**

cs.CL

22 pages, 10 figures, accepted to ACL 2025 findings

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2502.11084v2) [paper-pdf](http://arxiv.org/pdf/2502.11084v2)

**Authors**: Yuting Huang, Chengyuan Liu, Yifeng Feng, Yiquan Wu, Chao Wu, Fei Wu, Kun Kuang

**Abstract**: As Large Language Models (LLMs) are widely applied in various domains, the safety of LLMs is increasingly attracting attention to avoid their powerful capabilities being misused. Existing jailbreak methods create a forced instruction-following scenario, or search adversarial prompts with prefix or suffix tokens to achieve a specific representation manually or automatically. However, they suffer from low efficiency and explicit jailbreak patterns, far from the real deployment of mass attacks to LLMs. In this paper, we point out that simply rewriting the original instruction can achieve a jailbreak, and we find that this rewriting approach is learnable and transferable. We propose the Rewrite to Jailbreak (R2J) approach, a transferable black-box jailbreak method to attack LLMs by iteratively exploring the weakness of the LLMs and automatically improving the attacking strategy. The jailbreak is more efficient and hard to identify since no additional features are introduced. Extensive experiments and analysis demonstrate the effectiveness of R2J, and we find that the jailbreak is also transferable to multiple datasets and various types of models with only a few queries. We hope our work motivates further investigation of LLM safety. The code can be found at https://github.com/ythuang02/R2J/.



## **26. A Reward-driven Automated Webshell Malicious-code Generator for Red-teaming**

cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24252v1) [paper-pdf](http://arxiv.org/pdf/2505.24252v1)

**Authors**: Yizhong Ding

**Abstract**: Frequent cyber-attacks have elevated WebShell exploitation and defense to a critical research focus within network security. However, there remains a significant shortage of publicly available, well-categorized malicious-code datasets organized by obfuscation method. Existing malicious-code generation methods, which primarily rely on prompt engineering, often suffer from limited diversity and high redundancy in the payloads they produce. To address these limitations, we propose \textbf{RAWG}, a \textbf{R}eward-driven \textbf{A}utomated \textbf{W}ebshell Malicious-code \textbf{G}enerator designed for red-teaming applications. Our approach begins by categorizing webshell samples from common datasets into seven distinct types of obfuscation. We then employ a large language model (LLM) to extract and normalize key tokens from each sample, creating a standardized, high-quality corpus. Using this curated dataset, we perform supervised fine-tuning (SFT) on an open-source large model to enable the generation of diverse, highly obfuscated webshell malicious payloads. To further enhance generation quality, we apply Proximal Policy Optimization (PPO), treating malicious-code samples as "chosen" data and benign code as "rejected" data during reinforcement learning. Extensive experiments demonstrate that RAWG significantly outperforms current state-of-the-art methods in both payload diversity and escape effectiveness.



## **27. Safety Alignment Can Be Not Superficial With Explicit Safety Signals**

cs.CR

ICML 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.17072v2) [paper-pdf](http://arxiv.org/pdf/2505.17072v2)

**Authors**: Jianwei Li, Jung-Eun Kim

**Abstract**: Recent studies on the safety alignment of large language models (LLMs) have revealed that existing approaches often operate superficially, leaving models vulnerable to various adversarial attacks. Despite their significance, these studies generally fail to offer actionable solutions beyond data augmentation for achieving more robust safety mechanisms. This paper identifies a fundamental cause of this superficiality: existing alignment approaches often presume that models can implicitly learn a safety-related reasoning task during the alignment process, enabling them to refuse harmful requests. However, the learned safety signals are often diluted by other competing objectives, leading models to struggle with drawing a firm safety-conscious decision boundary when confronted with adversarial attacks. Based on this observation, by explicitly introducing a safety-related binary classification task and integrating its signals with our attention and decoding strategies, we eliminate this ambiguity and allow models to respond more responsibly to malicious queries. We emphasize that, with less than 0.2x overhead cost, our approach enables LLMs to assess the safety of both the query and the previously generated tokens at each necessary generating step. Extensive experiments demonstrate that our method significantly improves the resilience of LLMs against various adversarial attacks, offering a promising pathway toward more robust generative AI systems.



## **28. Benchmarking Foundation Models for Zero-Shot Biometric Tasks**

cs.CV

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24214v1) [paper-pdf](http://arxiv.org/pdf/2505.24214v1)

**Authors**: Redwan Sony, Parisa Farmanifard, Hamzeh Alzwairy, Nitish Shukla, Arun Ross

**Abstract**: The advent of foundation models, particularly Vision-Language Models (VLMs) and Multi-modal Large Language Models (MLLMs), has redefined the frontiers of artificial intelligence, enabling remarkable generalization across diverse tasks with minimal or no supervision. Yet, their potential in biometric recognition and analysis remains relatively underexplored. In this work, we introduce a comprehensive benchmark that evaluates the zero-shot and few-shot performance of state-of-the-art publicly available VLMs and MLLMs across six biometric tasks spanning the face and iris modalities: face verification, soft biometric attribute prediction (gender and race), iris recognition, presentation attack detection (PAD), and face manipulation detection (morphs and deepfakes). A total of 41 VLMs were used in this evaluation. Experiments show that embeddings from these foundation models can be used for diverse biometric tasks with varying degrees of success. For example, in the case of face verification, a True Match Rate (TMR) of 96.77 percent was obtained at a False Match Rate (FMR) of 1 percent on the Labeled Face in the Wild (LFW) dataset, without any fine-tuning. In the case of iris recognition, the TMR at 1 percent FMR on the IITD-R-Full dataset was 97.55 percent without any fine-tuning. Further, we show that applying a simple classifier head to these embeddings can help perform DeepFake detection for faces, Presentation Attack Detection (PAD) for irides, and extract soft biometric attributes like gender and ethnicity from faces with reasonably high accuracy. This work reiterates the potential of pretrained models in achieving the long-term vision of Artificial General Intelligence.



## **29. On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains**

cs.CR

Accepted by ICML 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2409.17275v2) [paper-pdf](http://arxiv.org/pdf/2409.17275v2)

**Authors**: Xun Xian, Ganghua Wang, Xuan Bi, Jayanth Srinivasa, Ashish Kundu, Charles Fleming, Mingyi Hong, Jie Ding

**Abstract**: Retrieval-Augmented Generation (RAG) has been empirically shown to enhance the performance of large language models (LLMs) in knowledge-intensive domains such as healthcare, finance, and legal contexts. Given a query, RAG retrieves relevant documents from a corpus and integrates them into the LLMs' generation process. In this study, we investigate the adversarial robustness of RAG, focusing specifically on examining the retrieval system. First, across 225 different setup combinations of corpus, retriever, query, and targeted information, we show that retrieval systems are vulnerable to universal poisoning attacks in medical Q\&A. In such attacks, adversaries generate poisoned documents containing a broad spectrum of targeted information, such as personally identifiable information. When these poisoned documents are inserted into a corpus, they can be accurately retrieved by any users, as long as attacker-specified queries are used. To understand this vulnerability, we discovered that the deviation from the query's embedding to that of the poisoned document tends to follow a pattern in which the high similarity between the poisoned document and the query is retained, thereby enabling precise retrieval. Based on these findings, we develop a new detection-based defense to ensure the safe use of RAG. Through extensive experiments spanning various Q\&A domains, we observed that our proposed method consistently achieves excellent detection rates in nearly all cases.



## **30. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

cs.CR

Accepted by ICLR 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2410.02644v4) [paper-pdf](http://arxiv.org/pdf/2410.02644v4)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.



## **31. Latent-space adversarial training with post-aware calibration for defending large language models against jailbreak attacks**

cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2501.10639v3) [paper-pdf](http://arxiv.org/pdf/2501.10639v3)

**Authors**: Xin Yi, Yue Li, Dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Ensuring safety alignment is a critical requirement for large language models (LLMs), particularly given increasing deployment in real-world applications. Despite considerable advancements, LLMs remain susceptible to jailbreak attacks, which exploit system vulnerabilities to circumvent safety measures and elicit harmful or inappropriate outputs. Furthermore, while adversarial training-based defense methods have shown promise, a prevalent issue is the unintended over-defense behavior, wherein models excessively reject benign queries, significantly undermining their practical utility. To address these limitations, we introduce LATPC, a Latent-space Adversarial Training with Post-aware Calibration framework. LATPC dynamically identifies safety-critical latent dimensions by contrasting harmful and benign inputs, enabling the adaptive construction of targeted refusal feature removal attacks. This mechanism allows adversarial training to concentrate on real-world jailbreak tactics that disguise harmful queries as benign ones. During inference, LATPC employs an efficient embedding-level calibration mechanism to minimize over-defense behaviors with negligible computational overhead. Experimental results across five types of disguise-based jailbreak attacks demonstrate that LATPC achieves a superior balance between safety and utility compared to existing defense frameworks. Further analysis demonstrates the effectiveness of leveraging safety-critical dimensions in developing robust defense methods against jailbreak attacks.



## **32. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

cs.CV

ICML 2025

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.05528v3) [paper-pdf](http://arxiv.org/pdf/2505.05528v3)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.



## **33. LLM Agents Should Employ Security Principles**

cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.24019v1) [paper-pdf](http://arxiv.org/pdf/2505.24019v1)

**Authors**: Kaiyuan Zhang, Zian Su, Pin-Yu Chen, Elisa Bertino, Xiangyu Zhang, Ninghui Li

**Abstract**: Large Language Model (LLM) agents show considerable promise for automating complex tasks using contextual reasoning; however, interactions involving multiple agents and the system's susceptibility to prompt injection and other forms of context manipulation introduce new vulnerabilities related to privacy leakage and system exploitation. This position paper argues that the well-established design principles in information security, which are commonly referred to as security principles, should be employed when deploying LLM agents at scale. Design principles such as defense-in-depth, least privilege, complete mediation, and psychological acceptability have helped guide the design of mechanisms for securing information systems over the last five decades, and we argue that their explicit and conscientious adoption will help secure agentic systems. To illustrate this approach, we introduce AgentSandbox, a conceptual framework embedding these security principles to provide safeguards throughout an agent's life-cycle. We evaluate with state-of-the-art LLMs along three dimensions: benign utility, attack utility, and attack success rate. AgentSandbox maintains high utility for its intended functions under both benign and adversarial evaluations while substantially mitigating privacy risks. By embedding secure design principles as foundational elements within emerging LLM agent protocols, we aim to promote trustworthy agent ecosystems aligned with user privacy expectations and evolving regulatory requirements.



## **34. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

cs.LG

22 pages

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.22307v2) [paper-pdf](http://arxiv.org/pdf/2410.22307v2)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: The ever-increasing size of open-source Large Language Models (LLMs) renders local deployment impractical for individual users. Decentralized computing has emerged as a cost-effective solution, allowing individuals and small companies to perform LLM inference for users using surplus computational power. However, a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby benefiting from cost savings. We introduce SVIP, a secret-based verifiable LLM inference protocol. Unlike existing solutions based on cryptographic or game-theoretic techniques, our method is computationally effective and does not rest on strong assumptions. Our protocol requires the computing provider to return both the generated text and processed hidden representations from LLMs. We then train a proxy task on these representations, effectively transforming them into a unique model identifier. With our protocol, users can reliably verify whether the computing provider is acting honestly. A carefully integrated secret mechanism further strengthens its security. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per prompt query for verification.



## **35. MCP Safety Training: Learning to Refuse Falsely Benign MCP Exploits using Improved Preference Alignment**

cs.LG

27 pages, 19 figures, 4 tables

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23634v1) [paper-pdf](http://arxiv.org/pdf/2505.23634v1)

**Authors**: John Halloran

**Abstract**: The model context protocol (MCP) has been widely adapted as an open standard enabling the seamless integration of generative AI agents. However, recent work has shown the MCP is susceptible to retrieval-based "falsely benign" attacks (FBAs), allowing malicious system access and credential theft, but requiring that users download compromised files directly to their systems. Herein, we show that the threat model of MCP-based attacks is significantly broader than previously thought, i.e., attackers need only post malicious content online to deceive MCP agents into carrying out their attacks on unsuspecting victims' systems.   To improve alignment guardrails against such attacks, we introduce a new MCP dataset of FBAs and (truly) benign samples to explore the effectiveness of direct preference optimization (DPO) for the refusal training of large language models (LLMs). While DPO improves model guardrails against such attacks, we show that the efficacy of refusal learning varies drastically depending on the model's original post-training alignment scheme--e.g., GRPO-based LLMs learn to refuse extremely poorly. Thus, to further improve FBA refusals, we introduce Retrieval Augmented Generation for Preference alignment (RAG-Pref), a novel preference alignment strategy based on RAG. We show that RAG-Pref significantly improves the ability of LLMs to refuse FBAs, particularly when combined with DPO alignment, thus drastically improving guardrails against MCP-based attacks.



## **36. Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models**

cs.CR

This paper is accepted by ACL 2025 main conference

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23561v1) [paper-pdf](http://arxiv.org/pdf/2505.23561v1)

**Authors**: Zenghui Yuan, Yangming Xu, Jiawen Shi, Pan Zhou, Lichao Sun

**Abstract**: Model merging for Large Language Models (LLMs) directly fuses the parameters of different models finetuned on various tasks, creating a unified model for multi-domain tasks. However, due to potential vulnerabilities in models available on open-source platforms, model merging is susceptible to backdoor attacks. In this paper, we propose Merge Hijacking, the first backdoor attack targeting model merging in LLMs. The attacker constructs a malicious upload model and releases it. Once a victim user merges it with any other models, the resulting merged model inherits the backdoor while maintaining utility across tasks. Merge Hijacking defines two main objectives-effectiveness and utility-and achieves them through four steps. Extensive experiments demonstrate the effectiveness of our attack across different models, merging algorithms, and tasks. Additionally, we show that the attack remains effective even when merging real-world models. Moreover, our attack demonstrates robustness against two inference-time defenses (Paraphrasing and CLEANGEN) and one training-time defense (Fine-pruning).



## **37. SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents**

cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23559v1) [paper-pdf](http://arxiv.org/pdf/2505.23559v1)

**Authors**: Kunlun Zhu, Jiaxun Zhang, Ziheng Qi, Nuoxing Shang, Zijia Liu, Peixuan Han, Yue Su, Haofei Yu, Jiaxuan You

**Abstract**: Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github.com/ulab-uiuc/SafeScientist. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.}



## **38. Hijacking Large Language Models via Adversarial In-Context Learning**

cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2311.09948v3) [paper-pdf](http://arxiv.org/pdf/2311.09948v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Prashant Khanduri, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the preconditioned prompts. Despite its promising performance, crafted adversarial attacks pose a notable threat to the robustness of LLMs. Existing attacks are either easy to detect, require a trigger in user input, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable prompt injection attack against ICL, aiming to hijack LLMs to generate the target output or elicit harmful responses. In our threat model, the hacker acts as a model publisher who leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos via prompt injection. We also propose effective defense strategies using a few shots of clean demos, enhancing the robustness of LLMs during ICL. Extensive experimental results across various classification and jailbreak tasks demonstrate the effectiveness of the proposed attack and defense strategies. This work highlights the significant security vulnerabilities of LLMs during ICL and underscores the need for further in-depth studies.



## **39. Learning to Poison Large Language Models for Downstream Manipulation**

cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2402.13459v3) [paper-pdf](http://arxiv.org/pdf/2402.13459v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where the adversary inserts backdoor triggers into training data to manipulate outputs. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the supervised fine-tuning (SFT) process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various language model tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during SFT of LLMs and the necessity of safeguarding LLMs against data poisoning attacks.



## **40. Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models**

cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2412.16555v3) [paper-pdf](http://arxiv.org/pdf/2412.16555v3)

**Authors**: Yanxu Mao, Peipei Liu, Tiehan Cui, Zhaoteng Yan, Congying Liu, Datao You

**Abstract**: Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead.



## **41. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.11647v2) [paper-pdf](http://arxiv.org/pdf/2502.11647v2)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.



## **42. Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models**

cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23404v1) [paper-pdf](http://arxiv.org/pdf/2505.23404v1)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin

**Abstract**: Adversarial attacks on Large Language Models (LLMs) via jailbreaking techniques-methods that circumvent their built-in safety and ethical constraints-have emerged as a critical challenge in AI security. These attacks compromise the reliability of LLMs by exploiting inherent weaknesses in their comprehension capabilities. This paper investigates the efficacy of jailbreaking strategies that are specifically adapted to the diverse levels of understanding exhibited by different LLMs. We propose the Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models, a novel framework that classifies LLMs into Type I and Type II categories according to their semantic comprehension abilities. For each category, we design tailored jailbreaking strategies aimed at leveraging their vulnerabilities to facilitate successful attacks. Extensive experiments conducted on multiple LLMs demonstrate that our adaptive strategy markedly improves the success rate of jailbreaking. Notably, our approach achieves an exceptional 98.9% success rate in jailbreaking GPT-4o(29 May 2025 release)



## **43. Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction**

cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.17541v2) [paper-pdf](http://arxiv.org/pdf/2502.17541v2)

**Authors**: Michal Bravansky, Vaclav Kubon, Suhas Hariharan, Robert Kirk

**Abstract**: Interpreting data is central to modern research. Large language models (LLMs) show promise in providing such natural language interpretations of data, yet simple feature extraction methods such as prompting often fail to produce accurate and versatile descriptions for diverse datasets and lack control over granularity and scale. To address these limitations, we propose a domain-agnostic method for dataset featurization that provides precise control over the number of features extracted while maintaining compact and descriptive representations comparable to human labeling. Our method optimizes the selection of informative binary features by evaluating the ability of an LLM to reconstruct the original data using those features. We demonstrate its effectiveness in dataset modeling tasks and through two case studies: (1) Constructing a feature representation of jailbreak tactics that compactly captures both the effectiveness and diversity of a larger set of human-crafted attacks; and (2) automating the discovery of features that align with human preferences, achieving accuracy and robustness comparable to human-crafted features. Moreover, we show that the pipeline scales effectively, improving as additional features are sampled, making it suitable for large and diverse datasets.



## **44. Disrupting Vision-Language Model-Driven Navigation Services via Adversarial Object Fusion**

cs.CR

Under review

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23266v1) [paper-pdf](http://arxiv.org/pdf/2505.23266v1)

**Authors**: Chunlong Xie, Jialing He, Shangwei Guo, Jiacheng Wang, Shudong Zhang, Tianwei Zhang, Tao Xiang

**Abstract**: We present Adversarial Object Fusion (AdvOF), a novel attack framework targeting vision-and-language navigation (VLN) agents in service-oriented environments by generating adversarial 3D objects. While foundational models like Large Language Models (LLMs) and Vision Language Models (VLMs) have enhanced service-oriented navigation systems through improved perception and decision-making, their integration introduces vulnerabilities in mission-critical service workflows. Existing adversarial attacks fail to address service computing contexts, where reliability and quality-of-service (QoS) are paramount. We utilize AdvOF to investigate and explore the impact of adversarial environments on the VLM-based perception module of VLN agents. In particular, AdvOF first precisely aggregates and aligns the victim object positions in both 2D and 3D space, defining and rendering adversarial objects. Then, we collaboratively optimize the adversarial object with regularization between the adversarial and victim object across physical properties and VLM perceptions. Through assigning importance weights to varying views, the optimization is processed stably and multi-viewedly by iterative fusions from local updates and justifications. Our extensive evaluations demonstrate AdvOF can effectively degrade agent performance under adversarial conditions while maintaining minimal interference with normal navigation tasks. This work advances the understanding of service security in VLM-powered navigation systems, providing computational foundations for robust service composition in physical-world deployments.



## **45. Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking**

cs.CL

18 pages

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.12970v2) [paper-pdf](http://arxiv.org/pdf/2502.12970v2)

**Authors**: Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha

**Abstract**: Large Reasoning Models (LRMs) have demonstrated impressive performances across diverse domains. However, how safety of Large Language Models (LLMs) benefits from enhanced reasoning capabilities against jailbreak queries remains unexplored. To bridge this gap, in this paper, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates a safety-aware reasoning mechanism into LLMs' generation. This enables self-evaluation at each step of the reasoning process, forming safety pivot tokens as indicators of the safety status of responses. Furthermore, in order to improve the accuracy of predicting pivot tokens, we propose Contrastive Pivot Optimization (CPO), which enhances the model's perception of the safety status of given dialogues. LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their safety capabilities defending jailbreak attacks. Extensive experiments demonstrate that R2D effectively mitigates various attacks and improves overall safety, while maintaining the original performances. This highlights the substantial potential of safety-aware reasoning in improving robustness of LRMs and LLMs against various jailbreaks.



## **46. Jailbreaking to Jailbreak**

cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.09638v2) [paper-pdf](http://arxiv.org/pdf/2502.09638v2)

**Authors**: Jeremy Kritz, Vaughn Robinson, Robert Vacareanu, Bijan Varjavand, Michael Choi, Bobby Gogov, Scale Red Team, Summer Yue, Willow E. Primack, Zifan Wang

**Abstract**: Large Language Models (LLMs) can be used to red team other models (e.g. jailbreaking) to elicit harmful contents. While prior works commonly employ open-weight models or private uncensored models for doing jailbreaking, as the refusal-training of strong LLMs (e.g. OpenAI o3) refuse to help jailbreaking, our work turn (almost) any black-box LLMs into attackers. The resulting $J_2$ (jailbreaking-to-jailbreak) attackers can effectively jailbreak the safeguard of target models using various strategies, both created by themselves or from expert human red teamers. In doing so, we show their strong but under-researched jailbreaking capabilities. Our experiments demonstrate that 1) prompts used to create $J_2$ attackers transfer across almost all black-box models; 2) an $J_2$ attacker can jailbreak a copy of itself, and this vulnerability develops rapidly over the past 12 months; 3) reasong models, such as Sonnet-3.7, are strong $J_2$ attackers compared to others. For example, when used against the safeguard of GPT-4o, $J_2$ (Sonnet-3.7) achieves 0.975 attack success rate (ASR), which matches expert human red teamers and surpasses the state-of-the-art algorithm-based attacks. Among $J_2$ attackers, $J_2$ (o3) achieves highest ASR (0.605) against Sonnet-3.5, one of the most robust models.



## **47. OMNIGUARD: An Efficient Approach for AI Safety Moderation Across Modalities**

cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23856v1) [paper-pdf](http://arxiv.org/pdf/2505.23856v1)

**Authors**: Sahil Verma, Keegan Hines, Jeff Bilmes, Charlotte Siska, Luke Zettlemoyer, Hila Gonen, Chandan Singh

**Abstract**: The emerging capabilities of large language models (LLMs) have sparked concerns about their immediate potential for harmful misuse. The core approach to mitigate these concerns is the detection of harmful queries to the model. Current detection approaches are fallible, and are particularly susceptible to attacks that exploit mismatched generalization of model capabilities (e.g., prompts in low-resource languages or prompts provided in non-text modalities such as image and audio). To tackle this challenge, we propose OMNIGUARD, an approach for detecting harmful prompts across languages and modalities. Our approach (i) identifies internal representations of an LLM/MLLM that are aligned across languages or modalities and then (ii) uses them to build a language-agnostic or modality-agnostic classifier for detecting harmful prompts. OMNIGUARD improves harmful prompt classification accuracy by 11.57\% over the strongest baseline in a multilingual setting, by 20.44\% for image-based prompts, and sets a new SOTA for audio-based prompts. By repurposing embeddings computed during generation, OMNIGUARD is also very efficient ($\approx 120 \times$ faster than the next fastest baseline). Code and data are available at: https://github.com/vsahil/OmniGuard.



## **48. DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors**

cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23001v1) [paper-pdf](http://arxiv.org/pdf/2505.23001v1)

**Authors**: Yize Cheng, Wenxiao Wang, Mazda Moayeri, Soheil Feizi

**Abstract**: Open benchmarks are essential for evaluating and advancing large language models, offering reproducibility and transparency. However, their accessibility makes them likely targets of test set contamination. In this work, we introduce DyePack, a framework that leverages backdoor attacks to identify models that used benchmark test sets during training, without requiring access to the loss, logits, or any internal details of the model. Like how banks mix dye packs with their money to mark robbers, DyePack mixes backdoor samples with the test data to flag models that trained on it. We propose a principled design incorporating multiple backdoors with stochastic targets, enabling exact false positive rate (FPR) computation when flagging every model. This provably prevents false accusations while providing strong evidence for every detected case of contamination. We evaluate DyePack on five models across three datasets, covering both multiple-choice and open-ended generation tasks. For multiple-choice questions, it successfully detects all contaminated models with guaranteed FPRs as low as 0.000073% on MMLU-Pro and 0.000017% on Big-Bench-Hard using eight backdoors. For open-ended generation tasks, it generalizes well and identifies all contaminated models on Alpaca with a guaranteed false positive rate of just 0.127% using six backdoors.



## **49. Revisiting Multi-Agent Debate as Test-Time Scaling: A Systematic Study of Conditional Effectiveness**

cs.AI

Preprint, under review

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.22960v1) [paper-pdf](http://arxiv.org/pdf/2505.22960v1)

**Authors**: Yongjin Yang, Euiin Yi, Jongwoo Ko, Kimin Lee, Zhijing Jin, Se-Young Yun

**Abstract**: The remarkable growth in large language model (LLM) capabilities has spurred exploration into multi-agent systems, with debate frameworks emerging as a promising avenue for enhanced problem-solving. These multi-agent debate (MAD) approaches, where agents collaboratively present, critique, and refine arguments, potentially offer improved reasoning, robustness, and diverse perspectives over monolithic models. Despite prior studies leveraging MAD, a systematic understanding of its effectiveness compared to self-agent methods, particularly under varying conditions, remains elusive. This paper seeks to fill this gap by conceptualizing MAD as a test-time computational scaling technique, distinguished by collaborative refinement and diverse exploration capabilities. We conduct a comprehensive empirical investigation comparing MAD with strong self-agent test-time scaling baselines on mathematical reasoning and safety-related tasks. Our study systematically examines the influence of task difficulty, model scale, and agent diversity on MAD's performance. Key findings reveal that, for mathematical reasoning, MAD offers limited advantages over self-agent scaling but becomes more effective with increased problem difficulty and decreased model capability, while agent diversity shows little benefit. Conversely, for safety tasks, MAD's collaborative refinement can increase vulnerability, but incorporating diverse agent configurations facilitates a gradual reduction in attack success through the collaborative refinement process. We believe our findings provide critical guidance for the future development of more effective and strategically deployed MAD systems.



## **50. Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates**

cs.CL

ACL 2025 Main. Code is released at  https://vision.snu.ac.kr/projects/mac

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22943v1) [paper-pdf](http://arxiv.org/pdf/2505.22943v1)

**Authors**: Jaewoo Ahn, Heeseung Yun, Dayoon Ko, Gunhee Kim

**Abstract**: While pre-trained multimodal representations (e.g., CLIP) have shown impressive capabilities, they exhibit significant compositional vulnerabilities leading to counterintuitive judgments. We introduce Multimodal Adversarial Compositionality (MAC), a benchmark that leverages large language models (LLMs) to generate deceptive text samples to exploit these vulnerabilities across different modalities and evaluates them through both sample-wise attack success rate and group-wise entropy-based diversity. To improve zero-shot methods, we propose a self-training approach that leverages rejection-sampling fine-tuning with diversity-promoting filtering, which enhances both attack success rate and sample diversity. Using smaller language models like Llama-3.1-8B, our approach demonstrates superior performance in revealing compositional vulnerabilities across various multimodal representations, including images, videos, and audios.



