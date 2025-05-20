# Latest Adversarial Attack Papers
**update at 2025-05-20 10:32:49**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants**

cs.CR

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2503.14281v2) [paper-pdf](http://arxiv.org/pdf/2503.14281v2)

**Authors**: Adam Štorek, Mukur Gupta, Noopur Bhatt, Aditya Gupta, Janie Kim, Prashast Srivastava, Suman Jana

**Abstract**: AI coding assistants are widely used for tasks like code generation. These tools now require large and complex contexts, automatically sourced from various origins$\unicode{x2014}$across files, projects, and contributors$\unicode{x2014}$forming part of the prompt fed to underlying LLMs. This automatic context-gathering introduces new vulnerabilities, allowing attackers to subtly poison input to compromise the assistant's outputs, potentially generating vulnerable code or introducing critical errors. We propose a novel attack, Cross-Origin Context Poisoning (XOXO), that is challenging to detect as it relies on adversarial code modifications that are semantically equivalent. Traditional program analysis techniques struggle to identify these perturbations since the semantics of the code remains correct, making it appear legitimate. This allows attackers to manipulate coding assistants into producing incorrect outputs, while shifting the blame to the victim developer. We introduce a novel, task-agnostic, black-box attack algorithm GCGS that systematically searches the transformation space using a Cayley Graph, achieving a 75.72% attack success rate on average across five tasks and eleven models, including GPT 4.1 and Claude 3.5 Sonnet v2 used by popular AI coding assistants. Furthermore, defenses like adversarial fine-tuning are ineffective against our attack, underscoring the need for new security measures in LLM-powered coding tools.



## **2. Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks**

cs.CL

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.13348v1) [paper-pdf](http://arxiv.org/pdf/2505.13348v1)

**Authors**: Narek Maloyan, Bislan Ashinov, Dmitry Namiot

**Abstract**: Large Language Models (LLMs) are increasingly employed as evaluators (LLM-as-a-Judge) for assessing the quality of machine-generated text. This paradigm offers scalability and cost-effectiveness compared to human annotation. However, the reliability and security of such systems, particularly their robustness against adversarial manipulations, remain critical concerns. This paper investigates the vulnerability of LLM-as-a-Judge architectures to prompt-injection attacks, where malicious inputs are designed to compromise the judge's decision-making process. We formalize two primary attack strategies: Comparative Undermining Attack (CUA), which directly targets the final decision output, and Justification Manipulation Attack (JMA), which aims to alter the model's generated reasoning. Using the Greedy Coordinate Gradient (GCG) optimization method, we craft adversarial suffixes appended to one of the responses being compared. Experiments conducted on the MT-Bench Human Judgments dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable effectiveness. These findings highlight substantial vulnerabilities in current LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and further research into adversarial evaluation and trustworthiness in LLM-based assessment frameworks.



## **3. FlowPure: Continuous Normalizing Flows for Adversarial Purification**

cs.LG

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.13280v1) [paper-pdf](http://arxiv.org/pdf/2505.13280v1)

**Authors**: Elias Collaert, Abel Rodríguez, Sander Joos, Lieven Desmet, Vera Rimmer

**Abstract**: Despite significant advancements in the area, adversarial robustness remains a critical challenge in systems employing machine learning models. The removal of adversarial perturbations at inference time, known as adversarial purification, has emerged as a promising defense strategy. To achieve this, state-of-the-art methods leverage diffusion models that inject Gaussian noise during a forward process to dilute adversarial perturbations, followed by a denoising step to restore clean samples before classification. In this work, we propose FlowPure, a novel purification method based on Continuous Normalizing Flows (CNFs) trained with Conditional Flow Matching (CFM) to learn mappings from adversarial examples to their clean counterparts. Unlike prior diffusion-based approaches that rely on fixed noise processes, FlowPure can leverage specific attack knowledge to improve robustness under known threats, while also supporting a more general stochastic variant trained on Gaussian perturbations for settings where such knowledge is unavailable. Experiments on CIFAR-10 and CIFAR-100 demonstrate that our method outperforms state-of-the-art purification-based defenses in preprocessor-blind and white-box scenarios, and can do so while fully preserving benign accuracy in the former. Moreover, our results show that not only is FlowPure a highly effective purifier but it also holds a strong potential for adversarial detection, identifying preprocessor-blind PGD samples with near-perfect accuracy.



## **4. Constrained Adversarial Learning for Automated Software Testing: a literature review**

cs.SE

36 pages, 4 tables, 2 figures, Discover Applied Sciences journal

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2303.07546v3) [paper-pdf](http://arxiv.org/pdf/2303.07546v3)

**Authors**: João Vitorino, Tiago Dias, Tiago Fonseca, Eva Maia, Isabel Praça

**Abstract**: It is imperative to safeguard computer applications and information systems against the growing number of cyber-attacks. Automated software testing tools can be developed to quickly analyze many lines of code and detect vulnerabilities by generating function-specific testing data. This process draws similarities to the constrained adversarial examples generated by adversarial machine learning methods, so there could be significant benefits to the integration of these methods in testing tools to identify possible attack vectors. Therefore, this literature review is focused on the current state-of-the-art of constrained data generation approaches applied for adversarial learning and software testing, aiming to guide researchers and developers to enhance their software testing tools with adversarial testing methods and improve the resilience and robustness of their information systems. The found approaches were systematized, and the advantages and limitations of those specific for white-box, grey-box, and black-box testing were analyzed, identifying research gaps and opportunities to automate the testing tools with data generated by adversarial attacks.



## **5. Test-time Adversarial Defense with Opposite Adversarial Path and High Attack Time Cost**

cs.LG

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2410.16805v2) [paper-pdf](http://arxiv.org/pdf/2410.16805v2)

**Authors**: Cheng-Han Yeh, Kuanchun Yu, Chun-Shien Lu

**Abstract**: Deep learning models are known to be vulnerable to adversarial attacks by injecting sophisticated designed perturbations to input data. Training-time defenses still exhibit a significant performance gap between natural accuracy and robust accuracy. In this paper, we investigate a new test-time adversarial defense method via diffusion-based recovery along opposite adversarial paths (OAPs). We present a purifier that can be plugged into a pre-trained model to resist adversarial attacks. Different from prior arts, the key idea is excessive denoising or purification by integrating the opposite adversarial direction with reverse diffusion to push the input image further toward the opposite adversarial direction. For the first time, we also exemplify the pitfall of conducting AutoAttack (Rand) for diffusion-based defense methods. Through the lens of time complexity, we examine the trade-off between the effectiveness of adaptive attack and its computation complexity against our defense. Experimental evaluation along with time cost analysis verifies the effectiveness of the proposed method.



## **6. Countermeasure against Detector Blinding Attack with Secret Key Leakage Estimation**

quant-ph

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12974v1) [paper-pdf](http://arxiv.org/pdf/2505.12974v1)

**Authors**: Dmitry M. Melkonian, Daniil S. Bulavkin, Kirill E. Bugai, Kirill A. Balygin, Dmitriy A. Dvoretskiy

**Abstract**: We present a countermeasure against the detector blinding attack (DBA) utilizing statistical analysis of error and double-click events accumulated during a quantum key distribution session under randomized modulation of single-photon avalanche diode (SPAD) detection efficiencies via gate voltage manipulation. Building upon prior work demonstrating the ineffectiveness of this countermeasure against continuous-wave (CW) DBA, we extend the analysis to evaluate its performance against pulsed DBA. Our findings reveal an approximately 25 dB increase in the trigger pulse energies difference between high and low gate voltage applied under pulsed DBA conditions compared to CW DBA. This heightened difference enables a re-evaluation of the feasibility of utilizing SPAD detection probability variations as a countermeasure and makes it possible to estimate the fraction of bits compromised by an adversary during pulsed DBA.



## **7. DICTION:DynamIC robusT whIte bOx watermarkiNg scheme for deep neural networks**

cs.CR

24 pages, 4 figures, PrePrint

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2210.15745v2) [paper-pdf](http://arxiv.org/pdf/2210.15745v2)

**Authors**: Reda Bellafqira, Gouenou Coatrieux

**Abstract**: Deep neural network (DNN) watermarking is a suitable method for protecting the ownership of deep learning (DL) models. It secretly embeds an identifier (watermark) within the model, which can be retrieved by the owner to prove ownership. In this paper, we first provide a unified framework for white box DNN watermarking schemes. It includes current state-of-the-art methods outlining their theoretical inter-connections. Next, we introduce DICTION, a new white-box Dynamic Robust watermarking scheme, we derived from this framework. Its main originality stands on a generative adversarial network (GAN) strategy where the watermark extraction function is a DNN trained as a GAN discriminator taking the target model to watermark as a GAN generator with a latent space as the input of the GAN trigger set. DICTION can be seen as a generalization of DeepSigns which, to the best of our knowledge, is the only other Dynamic white-box watermarking scheme from the literature. Experiments conducted on the same model test set as Deepsigns demonstrate that our scheme achieves much better performance. Especially, with DICTION, one can increase the watermark capacity while preserving the target model accuracy at best and simultaneously ensuring strong watermark robustness against a wide range of watermark removal and detection attacks.



## **8. Language Models That Walk the Talk: A Framework for Formal Fairness Certificates**

cs.AI

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12767v1) [paper-pdf](http://arxiv.org/pdf/2505.12767v1)

**Authors**: Danqing Chen, Tobias Ladner, Ahmed Rayen Mhadhbi, Matthias Althoff

**Abstract**: As large language models become integral to high-stakes applications, ensuring their robustness and fairness is critical. Despite their success, large language models remain vulnerable to adversarial attacks, where small perturbations, such as synonym substitutions, can alter model predictions, posing risks in fairness-critical areas, such as gender bias mitigation, and safety-critical areas, such as toxicity detection. While formal verification has been explored for neural networks, its application to large language models remains limited. This work presents a holistic verification framework to certify the robustness of transformer-based language models, with a focus on ensuring gender fairness and consistent outputs across different gender-related terms. Furthermore, we extend this methodology to toxicity detection, offering formal guarantees that adversarially manipulated toxic inputs are consistently detected and appropriately censored, thereby ensuring the reliability of moderation systems. By formalizing robustness within the embedding space, this work strengthens the reliability of language models in ethical AI deployment and content moderation.



## **9. BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models**

cs.AI

22 pages

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2408.12798v2) [paper-pdf](http://arxiv.org/pdf/2408.12798v2)

**Authors**: Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, Jun Sun

**Abstract**: Generative large language models (LLMs) have achieved state-of-the-art results on a wide range of tasks, yet they remain susceptible to backdoor attacks: carefully crafted triggers in the input can manipulate the model to produce adversary-specified outputs. While prior research has predominantly focused on backdoor risks in vision and classification settings, the vulnerability of LLMs in open-ended text generation remains underexplored. To fill this gap, we introduce BackdoorLLM (Our BackdoorLLM benchmark was awarded First Prize in the SafetyBench competition, https://www.mlsafety.org/safebench/winners, organized by the Center for AI Safety, https://safe.ai/.), the first comprehensive benchmark for systematically evaluating backdoor threats in text-generation LLMs. BackdoorLLM provides: (i) a unified repository of benchmarks with a standardized training and evaluation pipeline; (ii) a diverse suite of attack modalities, including data poisoning, weight poisoning, hidden-state manipulation, and chain-of-thought hijacking; (iii) over 200 experiments spanning 8 distinct attack strategies, 7 real-world scenarios, and 6 model architectures; (iv) key insights into the factors that govern backdoor effectiveness and failure modes in LLMs; and (v) a defense toolkit encompassing 7 representative mitigation techniques. Our code and datasets are available at https://github.com/bboylyg/BackdoorLLM. We will continuously incorporate emerging attack and defense methodologies to support the research in advancing the safety and reliability of LLMs.



## **10. Bullying the Machine: How Personas Increase LLM Vulnerability**

cs.AI

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12692v1) [paper-pdf](http://arxiv.org/pdf/2505.12692v1)

**Authors**: Ziwei Xu, Udit Sanghi, Mohan Kankanhalli

**Abstract**: Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies.



## **11. RoVo: Robust Voice Protection Against Unauthorized Speech Synthesis with Embedding-Level Perturbations**

cs.LG

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12686v1) [paper-pdf](http://arxiv.org/pdf/2505.12686v1)

**Authors**: Seungmin Kim, Sohee Park, Donghyun Kim, Jisu Lee, Daeseon Choi

**Abstract**: With the advancement of AI-based speech synthesis technologies such as Deep Voice, there is an increasing risk of voice spoofing attacks, including voice phishing and fake news, through unauthorized use of others' voices. Existing defenses that inject adversarial perturbations directly into audio signals have limited effectiveness, as these perturbations can easily be neutralized by speech enhancement methods. To overcome this limitation, we propose RoVo (Robust Voice), a novel proactive defense technique that injects adversarial perturbations into high-dimensional embedding vectors of audio signals, reconstructing them into protected speech. This approach effectively defends against speech synthesis attacks and also provides strong resistance to speech enhancement models, which represent a secondary attack threat.   In extensive experiments, RoVo increased the Defense Success Rate (DSR) by over 70% compared to unprotected speech, across four state-of-the-art speech synthesis models. Specifically, RoVo achieved a DSR of 99.5% on a commercial speaker-verification API, effectively neutralizing speech synthesis attack. Moreover, RoVo's perturbations remained robust even under strong speech enhancement conditions, outperforming traditional methods. A user study confirmed that RoVo preserves both naturalness and usability of protected speech, highlighting its effectiveness in complex and evolving threat scenarios.



## **12. On the Mechanisms of Adversarial Data Augmentation for Robust and Adaptive Transfer Learning**

cs.LG

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12681v1) [paper-pdf](http://arxiv.org/pdf/2505.12681v1)

**Authors**: Hana Satou, Alan Mitkiy

**Abstract**: Transfer learning across domains with distribution shift remains a fundamental challenge in building robust and adaptable machine learning systems. While adversarial perturbations are traditionally viewed as threats that expose model vulnerabilities, recent studies suggest that they can also serve as constructive tools for data augmentation. In this work, we systematically investigate the role of adversarial data augmentation (ADA) in enhancing both robustness and adaptivity in transfer learning settings. We analyze how adversarial examples, when used strategically during training, improve domain generalization by enriching decision boundaries and reducing overfitting to source-domain-specific features. We further propose a unified framework that integrates ADA with consistency regularization and domain-invariant representation learning. Extensive experiments across multiple benchmark datasets -- including VisDA, DomainNet, and Office-Home -- demonstrate that our method consistently improves target-domain performance under both unsupervised and few-shot domain adaptation settings. Our results highlight a constructive perspective of adversarial learning, transforming perturbation from a destructive attack into a regularizing force for cross-domain transferability.



## **13. Spiking Neural Network: a low power solution for physical layer authentication**

cs.LG

11 pages, 7 figures and 2 pages

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12647v1) [paper-pdf](http://arxiv.org/pdf/2505.12647v1)

**Authors**: Jung Hoon Lee, Sujith Vijayan

**Abstract**: Deep learning (DL) is a powerful tool that can solve complex problems, and thus, it seems natural to assume that DL can be used to enhance the security of wireless communication. However, deploying DL models to edge devices in wireless networks is challenging, as they require significant amounts of computing and power resources. Notably, Spiking Neural Networks (SNNs) are known to be efficient in terms of power consumption, meaning they can be an alternative platform for DL models for edge devices. In this study, we ask if SNNs can be used in physical layer authentication. Our evaluation suggests that SNNs can learn unique physical properties (i.e., `fingerprints') of RF transmitters and use them to identify individual devices. Furthermore, we find that SNNs are also vulnerable to adversarial attacks and that an autoencoder can be used clean out adversarial perturbations to harden SNNs against them.



## **14. Adversarial Attacks on Data Attribution**

cs.LG

Accepted at the 13th International Conference on Learning  Representations (ICLR 2025)

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2409.05657v4) [paper-pdf](http://arxiv.org/pdf/2409.05657v4)

**Authors**: Xinhe Wang, Pingbang Hu, Junwei Deng, Jiaqi W. Ma

**Abstract**: Data attribution aims to quantify the contribution of individual training data points to the outputs of an AI model, which has been used to measure the value of training data and compensate data providers. Given the impact on financial decisions and compensation mechanisms, a critical question arises concerning the adversarial robustness of data attribution methods. However, there has been little to no systematic research addressing this issue. In this work, we aim to bridge this gap by detailing a threat model with clear assumptions about the adversary's goal and capabilities and proposing principled adversarial attack methods on data attribution. We present two methods, Shadow Attack and Outlier Attack, which generate manipulated datasets to inflate the compensation adversarially. The Shadow Attack leverages knowledge about the data distribution in the AI applications, and derives adversarial perturbations through "shadow training", a technique commonly used in membership inference attacks. In contrast, the Outlier Attack does not assume any knowledge about the data distribution and relies solely on black-box queries to the target model's predictions. It exploits an inductive bias present in many data attribution methods - outlier data points are more likely to be influential - and employs adversarial examples to generate manipulated datasets. Empirically, in image classification and text generation tasks, the Shadow Attack can inflate the data-attribution-based compensation by at least 200%, while the Outlier Attack achieves compensation inflation ranging from 185% to as much as 643%. Our implementation is ready at https://github.com/TRAIS-Lab/adversarial-attack-data-attribution.



## **15. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

cs.IR

29 pages

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12574v1) [paper-pdf](http://arxiv.org/pdf/2505.12574v1)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.



## **16. A Survey of Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12567v1) [paper-pdf](http://arxiv.org/pdf/2505.12567v1)

**Authors**: Wenrui Xu, Keshab K. Parhi

**Abstract**: Large language models (LLMs) and LLM-based agents have been widely deployed in a wide range of applications in the real world, including healthcare diagnostics, financial analysis, customer support, robotics, and autonomous driving, expanding their powerful capability of understanding, reasoning, and generating natural languages. However, the wide deployment of LLM-based applications exposes critical security and reliability risks, such as the potential for malicious misuse, privacy leakage, and service disruption that weaken user trust and undermine societal safety. This paper provides a systematic overview of the details of adversarial attacks targeting both LLMs and LLM-based agents. These attacks are organized into three phases in LLMs: Training-Phase Attacks, Inference-Phase Attacks, and Availability & Integrity Attacks. For each phase, we analyze the details of representative and recently introduced attack methods along with their corresponding defenses. We hope our survey will provide a good tutorial and a comprehensive understanding of LLM security, especially for attacks on LLMs. We desire to raise attention to the risks inherent in widely deployed LLM-based applications and highlight the urgent need for robust mitigation strategies for evolving threats.



## **17. An In-kernel Forensics Engine for Investigating Evasive Attacks**

cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.06498v2) [paper-pdf](http://arxiv.org/pdf/2505.06498v2)

**Authors**: Javad Zandi, Lalchandra Rampersaud, Amin Kharraz

**Abstract**: Over the years, adversarial attempts against critical services have become more effective and sophisticated in launching low-profile attacks. This trend has always been concerning. However, an even more alarming trend is the increasing difficulty of collecting relevant evidence about these attacks and the involved threat actors in the early stages before significant damage is done. This issue puts defenders at a significant disadvantage, as it becomes exceedingly difficult to understand the attack details and formulate an appropriate response. Developing robust forensics tools to collect evidence about modern threats has never been easy. One main challenge is to provide a robust trade-off between achieving sufficient visibility while leaving minimal detectable artifacts. This paper will introduce LASE, an open-source Low-Artifact Forensics Engine to perform threat analysis and forensics in Windows operating system. LASE augments current analysis tools by providing detailed, system-wide monitoring capabilities while minimizing detectable artifacts. We designed multiple deployment scenarios, showing LASE's potential in evidence gathering and threat reasoning in a real-world setting. By making LASE and its execution trace data available to the broader research community, this work encourages further exploration in the field by reducing the engineering costs for threat analysis and building a longitudinal behavioral analysis catalog for diverse security domains.



## **18. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12442v1) [paper-pdf](http://arxiv.org/pdf/2505.12442v1)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.



## **19. CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement**

cs.CL

Accepted in ACL LLMSec Workshop 2025

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12368v1) [paper-pdf](http://arxiv.org/pdf/2505.12368v1)

**Authors**: Gauri Kholkar, Ratinder Ahuja

**Abstract**: Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations.



## **20. The Tower of Babel Revisited: Multilingual Jailbreak Prompts on Closed-Source Large Language Models**

cs.CL

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12287v1) [paper-pdf](http://arxiv.org/pdf/2505.12287v1)

**Authors**: Linghan Huang, Haolin Jin, Zhaoge Bi, Pengyue Yang, Peizhou Zhao, Taozhao Chen, Xiongfei Wu, Lei Ma, Huaming Chen

**Abstract**: Large language models (LLMs) have seen widespread applications across various domains, yet remain vulnerable to adversarial prompt injections. While most existing research on jailbreak attacks and hallucination phenomena has focused primarily on open-source models, we investigate the frontier of closed-source LLMs under multilingual attack scenarios. We present a first-of-its-kind integrated adversarial framework that leverages diverse attack techniques to systematically evaluate frontier proprietary solutions, including GPT-4o, DeepSeek-R1, Gemini-1.5-Pro, and Qwen-Max. Our evaluation spans six categories of security contents in both English and Chinese, generating 38,400 responses across 32 types of jailbreak attacks. Attack success rate (ASR) is utilized as the quantitative metric to assess performance from three dimensions: prompt design, model architecture, and language environment. Our findings suggest that Qwen-Max is the most vulnerable, while GPT-4o shows the strongest defense. Notably, prompts in Chinese consistently yield higher ASRs than their English counterparts, and our novel Two-Sides attack technique proves to be the most effective across all models. This work highlights a dire need for language-aware alignment and robust cross-lingual defenses in LLMs, and we hope it will inspire researchers, developers, and policymakers toward more robust and inclusive AI systems.



## **21. `Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs**

cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2502.00735v3) [paper-pdf](http://arxiv.org/pdf/2502.00735v3)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen, Kim-Kwang Raymond Choo

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.



## **22. Self-Destructive Language Model**

cs.LG

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12186v1) [paper-pdf](http://arxiv.org/pdf/2505.12186v1)

**Authors**: Yuhui Wang, Rongyi Zhu, Ting Wang

**Abstract**: Harmful fine-tuning attacks pose a major threat to the security of large language models (LLMs), allowing adversaries to compromise safety guardrails with minimal harmful data. While existing defenses attempt to reinforce LLM alignment, they fail to address models' inherent "trainability" on harmful data, leaving them vulnerable to stronger attacks with increased learning rates or larger harmful datasets. To overcome this critical limitation, we introduce SEAM, a novel alignment-enhancing defense that transforms LLMs into self-destructive models with intrinsic resilience to misalignment attempts. Specifically, these models retain their capabilities for legitimate tasks while exhibiting substantial performance degradation when fine-tuned on harmful data. The protection is achieved through a novel loss function that couples the optimization trajectories of benign and harmful data, enhanced with adversarial gradient ascent to amplify the self-destructive effect. To enable practical training, we develop an efficient Hessian-free gradient estimate with theoretical error bounds. Extensive evaluation across LLMs and datasets demonstrates that SEAM creates a no-win situation for adversaries: the self-destructive models achieve state-of-the-art robustness against low-intensity attacks and undergo catastrophic performance collapse under high-intensity attacks, rendering them effectively unusable. (warning: this paper contains potentially harmful content generated by LLMs.)



## **23. EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective**

cs.SE

19 pages, 11 figures

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12185v1) [paper-pdf](http://arxiv.org/pdf/2505.12185v1)

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.



## **24. ImF: Implicit Fingerprint for Large Language Models**

cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2503.21805v2) [paper-pdf](http://arxiv.org/pdf/2503.21805v2)

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing fingerprinting techniques typically embed identifiable patterns with weak semantic coherence, resulting in fingerprints that significantly differ from the natural question-answering (QA) behavior inherent to LLMs. This discrepancy undermines the stealthiness of the embedded fingerprints and makes them vulnerable to adversarial attacks. In this paper, we first demonstrate the critical vulnerability of existing fingerprint embedding methods by introducing a novel adversarial attack named Generation Revision Intervention (GRI) attack. GRI attack exploits the semantic fragility of current fingerprinting methods, effectively erasing fingerprints by disrupting their weakly correlated semantic structures. Our empirical evaluation highlights that traditional fingerprinting approaches are significantly compromised by the GRI attack, revealing severe limitations in their robustness under realistic adversarial conditions. To advance the state-of-the-art in model fingerprinting, we propose a novel model fingerprint paradigm called Implicit Fingerprints (ImF). ImF leverages steganography techniques to subtly embed ownership information within natural texts, subsequently using Chain-of-Thought (CoT) prompting to construct semantically coherent and contextually natural QA pairs. This design ensures that fingerprints seamlessly integrate with the standard model behavior, remaining indistinguishable from regular outputs and substantially reducing the risk of accidental triggering and targeted removal. We conduct a comprehensive evaluation of ImF on 15 diverse LLMs, spanning different architectures and varying scales.



## **25. FABLE: A Localized, Targeted Adversarial Attack on Weather Forecasting Models**

cs.LG

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.12167v1) [paper-pdf](http://arxiv.org/pdf/2505.12167v1)

**Authors**: Yue Deng, Asadullah Hill Galib, Xin Lan, Pang-Ning Tan, Lifeng Luo

**Abstract**: Deep learning-based weather forecasting models have recently demonstrated significant performance improvements over gold-standard physics-based simulation tools. However, these models are vulnerable to adversarial attacks, which raises concerns about their trustworthiness. In this paper, we first investigate the feasibility of applying existing adversarial attack methods to weather forecasting models. We argue that a successful attack should (1) not modify significantly its original inputs, (2) be faithful, i.e., achieve the desired forecast at targeted locations with minimal changes to non-targeted locations, and (3) be geospatio-temporally realistic. However, balancing these criteria is a challenge as existing methods are not designed to preserve the geospatio-temporal dependencies of the original samples. To address this challenge, we propose a novel framework called FABLE (Forecast Alteration By Localized targeted advErsarial attack), which employs a 3D discrete wavelet decomposition to extract the varying components of the geospatio-temporal data. By regulating the magnitude of adversarial perturbations across different components, FABLE can generate adversarial inputs that maintain geospatio-temporal coherence while remaining faithful and closely aligned with the original inputs. Experimental results on multiple real-world datasets demonstrate the effectiveness of our framework over baseline methods across various metrics.



## **26. Black-box Adversaries from Latent Space: Unnoticeable Attacks on Human Pose and Shape Estimation**

cs.CV

17 pages, 6 figures

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.12009v1) [paper-pdf](http://arxiv.org/pdf/2505.12009v1)

**Authors**: Zhiying Li, Guanggang Geng, Yeying Jin, Zhizhi Guo, Bruce Gu, Jidong Huo, Zhaoxin Fan, Wenjun Wu

**Abstract**: Expressive human pose and shape (EHPS) estimation is vital for digital human generation, particularly in live-streaming applications. However, most existing EHPS models focus primarily on minimizing estimation errors, with limited attention on potential security vulnerabilities. Current adversarial attacks on EHPS models often require white-box access (e.g., model details or gradients) or generate visually conspicuous perturbations, limiting their practicality and ability to expose real-world security threats. To address these limitations, we propose a novel Unnoticeable Black-Box Attack (UBA) against EHPS models. UBA leverages the latent-space representations of natural images to generate an optimal adversarial noise pattern and iteratively refine its attack potency along an optimized direction in digital space. Crucially, this process relies solely on querying the model's output, requiring no internal knowledge of the EHPS architecture, while guiding the noise optimization toward greater stealth and effectiveness. Extensive experiments and visual analyses demonstrate the superiority of UBA. Notably, UBA increases the pose estimation errors of EHPS models by 17.27%-58.21% on average, revealing critical vulnerabilities. These findings underscore the urgent need to address and mitigate security risks associated with digital human generation systems.



## **27. Adversarial Attacks on Both Face Recognition and Face Anti-spoofing Models**

cs.CV

Proceedings of the 34th International Joint Conference on Artificial  Intelligence (IJCAI 2025)

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2405.16940v2) [paper-pdf](http://arxiv.org/pdf/2405.16940v2)

**Authors**: Fengfan Zhou, Qianyu Zhou, Hefei Ling, Xuequan Lu

**Abstract**: Adversarial attacks on Face Recognition (FR) systems have demonstrated significant effectiveness against standalone FR models. However, their practicality diminishes in complete FR systems that incorporate Face Anti-Spoofing (FAS) models, as these models can detect and mitigate a substantial number of adversarial examples. To address this critical yet under-explored challenge, we introduce a novel attack setting that targets both FR and FAS models simultaneously, thereby enhancing the practicability of adversarial attacks on integrated FR systems. Specifically, we propose a new attack method, termed Reference-free Multi-level Alignment (RMA), designed to improve the capacity of black-box attacks on both FR and FAS models. The RMA framework is built upon three key components. Firstly, we propose an Adaptive Gradient Maintenance module to address the imbalances in gradient contributions between FR and FAS models. Secondly, we develop a Reference-free Intermediate Biasing module to improve the transferability of adversarial examples against FAS models. In addition, we introduce a Multi-level Feature Alignment module to reduce feature discrepancies at various levels of representation. Extensive experiments showcase the superiority of our proposed attack method to state-of-the-art adversarial attacks.



## **28. Understanding and Enhancing the Transferability of Jailbreaking Attacks**

cs.LG

Accepted by ICLR 2025

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2502.03052v2) [paper-pdf](http://arxiv.org/pdf/2502.03052v2)

**Authors**: Runqi Lin, Bo Han, Fengwang Li, Tongling Liu

**Abstract**: Jailbreaking attacks can effectively manipulate open-source large language models (LLMs) to produce harmful responses. However, these attacks exhibit limited transferability, failing to disrupt proprietary LLMs consistently. To reliably identify vulnerabilities in proprietary LLMs, this work investigates the transferability of jailbreaking attacks by analysing their impact on the model's intent perception. By incorporating adversarial sequences, these attacks can redirect the source LLM's focus away from malicious-intent tokens in the original input, thereby obstructing the model's intent recognition and eliciting harmful responses. Nevertheless, these adversarial sequences fail to mislead the target LLM's intent perception, allowing the target LLM to refocus on malicious-intent tokens and abstain from responding. Our analysis further reveals the inherent distributional dependency within the generated adversarial sequences, whose effectiveness stems from overfitting the source LLM's parameters, resulting in limited transferability to target LLMs. To this end, we propose the Perceived-importance Flatten (PiF) method, which uniformly disperses the model's focus across neutral-intent tokens in the original input, thus obscuring malicious-intent tokens without relying on overfitted adversarial sequences. Extensive experiments demonstrate that PiF provides an effective and efficient red-teaming evaluation for proprietary LLMs.



## **29. Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation**

cs.LG

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2406.03862v2) [paper-pdf](http://arxiv.org/pdf/2406.03862v2)

**Authors**: Shojiro Yamabe, Kazuto Fukuchi, Jun Sakuma

**Abstract**: This study investigates behavior-targeted attacks on reinforcement learning and their countermeasures. Behavior-targeted attacks aim to manipulate the victim's behavior as desired by the adversary through adversarial interventions in state observations. Existing behavior-targeted attacks have some limitations, such as requiring white-box access to the victim's policy. To address this, we propose a novel attack method using imitation learning from adversarial demonstrations, which works under limited access to the victim's policy and is environment-agnostic. In addition, our theoretical analysis proves that the policy's sensitivity to state changes impacts defense performance, particularly in the early stages of the trajectory. Based on this insight, we propose time-discounted regularization, which enhances robustness against attacks while maintaining task performance. To the best of our knowledge, this is the first defense strategy specifically designed for behavior-targeted attacks.



## **30. Adversarial Robustness for Unified Multi-Modal Encoders via Efficient Calibration**

cs.CV

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.11895v1) [paper-pdf](http://arxiv.org/pdf/2505.11895v1)

**Authors**: Chih-Ting Liao, Bin Ren, Guofeng Mei, Xu Zheng

**Abstract**: Recent unified multi-modal encoders align a wide range of modalities into a shared representation space, enabling diverse cross-modal tasks. Despite their impressive capabilities, the robustness of these models under adversarial perturbations remains underexplored, which is a critical concern for safety-sensitive applications. In this work, we present the first comprehensive study of adversarial vulnerability in unified multi-modal encoders. We find that even mild adversarial perturbations lead to substantial performance drops across all modalities. Non-visual inputs, such as audio and point clouds, are especially fragile, while visual inputs like images and videos also degrade significantly. To address this, we propose an efficient adversarial calibration framework that improves robustness across modalities without modifying pretrained encoders or semantic centers, ensuring compatibility with existing foundation models. Our method introduces modality-specific projection heads trained solely on adversarial examples, while keeping the backbone and embeddings frozen. We explore three training objectives: fixed-center cross-entropy, clean-to-adversarial L2 alignment, and clean-adversarial InfoNCE, and we introduce a regularization strategy to ensure modality-consistent alignment under attack. Experiments on six modalities and three Bind-style models show that our method improves adversarial robustness by up to 47.3 percent at epsilon = 4/255, while preserving or even improving clean zero-shot and retrieval performance with less than 1 percent trainable parameters.



## **31. SynFuzz: Leveraging Fuzzing of Netlist to Detect Synthesis Bugs**

cs.CR

15 pages, 10 figures, 5 tables

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2504.18812v2) [paper-pdf](http://arxiv.org/pdf/2504.18812v2)

**Authors**: Raghul Saravanan, Sudipta Paria, Aritra Dasgupta, Venkat Nitin Patnala, Swarup Bhunia, Sai Manoj P D

**Abstract**: In the evolving landscape of integrated circuit (IC) design, the increasing complexity of modern processors and intellectual property (IP) cores has introduced new challenges in ensuring design correctness and security. The recent advancements in hardware fuzzing techniques have shown their efficacy in detecting hardware bugs and vulnerabilities at the RTL abstraction level of hardware. However, they suffer from several limitations, including an inability to address vulnerabilities introduced during synthesis and gate-level transformations. These methods often fail to detect issues arising from library adversaries, where compromised or malicious library components can introduce backdoors or unintended behaviors into the design. In this paper, we present a novel hardware fuzzer, SynFuzz, designed to overcome the limitations of existing hardware fuzzing frameworks. SynFuzz focuses on fuzzing hardware at the gate-level netlist to identify synthesis bugs and vulnerabilities that arise during the transition from RTL to the gate-level. We analyze the intrinsic hardware behaviors using coverage metrics specifically tailored for the gate-level. Furthermore, SynFuzz implements differential fuzzing to uncover bugs associated with EDA libraries. We evaluated SynFuzz on popular open-source processors and IP designs, successfully identifying 7 new synthesis bugs. Additionally, by exploiting the optimization settings of EDA tools, we performed a compromised library mapping attack (CLiMA), creating a malicious version of hardware designs that remains undetectable by traditional verification methods. We also demonstrate how SynFuzz overcomes the limitations of the industry-standard formal verification tool, Cadence Conformal, providing a more robust and comprehensive approach to hardware verification.



## **32. Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey**

cs.CV

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2410.23687v2) [paper-pdf](http://arxiv.org/pdf/2410.23687v2)

**Authors**: Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Zhe Liu

**Abstract**: With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreaking, have emerged. Understanding these attacks promotes system robustness improvement and neural networks demystification. However, existing surveys often target attack taxonomy and lack in-depth analysis like 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations framework; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.



## **33. Kick Bad Guys Out! Conditionally Activated Anomaly Detection in Federated Learning with Zero-Knowledge Proof Verification**

cs.CR

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2310.04055v5) [paper-pdf](http://arxiv.org/pdf/2310.04055v5)

**Authors**: Shanshan Han, Wenxuan Wu, Baturalp Buyukates, Weizhao Jin, Qifan Zhang, Yuhang Yao, Salman Avestimehr, Chaoyang He

**Abstract**: Federated Learning (FL) systems are vulnerable to adversarial attacks, such as model poisoning and backdoor attacks. However, existing defense mechanisms often fall short in real-world settings due to key limitations: they may rely on impractical assumptions, introduce distortions by modifying aggregation functions, or degrade model performance even in benign scenarios. To address these issues, we propose a novel anomaly detection method designed specifically for practical FL scenarios. Our approach employs a two-stage, conditionally activated detection mechanism: cross-round check first detects whether suspicious activity has occurred, and, if warranted, a cross-client check filters out malicious participants. This mechanism preserves utility while avoiding unrealistic assumptions. Moreover, to ensure the transparency and integrity of the defense mechanism, we incorporate zero-knowledge proofs, enabling clients to verify the detection without relying solely on the server's goodwill. To the best of our knowledge, this is the first method to bridge the gap between theoretical advances in FL security and the demands of real-world deployment. Extensive experiments across diverse tasks and real-world edge devices demonstrate the effectiveness of our method over state-of-the-art defenses.



## **34. DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising**

cs.CL

Accepted to ACL 2025

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2407.00248v2) [paper-pdf](http://arxiv.org/pdf/2407.00248v2)

**Authors**: Zhenhao Li, Huichi Zhou, Marek Rei, Lucia Specia

**Abstract**: Pretrained language models have significantly advanced performance across various natural language processing tasks. However, adversarial attacks continue to pose a critical challenge to systems built using these models, as they can be exploited with carefully crafted adversarial texts. Inspired by the ability of diffusion models to predict and reduce noise in computer vision, we propose a novel and flexible adversarial defense method for language classification tasks, DiffuseDef, which incorporates a diffusion layer as a denoiser between the encoder and the classifier. The diffusion layer is trained on top of the existing classifier, ensuring seamless integration with any model in a plug-and-play manner. During inference, the adversarial hidden state is first combined with sampled noise, then denoised iteratively and finally ensembled to produce a robust text representation. By integrating adversarial training, denoising, and ensembling techniques, we show that DiffuseDef improves over existing adversarial defense methods and achieves state-of-the-art performance against common black-box and white-box adversarial attacks.



## **35. Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework**

cs.CV

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2306.07992v2) [paper-pdf](http://arxiv.org/pdf/2306.07992v2)

**Authors**: Minglei Yin, Bin Liu, Neil Zhenqiang Gong, Xin Li

**Abstract**: With rich visual data, such as images, becoming readily associated with items, visually-aware recommendation systems (VARS) have been widely used in different applications. Recent studies have shown that VARS are vulnerable to item-image adversarial attacks, which add human-imperceptible perturbations to the clean images associated with those items. Attacks on VARS pose new security challenges to a wide range of applications such as e-Commerce and social networks where VARS are widely used. How to secure VARS from such adversarial attacks becomes a critical problem. Currently, there is still a lack of systematic study on how to design secure defense strategies against visual attacks on VARS. In this paper, we attempt to fill this gap by proposing an adversarial image reconstruction and detection framework to secure VARS. Our proposed method can simultaneously (1) secure VARS from adversarial attacks characterized by local perturbations by image reconstruction based on global vision transformers; and (2) accurately detect adversarial examples using a novel contrastive learning approach. Meanwhile, our framework is designed to be used as both a filter and a detector so that they can be jointly trained to improve the flexibility of our defense strategy to a variety of attacks and VARS models. We have conducted extensive experimental studies with two popular attack methods (FGSM and PGD). Our experimental results on two real-world datasets show that our defense strategy against visual attacks is effective and outperforms existing methods on different attacks. Moreover, our method can detect adversarial examples with high accuracy.



## **36. Co-Evolutionary Defence of Active Directory Attack Graphs via GNN-Approximated Dynamic Programming**

cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11710v1) [paper-pdf](http://arxiv.org/pdf/2505.11710v1)

**Authors**: Diksha Goel, Hussain Ahmad, Kristen Moore, Mingyu Guo

**Abstract**: Modern enterprise networks increasingly rely on Active Directory (AD) for identity and access management. However, this centralization exposes a single point of failure, allowing adversaries to compromise high-value assets. Existing AD defense approaches often assume static attacker behavior, but real-world adversaries adapt dynamically, rendering such methods brittle. To address this, we model attacker-defender interactions in AD as a Stackelberg game between an adaptive attacker and a proactive defender. We propose a co-evolutionary defense framework that combines Graph Neural Network Approximated Dynamic Programming (GNNDP) to model attacker strategies, with Evolutionary Diversity Optimization (EDO) to generate resilient blocking strategies. To ensure scalability, we introduce a Fixed-Parameter Tractable (FPT) graph reduction method that reduces complexity while preserving strategic structure. Our framework jointly refines attacker and defender policies to improve generalization and prevent premature convergence. Experiments on synthetic AD graphs show near-optimal results (within 0.1 percent of optimality on r500) and improved performance on larger graphs (r1000 and r2000), demonstrating the framework's scalability and effectiveness.



## **37. Unveiling the Black Box: A Multi-Layer Framework for Explaining Reinforcement Learning-Based Cyber Agents**

cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11708v1) [paper-pdf](http://arxiv.org/pdf/2505.11708v1)

**Authors**: Diksha Goel, Kristen Moore, Jeff Wang, Minjune Kim, Thanh Thi Nguyen

**Abstract**: Reinforcement Learning (RL) agents are increasingly used to simulate sophisticated cyberattacks, but their decision-making processes remain opaque, hindering trust, debugging, and defensive preparedness. In high-stakes cybersecurity contexts, explainability is essential for understanding how adversarial strategies are formed and evolve over time. In this paper, we propose a unified, multi-layer explainability framework for RL-based attacker agents that reveals both strategic (MDP-level) and tactical (policy-level) reasoning. At the MDP level, we model cyberattacks as a Partially Observable Markov Decision Processes (POMDPs) to expose exploration-exploitation dynamics and phase-aware behavioural shifts. At the policy level, we analyse the temporal evolution of Q-values and use Prioritised Experience Replay (PER) to surface critical learning transitions and evolving action preferences. Evaluated across CyberBattleSim environments of increasing complexity, our framework offers interpretable insights into agent behaviour at scale. Unlike previous explainable RL methods, which are often post-hoc, domain-specific, or limited in depth, our approach is both agent- and environment-agnostic, supporting use cases ranging from red-team simulation to RL policy debugging. By transforming black-box learning into actionable behavioural intelligence, our framework enables both defenders and developers to better anticipate, analyse, and respond to autonomous cyber threats.



## **38. On the Sharp Input-Output Analysis of Nonlinear Systems under Adversarial Attacks**

math.OC

28 pages, 2 figures

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11688v1) [paper-pdf](http://arxiv.org/pdf/2505.11688v1)

**Authors**: Jihun Kim, Yuchen Fang, Javad Lavaei

**Abstract**: This paper is concerned with learning the input-output mapping of general nonlinear dynamical systems. While the existing literature focuses on Gaussian inputs and benign disturbances, we significantly broaden the scope of admissible control inputs and allow correlated, nonzero-mean, adversarial disturbances. With our reformulation as a linear combination of basis functions, we prove that the $l_1$-norm estimator overcomes the challenges as long as the probability that the system is under adversarial attack at a given time is smaller than a certain threshold. We provide an estimation error bound that decays with the input memory length and prove its optimality by constructing a problem instance that suffers from the same bound under adversarial attacks. Our work provides a sharp input-output analysis for a generic nonlinear and partially observed system under significantly generalized assumptions compared to existing works.



## **39. Benchmarking Unsupervised Online IDS for Masquerade Attacks in CAN**

cs.CR

17 pages, 10 figures, 4 tables

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2406.13778v2) [paper-pdf](http://arxiv.org/pdf/2406.13778v2)

**Authors**: Pablo Moriano, Steven C. Hespeler, Mingyan Li, Robert A. Bridges

**Abstract**: Vehicular controller area networks (CANs) are susceptible to masquerade attacks by malicious adversaries. In masquerade attacks, adversaries silence a targeted ID and then send malicious frames with forged content at the expected timing of benign frames. As masquerade attacks could seriously harm vehicle functionality and are the stealthiest attacks to detect in CAN, recent work has devoted attention to compare frameworks for detecting masquerade attacks in CAN. However, most existing works report offline evaluations using CAN logs already collected using simulations that do not comply with the domain's real-time constraints. Here we contribute to advance the state of the art by introducing a benchmark study of four different non-deep learning (DL)-based unsupervised online intrusion detection systems (IDS) for masquerade attacks in CAN. Our approach differs from existing benchmarks in that we analyze the effect of controlling streaming data conditions in a sliding window setting. In doing so, we use realistic masquerade attacks being replayed from the ROAD dataset. We show that although benchmarked IDS are not effective at detecting every attack type, the method that relies on detecting changes in the hierarchical structure of clusters of time series produces the best results at the expense of higher computational overhead. We discuss limitations, open challenges, and how the benchmarked methods can be used for practical unsupervised online CAN IDS for masquerade attacks.



## **40. To Think or Not to Think: Exploring the Unthinking Vulnerability in Large Reasoning Models**

cs.CL

39 pages, 13 tables, 14 figures

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2502.12202v2) [paper-pdf](http://arxiv.org/pdf/2502.12202v2)

**Authors**: Zihao Zhu, Hongbao Zhang, Ruotong Wang, Ke Xu, Siwei Lyu, Baoyuan Wu

**Abstract**: Large Reasoning Models (LRMs) are designed to solve complex tasks by generating explicit reasoning traces before producing final answers. However, we reveal a critical vulnerability in LRMs -- termed Unthinking Vulnerability -- wherein the thinking process can be bypassed by manipulating special delimiter tokens. It is empirically demonstrated to be widespread across mainstream LRMs, posing both a significant risk and potential utility, depending on how it is exploited. In this paper, we systematically investigate this vulnerability from both malicious and beneficial perspectives. On the malicious side, we introduce Breaking of Thought (BoT), a novel attack that enables adversaries to bypass the thinking process of LRMs, thereby compromising their reliability and availability. We present two variants of BoT: a training-based version that injects backdoor during the fine-tuning stage, and a training-free version based on adversarial attack during the inference stage. As a potential defense, we propose thinking recovery alignment to partially mitigate the vulnerability. On the beneficial side, we introduce Monitoring of Thought (MoT), a plug-and-play framework that allows model owners to enhance efficiency and safety. It is implemented by leveraging the same vulnerability to dynamically terminate redundant or risky reasoning through external monitoring. Extensive experiments show that BoT poses a significant threat to reasoning reliability, while MoT provides a practical solution for preventing overthinking and jailbreaking. Our findings expose an inherent flaw in current LRM architectures and underscore the need for more robust reasoning systems in the future.



## **41. LLMs unlock new paths to monetizing exploits**

cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11449v1) [paper-pdf](http://arxiv.org/pdf/2505.11449v1)

**Authors**: Nicholas Carlini, Milad Nasr, Edoardo Debenedetti, Barry Wang, Christopher A. Choquette-Choo, Daphne Ippolito, Florian Tramèr, Matthew Jagielski

**Abstract**: We argue that Large language models (LLMs) will soon alter the economics of cyberattacks. Instead of attacking the most commonly used software and monetizing exploits by targeting the lowest common denominator among victims, LLMs enable adversaries to launch tailored attacks on a user-by-user basis. On the exploitation front, instead of human attackers manually searching for one difficult-to-identify bug in a product with millions of users, LLMs can find thousands of easy-to-identify bugs in products with thousands of users. And on the monetization front, instead of generic ransomware that always performs the same attack (encrypt all your data and request payment to decrypt), an LLM-driven ransomware attack could tailor the ransom demand based on the particular content of each exploited device.   We show that these two attacks (and several others) are imminently practical using state-of-the-art LLMs. For example, we show that without any human intervention, an LLM finds highly sensitive personal information in the Enron email dataset (e.g., an executive having an affair with another employee) that could be used for blackmail. While some of our attacks are still too expensive to scale widely today, the incentives to implement these attacks will only increase as LLMs get cheaper. Thus, we argue that LLMs create a need for new defense-in-depth approaches.



## **42. CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs**

cs.CL

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11413v1) [paper-pdf](http://arxiv.org/pdf/2505.11413v1)

**Authors**: Sijia Chen, Xiaomin Li, Mengxue Zhang, Eric Hanchen Jiang, Qingcheng Zeng, Chen-Hsiang Yu

**Abstract**: Large language models (LLMs) are increasingly deployed in medical contexts, raising critical concerns about safety, alignment, and susceptibility to adversarial manipulation. While prior benchmarks assess model refusal capabilities for harmful prompts, they often lack clinical specificity, graded harmfulness levels, and coverage of jailbreak-style attacks. We introduce CARES (Clinical Adversarial Robustness and Evaluation of Safety), a benchmark for evaluating LLM safety in healthcare. CARES includes over 18,000 prompts spanning eight medical safety principles, four harm levels, and four prompting styles: direct, indirect, obfuscated, and role-play, to simulate both malicious and benign use cases. We propose a three-way response evaluation protocol (Accept, Caution, Refuse) and a fine-grained Safety Score metric to assess model behavior. Our analysis reveals that many state-of-the-art LLMs remain vulnerable to jailbreaks that subtly rephrase harmful prompts, while also over-refusing safe but atypically phrased queries. Finally, we propose a mitigation strategy using a lightweight classifier to detect jailbreak attempts and steer models toward safer behavior via reminder-based conditioning. CARES provides a rigorous framework for testing and improving medical LLM safety under adversarial and ambiguous conditions.



## **43. Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities**

stat.ML

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2501.02406v4) [paper-pdf](http://arxiv.org/pdf/2501.02406v4)

**Authors**: Tara Radvand, Mojtaba Abdolmaleki, Mohamed Mostagir, Ambuj Tewari

**Abstract**: Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. In this paper, we answer the following question: Given a piece of text, can we identify whether it was produced by a particular LLM or not? We model LLM-generated text as a sequential stochastic process with complete dependence on history. We then design zero-shot statistical tests to (i) distinguish between text generated by two different known sets of LLMs $A$ (non-sanctioned) and $B$ (in-house), and (ii) identify whether text was generated by a known LLM or generated by any unknown model, e.g., a human or some other language generation process. We prove that the type I and type II errors of our test decrease exponentially with the length of the text. For that, we show that if $B$ generates the text, then except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. We then present experiments using LLMs with white-box access to support our theoretical results and empirically examine the robustness of our results to black-box settings and adversarial attacks. In the black-box setting, our method achieves an average TPR of 82.5\% at a fixed FPR of 5\%. Under adversarial perturbations, our minimum TPR is 48.6\% at the same FPR threshold. Both results outperform all non-commercial baselines. See https://github.com/TaraRadvand74/llm-text-detection for code, data, and an online demo of the project.



## **44. GenoArmory: A Unified Evaluation Framework for Adversarial Attacks on Genomic Foundation Models**

cs.LG

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.10983v1) [paper-pdf](http://arxiv.org/pdf/2505.10983v1)

**Authors**: Haozheng Luo, Chenghao Qiu, Yimin Wang, Shang Wu, Jiahao Yu, Han Liu, Binghui Wang, Yan Chen

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GenoArmory. Unlike existing GFM benchmarks, GenoArmory offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Additionally, we introduce GenoAdv, a new adversarial sample dataset designed to improve GFM safety. Empirically, classification models exhibit greater robustness to adversarial perturbations compared to generative models, highlighting the impact of task type on model vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features.



## **45. On the Security Risks of ML-based Malware Detection Systems: A Survey**

cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.10903v1) [paper-pdf](http://arxiv.org/pdf/2505.10903v1)

**Authors**: Ping He, Yuhao Mao, Changjiang Li, Lorenzo Cavallaro, Ting Wang, Shouling Ji

**Abstract**: Malware presents a persistent threat to user privacy and data integrity. To combat this, machine learning-based (ML-based) malware detection (MD) systems have been developed. However, these systems have increasingly been attacked in recent years, undermining their effectiveness in practice. While the security risks associated with ML-based MD systems have garnered considerable attention, the majority of prior works is limited to adversarial malware examples, lacking a comprehensive analysis of practical security risks. This paper addresses this gap by utilizing the CIA principles to define the scope of security risks. We then deconstruct ML-based MD systems into distinct operational stages, thus developing a stage-based taxonomy. Utilizing this taxonomy, we summarize the technical progress and discuss the gaps in the attack and defense proposals related to the ML-based MD systems within each stage. Subsequently, we conduct two case studies, using both inter-stage and intra-stage analyses according to the stage-based taxonomy to provide new empirical insights. Based on these analyses and insights, we suggest potential future directions from both inter-stage and intra-stage perspectives.



## **46. LARGO: Latent Adversarial Reflection through Gradient Optimization for Jailbreaking LLMs**

cs.LG

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.10838v1) [paper-pdf](http://arxiv.org/pdf/2505.10838v1)

**Authors**: Ran Li, Hao Wang, Chengzhi Mao

**Abstract**: Efficient red-teaming method to uncover vulnerabilities in Large Language Models (LLMs) is crucial. While recent attacks often use LLMs as optimizers, the discrete language space make gradient-based methods struggle. We introduce LARGO (Latent Adversarial Reflection through Gradient Optimization), a novel latent self-reflection attack that reasserts the power of gradient-based optimization for generating fluent jailbreaking prompts. By operating within the LLM's continuous latent space, LARGO first optimizes an adversarial latent vector and then recursively call the same LLM to decode the latent into natural language. This methodology yields a fast, effective, and transferable attack that produces fluent and stealthy prompts. On standard benchmarks like AdvBench and JailbreakBench, LARGO surpasses leading jailbreaking techniques, including AutoDAN, by 44 points in attack success rate. Our findings demonstrate a potent alternative to agentic LLM prompting, highlighting the efficacy of interpreting and attacking LLM internals through gradient optimization.



## **47. Mitigating Many-Shot Jailbreaking**

cs.LG

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2504.09604v3) [paper-pdf](http://arxiv.org/pdf/2504.09604v3)

**Authors**: Christopher M. Ackerman, Nina Panickssery

**Abstract**: Many-shot jailbreaking (MSJ) is an adversarial technique that exploits the long context windows of modern LLMs to circumvent model safety training by including in the prompt many examples of a "fake" assistant responding inappropriately before the final request. With enough examples, the model's in-context learning abilities override its safety training, and it responds as if it were the "fake" assistant. In this work, we probe the effectiveness of different fine-tuning and input sanitization approaches on mitigating MSJ attacks, alone and in combination. We find incremental mitigation effectiveness for each, and show that the combined techniques significantly reduce the effectiveness of MSJ attacks, while retaining model performance in benign in-context learning and conversational tasks. We suggest that our approach could meaningfully ameliorate this vulnerability if incorporated into model safety post-training.



## **48. SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models**

cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.07584v2) [paper-pdf](http://arxiv.org/pdf/2505.07584v2)

**Authors**: Huining Cui, Wei Liu

**Abstract**: The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security.



## **49. Model-Targeted Data Poisoning Attacks against ITS Applications with Provable Convergence**

math.OC

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.03966v2) [paper-pdf](http://arxiv.org/pdf/2505.03966v2)

**Authors**: Xin Wang, Feilong Wang, Yuan Hong, R. Tyrrell Rockafellar, Xuegang, Ban

**Abstract**: The growing reliance of intelligent systems on data makes the systems vulnerable to data poisoning attacks. Such attacks could compromise machine learning or deep learning models by disrupting the input data. Previous studies on data poisoning attacks are subject to specific assumptions, and limited attention is given to learning models with general (equality and inequality) constraints or lacking differentiability. Such learning models are common in practice, especially in Intelligent Transportation Systems (ITS) that involve physical or domain knowledge as specific model constraints. Motivated by ITS applications, this paper formulates a model-target data poisoning attack as a bi-level optimization problem with a constrained lower-level problem, aiming to induce the model solution toward a target solution specified by the adversary by modifying the training data incrementally. As the gradient-based methods fail to solve this optimization problem, we propose to study the Lipschitz continuity property of the model solution, enabling us to calculate the semi-derivative, a one-sided directional derivative, of the solution over data. We leverage semi-derivative descent to solve the bi-level optimization problem, and establish the convergence conditions of the method to any attainable target model. The model and solution method are illustrated with a simulation of a poisoning attack on the lane change detection using SVM.



## **50. Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines**

cs.CL

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2501.00745v2) [paper-pdf](http://arxiv.org/pdf/2501.00745v2)

**Authors**: Xiyang Hu

**Abstract**: The increasing integration of Large Language Model (LLM) based search engines has transformed the landscape of information retrieval. However, these systems are vulnerable to adversarial attacks, especially ranking manipulation attacks, where attackers craft webpage content to manipulate the LLM's ranking and promote specific content, gaining an unfair advantage over competitors. In this paper, we study the dynamics of ranking manipulation attacks. We frame this problem as an Infinitely Repeated Prisoners' Dilemma, where multiple players strategically decide whether to cooperate or attack. We analyze the conditions under which cooperation can be sustained, identifying key factors such as attack costs, discount rates, attack success rates, and trigger strategies that influence player behavior. We identify tipping points in the system dynamics, demonstrating that cooperation is more likely to be sustained when players are forward-looking. However, from a defense perspective, we find that simply reducing attack success probabilities can, paradoxically, incentivize attacks under certain conditions. Furthermore, defensive measures to cap the upper bound of attack success rates may prove futile in some scenarios. These insights highlight the complexity of securing LLM-based systems. Our work provides a theoretical foundation and practical insights for understanding and mitigating their vulnerabilities, while emphasizing the importance of adaptive security strategies and thoughtful ecosystem design.



