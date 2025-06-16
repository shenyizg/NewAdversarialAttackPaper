# Latest Large Language Model Attack Papers
**update at 2025-06-16 10:29:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Improving Large Language Model Safety with Contrastive Representation Learning**

cs.CL

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11938v1) [paper-pdf](http://arxiv.org/pdf/2506.11938v1)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense



## **2. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2501.18638v2) [paper-pdf](http://arxiv.org/pdf/2501.18638v2)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.



## **3. Black-Box Adversarial Attacks on LLM-Based Code Completion**

cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2408.02509v2) [paper-pdf](http://arxiv.org/pdf/2408.02509v2)

**Authors**: Slobodan Jenko, Niels Mündler, Jingxuan He, Mark Vero, Martin Vechev

**Abstract**: Modern code completion engines, powered by large language models (LLMs), assist millions of developers with their strong capabilities to generate functionally correct code. Due to this popularity, it is crucial to investigate the security implications of relying on LLM-based code completion. In this work, we demonstrate that state-of-the-art black-box LLM-based code completion engines can be stealthily biased by adversaries to significantly increase their rate of insecure code generation. We present the first attack, named INSEC, that achieves this goal. INSEC works by injecting an attack string as a short comment in the completion input. The attack string is crafted through a query-based optimization procedure starting from a set of carefully designed initialization schemes. We demonstrate INSEC's broad applicability and effectiveness by evaluating it on various state-of-the-art open-source models and black-box commercial services (e.g., OpenAI API and GitHub Copilot). On a diverse set of security-critical test cases, covering 16 CWEs across 5 programming languages, INSEC increases the rate of generated insecure code by more than 50%, while maintaining the functional correctness of generated code. We consider INSEC practical -- it requires low resources and costs less than 10 US dollars to develop on commodity hardware. Moreover, we showcase the attack's real-world deployability, by developing an IDE plug-in that stealthily injects INSEC into the GitHub Copilot extension.



## **4. TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks**

cs.LG

12 pages, 5 figures, in KDD 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11844v1) [paper-pdf](http://arxiv.org/pdf/2506.11844v1)

**Authors**: Qihai Zhang, Xinyue Sheng, Yuanfu Sun, Qiaoyu Tan

**Abstract**: Inspired by the success of large language models (LLMs), there is a significant research shift from traditional graph learning methods to LLM-based graph frameworks, formally known as GraphLLMs. GraphLLMs leverage the reasoning power of LLMs by integrating three key components: the textual attributes of input nodes, the structural information of node neighborhoods, and task-specific prompts that guide decision-making. Despite their promise, the robustness of GraphLLMs against adversarial perturbations remains largely unexplored-a critical concern for deploying these models in high-stakes scenarios. To bridge the gap, we introduce TrustGLM, a comprehensive study evaluating the vulnerability of GraphLLMs to adversarial attacks across three dimensions: text, graph structure, and prompt manipulations. We implement state-of-the-art attack algorithms from each perspective to rigorously assess model resilience. Through extensive experiments on six benchmark datasets from diverse domains, our findings reveal that GraphLLMs are highly susceptible to text attacks that merely replace a few semantically similar words in a node's textual attribute. We also find that standard graph structure attack methods can significantly degrade model performance, while random shuffling of the candidate label set in prompt templates leads to substantial performance drops. Beyond characterizing these vulnerabilities, we investigate defense techniques tailored to each attack vector through data-augmented training and adversarial training, which show promising potential to enhance the robustness of GraphLLMs. We hope that our open-sourced library will facilitate rapid, equitable evaluation and inspire further innovative research in this field.



## **5. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

cs.CL

Accepted to ACL 2025 Findings

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2406.14023v4) [paper-pdf](http://arxiv.org/pdf/2406.14023v4)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data, and benchmarks are available at https://yuchenwen1.github.io/ImplicitBiasEvaluation/.



## **6. Investigating Vulnerabilities and Defenses Against Audio-Visual Attacks: A Comprehensive Survey Emphasizing Multimodal Models**

cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11521v1) [paper-pdf](http://arxiv.org/pdf/2506.11521v1)

**Authors**: Jinming Wen, Xinyi Wu, Shuai Zhao, Yanhao Jia, Yuwen Li

**Abstract**: Multimodal large language models (MLLMs), which bridge the gap between audio-visual and natural language processing, achieve state-of-the-art performance on several audio-visual tasks. Despite the superior performance of MLLMs, the scarcity of high-quality audio-visual training data and computational resources necessitates the utilization of third-party data and open-source MLLMs, a trend that is increasingly observed in contemporary research. This prosperity masks significant security risks. Empirical studies demonstrate that the latest MLLMs can be manipulated to produce malicious or harmful content. This manipulation is facilitated exclusively through instructions or inputs, including adversarial perturbations and malevolent queries, effectively bypassing the internal security mechanisms embedded within the models. To gain a deeper comprehension of the inherent security vulnerabilities associated with audio-visual-based multimodal models, a series of surveys investigates various types of attacks, including adversarial and backdoor attacks. While existing surveys on audio-visual attacks provide a comprehensive overview, they are limited to specific types of attacks, which lack a unified review of various types of attacks. To address this issue and gain insights into the latest trends in the field, this paper presents a comprehensive and systematic review of audio-visual attacks, which include adversarial attacks, backdoor attacks, and jailbreak attacks. Furthermore, this paper also reviews various types of attacks in the latest audio-visual-based MLLMs, a dimension notably absent in existing surveys. Drawing upon comprehensive insights from a substantial review, this paper delineates both challenges and emergent trends for future research on audio-visual attacks and defense.



## **7. Bias Amplification in RAG: Poisoning Knowledge Retrieval to Steer LLMs**

cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11415v1) [paper-pdf](http://arxiv.org/pdf/2506.11415v1)

**Authors**: Linlin Wang, Tianqing Zhu, Laiqiao Qin, Longxiang Gao, Wanlei Zhou

**Abstract**: In Large Language Models, Retrieval-Augmented Generation (RAG) systems can significantly enhance the performance of large language models by integrating external knowledge. However, RAG also introduces new security risks. Existing research focuses mainly on how poisoning attacks in RAG systems affect model output quality, overlooking their potential to amplify model biases. For example, when querying about domestic violence victims, a compromised RAG system might preferentially retrieve documents depicting women as victims, causing the model to generate outputs that perpetuate gender stereotypes even when the original query is gender neutral. To show the impact of the bias, this paper proposes a Bias Retrieval and Reward Attack (BRRA) framework, which systematically investigates attack pathways that amplify language model biases through a RAG system manipulation. We design an adversarial document generation method based on multi-objective reward functions, employ subspace projection techniques to manipulate retrieval results, and construct a cyclic feedback mechanism for continuous bias amplification. Experiments on multiple mainstream large language models demonstrate that BRRA attacks can significantly enhance model biases in dimensions. In addition, we explore a dual stage defense mechanism to effectively mitigate the impacts of the attack. This study reveals that poisoning attacks in RAG systems directly amplify model output biases and clarifies the relationship between RAG system security and model fairness. This novel potential attack indicates that we need to keep an eye on the fairness issues of the RAG system.



## **8. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2405.06823v3) [paper-pdf](http://arxiv.org/pdf/2405.06823v3)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.



## **9. Improving LLM Safety Alignment with Dual-Objective Optimization**

cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2503.03710v2) [paper-pdf](http://arxiv.org/pdf/2503.03710v2)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment



## **10. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2401.17256v3) [paper-pdf](http://arxiv.org/pdf/2401.17256v3)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **11. PRSA: Prompt Stealing Attacks against Real-World Prompt Services**

cs.CR

This is the extended version of the paper accepted at the 34th USENIX  Security Symposium (USENIX Security 2025)

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2402.19200v3) [paper-pdf](http://arxiv.org/pdf/2402.19200v3)

**Authors**: Yong Yang, Changjiang Li, Qingming Li, Oubo Ma, Haoyu Wang, Zonghui Wang, Yandong Gao, Wenzhi Chen, Shouling Ji

**Abstract**: Recently, large language models (LLMs) have garnered widespread attention for their exceptional capabilities. Prompts are central to the functionality and performance of LLMs, making them highly valuable assets. The increasing reliance on high-quality prompts has driven significant growth in prompt services. However, this growth also expands the potential for prompt leakage, increasing the risk that attackers could replicate original functionalities, create competing products, and severely infringe on developers' intellectual property. Despite these risks, prompt leakage in real-world prompt services remains underexplored.   In this paper, we present PRSA, a practical attack framework designed for prompt stealing. PRSA infers the detailed intent of prompts through very limited input-output analysis and can successfully generate stolen prompts that replicate the original functionality. Extensive evaluations demonstrate PRSA's effectiveness across two main types of real-world prompt services. Specifically, compared to previous works, it improves the attack success rate from 17.8% to 46.1% in prompt marketplaces and from 39% to 52% in LLM application stores, respectively. Notably, in the attack on "Math", one of the most popular educational applications in OpenAI's GPT Store with over 1 million conversations, PRSA uncovered a hidden Easter egg that had not been revealed previously. Besides, our analysis reveals that higher mutual information between a prompt and its output correlates with an increased risk of leakage. This insight guides the design and evaluation of two potential defenses against the security threats posed by PRSA. We have reported these findings to the prompt service vendors, including PromptBase and OpenAI, and actively collaborate with them to implement defensive measures.



## **12. Unsourced Adversarial CAPTCHA: A Bi-Phase Adversarial CAPTCHA Framework**

cs.CV

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10685v1) [paper-pdf](http://arxiv.org/pdf/2506.10685v1)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: With the rapid advancements in deep learning, traditional CAPTCHA schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on original image characteristics, resulting in distortions that hinder human interpretation and limit applicability in scenarios lacking initial input images. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (UAC), a novel framework generating high-fidelity adversarial examples guided by attacker-specified text prompts. Leveraging a Large Language Model (LLM), UAC enhances CAPTCHA diversity and supports both targeted and untargeted attacks. For targeted attacks, the EDICT method optimizes dual latent variables in a diffusion model for superior image quality. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-UAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show BP-UAC achieves high attack success rates across diverse systems, generating natural CAPTCHAs indistinguishable to humans and DNNs.



## **13. SoK: Evaluating Jailbreak Guardrails for Large Language Models**

cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10597v1) [paper-pdf](http://arxiv.org/pdf/2506.10597v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Wenxuan Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress, but their deployment has exposed critical vulnerabilities, particularly to jailbreak attacks that circumvent safety mechanisms. Guardrails--external defense mechanisms that monitor and control LLM interaction--have emerged as a promising solution. However, the current landscape of LLM guardrails is fragmented, lacking a unified taxonomy and comprehensive evaluation framework. In this Systematization of Knowledge (SoK) paper, we present the first holistic analysis of jailbreak guardrails for LLMs. We propose a novel, multi-dimensional taxonomy that categorizes guardrails along six key dimensions, and introduce a Security-Efficiency-Utility evaluation framework to assess their practical effectiveness. Through extensive analysis and experiments, we identify the strengths and limitations of existing guardrail approaches, explore their universality across attack types, and provide insights into optimizing defense combinations. Our work offers a structured foundation for future research and development, aiming to guide the principled advancement and deployment of robust LLM guardrails. The code is available at https://github.com/xunguangwang/SoK4JailbreakGuardrails.



## **14. Towards Action Hijacking of Large Language Model-based Agent**

cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2412.10807v2) [paper-pdf](http://arxiv.org/pdf/2412.10807v2)

**Authors**: Yuyang Zhang, Kangjie Chen, Jiaxin Gao, Ronghao Cui, Run Wang, Lina Wang, Tianwei Zhang

**Abstract**: Recently, applications powered by Large Language Models (LLMs) have made significant strides in tackling complex tasks. By harnessing the advanced reasoning capabilities and extensive knowledge embedded in LLMs, these applications can generate detailed action plans that are subsequently executed by external tools. Furthermore, the integration of retrieval-augmented generation (RAG) enhances performance by incorporating up-to-date, domain-specific knowledge into the planning and execution processes. This approach has seen widespread adoption across various sectors, including healthcare, finance, and software development. Meanwhile, there are also growing concerns regarding the security of LLM-based applications. Researchers have disclosed various attacks, represented by jailbreak and prompt injection, to hijack the output actions of these applications. Existing attacks mainly focus on crafting semantically harmful prompts, and their validity could diminish when security filters are employed. In this paper, we introduce AI$\mathbf{^2}$, a novel attack to manipulate the action plans of LLM-based applications. Different from existing solutions, the innovation of AI$\mathbf{^2}$ lies in leveraging the knowledge from the application's database to facilitate the construction of malicious but semantically-harmless prompts. To this end, it first collects action-aware knowledge from the victim application. Based on such knowledge, the attacker can generate misleading input, which can mislead the LLM to generate harmful action plans, while bypassing possible detection mechanisms easily. Our evaluations on three real-world applications demonstrate the effectiveness of AI$\mathbf{^2}$: it achieves an average attack success rate of 84.30\% with the best of 99.70\%. Besides, it gets an average bypass rate of 92.7\% against common safety filters and 59.45\% against dedicated defense.



## **15. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2502.16750v4) [paper-pdf](http://arxiv.org/pdf/2502.16750v4)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **16. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

cs.AI

Accepted at VecDB @ ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2504.04858v2) [paper-pdf](http://arxiv.org/pdf/2504.04858v2)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.



## **17. SOFT: Selective Data Obfuscation for Protecting LLM Fine-tuning against Membership Inference Attacks**

cs.CR

Accepted by the 34th USENIX Security Symposium 2025. Code is  available at https://github.com/KaiyuanZh/SOFT

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10424v1) [paper-pdf](http://arxiv.org/pdf/2506.10424v1)

**Authors**: Kaiyuan Zhang, Siyuan Cheng, Hanxi Guo, Yuetian Chen, Zian Su, Shengwei An, Yuntao Du, Charles Fleming, Ashish Kundu, Xiangyu Zhang, Ninghui Li

**Abstract**: Large language models (LLMs) have achieved remarkable success and are widely adopted for diverse applications. However, fine-tuning these models often involves private or sensitive information, raising critical privacy concerns. In this work, we conduct the first comprehensive study evaluating the vulnerability of fine-tuned LLMs to membership inference attacks (MIAs). Our empirical analysis demonstrates that MIAs exploit the loss reduction during fine-tuning, making them highly effective in revealing membership information. These findings motivate the development of our defense. We propose SOFT (\textbf{S}elective data \textbf{O}bfuscation in LLM \textbf{F}ine-\textbf{T}uning), a novel defense technique that mitigates privacy leakage by leveraging influential data selection with an adjustable parameter to balance utility preservation and privacy protection. Our extensive experiments span six diverse domains and multiple LLM architectures and scales. Results show that SOFT effectively reduces privacy risks while maintaining competitive model performance, offering a practical and scalable solution to safeguard sensitive information in fine-tuned LLMs.



## **18. Can We Infer Confidential Properties of Training Data from LLMs?**

cs.LG

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10364v1) [paper-pdf](http://arxiv.org/pdf/2506.10364v1)

**Authors**: Penguin Huang, Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri

**Abstract**: Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs.



## **19. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2403.12503v2) [paper-pdf](http://arxiv.org/pdf/2403.12503v2)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He, Erfan Shayegani

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.



## **20. AURA: A Multi-Agent Intelligence Framework for Knowledge-Enhanced Cyber Threat Attribution**

cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.10175v1) [paper-pdf](http://arxiv.org/pdf/2506.10175v1)

**Authors**: Nanda Rani, Sandeep Kumar Shukla

**Abstract**: Effective attribution of Advanced Persistent Threats (APTs) increasingly hinges on the ability to correlate behavioral patterns and reason over complex, varied threat intelligence artifacts. We present AURA (Attribution Using Retrieval-Augmented Agents), a multi-agent, knowledge-enhanced framework for automated and interpretable APT attribution. AURA ingests diverse threat data including Tactics, Techniques, and Procedures (TTPs), Indicators of Compromise (IoCs), malware details, adversarial tools, and temporal information, which are processed through a network of collaborative agents. These agents are designed for intelligent query rewriting, context-enriched retrieval from structured threat knowledge bases, and natural language justification of attribution decisions. By combining Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs), AURA enables contextual linking of threat behaviors to known APT groups and supports traceable reasoning across multiple attack phases. Experiments on recent APT campaigns demonstrate AURA's high attribution consistency, expert-aligned justifications, and scalability. This work establishes AURA as a promising direction for advancing transparent, data-driven, and scalable threat attribution using multi-agent intelligence.



## **21. DAWN: Designing Distributed Agents in a Worldwide Network**

cs.NI

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2410.22339v3) [paper-pdf](http://arxiv.org/pdf/2410.22339v3)

**Authors**: Zahra Aminiranjbar, Jianan Tang, Qiudan Wang, Shubha Pant, Mahesh Viswanathan

**Abstract**: The rapid evolution of Large Language Models (LLMs) has transformed them from basic conversational tools into sophisticated entities capable of complex reasoning and decision-making. These advancements have led to the development of specialized LLM-based agents designed for diverse tasks such as coding and web browsing. As these agents become more capable, the need for a robust framework that facilitates global communication and collaboration among them towards advanced objectives has become increasingly critical. Distributed Agents in a Worldwide Network (DAWN) addresses this need by offering a versatile framework that integrates LLM-based agents with traditional software systems, enabling the creation of agentic applications suited for a wide range of use cases. DAWN enables distributed agents worldwide to register and be easily discovered through Gateway Agents. Collaborations among these agents are coordinated by a Principal Agent equipped with reasoning strategies. DAWN offers three operational modes: No-LLM Mode for deterministic tasks, Copilot for augmented decision-making, and LLM Agent for autonomous operations. Additionally, DAWN ensures the safety and security of agent collaborations globally through a dedicated safety, security, and compliance layer, protecting the network against attackers and adhering to stringent security and compliance standards. These features make DAWN a robust network for deploying agent-based applications across various industries.



## **22. Trustworthy AI: Safety, Bias, and Privacy -- A Survey**

cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2502.10450v2) [paper-pdf](http://arxiv.org/pdf/2502.10450v2)

**Authors**: Xingli Fang, Jianwei Li, Varun Mulchandani, Jung-Eun Kim

**Abstract**: The capabilities of artificial intelligence systems have been advancing to a great extent, but these systems still struggle with failure modes, vulnerabilities, and biases. In this paper, we study the current state of the field, and present promising insights and perspectives regarding concerns that challenge the trustworthiness of AI models. In particular, this paper investigates the issues regarding three thrusts: safety, privacy, and bias, which hurt models' trustworthiness. For safety, we discuss safety alignment in the context of large language models, preventing them from generating toxic or harmful content. For bias, we focus on spurious biases that can mislead a network. Lastly, for privacy, we cover membership inference attacks in deep neural networks. The discussions addressed in this paper reflect our own experiments and observations.



## **23. LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge**

cs.CR

Dataset at:  https://huggingface.co/datasets/microsoft/llmail-inject-challenge

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09956v1) [paper-pdf](http://arxiv.org/pdf/2506.09956v1)

**Authors**: Sahar Abdelnabi, Aideen Fay, Ahmed Salem, Egor Zverev, Kai-Chieh Liao, Chi-Huang Liu, Chun-Chih Kuo, Jannis Weigend, Danyael Manlangit, Alex Apostolov, Haris Umair, João Donato, Masayuki Kawakita, Athar Mahboob, Tran Huu Bach, Tsun-Han Chiang, Myeongjin Cho, Hajin Choi, Byeonghyeon Kim, Hyeonjin Lee, Benjamin Pannell, Conor McCauley, Mark Russinovich, Andrew Paverd, Giovanni Cherubin

**Abstract**: Indirect Prompt Injection attacks exploit the inherent limitation of Large Language Models (LLMs) to distinguish between instructions and data in their inputs. Despite numerous defense proposals, the systematic evaluation against adaptive adversaries remains limited, even when successful attacks can have wide security and privacy implications, and many real-world LLM-based applications remain vulnerable. We present the results of LLMail-Inject, a public challenge simulating a realistic scenario in which participants adaptively attempted to inject malicious instructions into emails in order to trigger unauthorized tool calls in an LLM-based email assistant. The challenge spanned multiple defense strategies, LLM architectures, and retrieval configurations, resulting in a dataset of 208,095 unique attack submissions from 839 participants. We release the challenge code, the full dataset of submissions, and our analysis demonstrating how this data can provide new insights into the instruction-data separation problem. We hope this will serve as a foundation for future research towards practical structural solutions to prompt injection.



## **24. CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization**

cs.CL

Accepted at ICML 2025, 20 pages

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2411.12768v2) [paper-pdf](http://arxiv.org/pdf/2411.12768v2)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Large Language Models (LLMs) are vulnerable to backdoor attacks that manipulate outputs via hidden triggers. Existing defense methods--designed for vision/text classification tasks--fail for text generation. We propose Internal Consistency Regularization (CROW), a defense leveraging the observation that backdoored models exhibit unstable layer-wise hidden representations when triggered, while clean models show smooth transitions. CROW enforces consistency across layers via adversarial perturbations and regularization during finetuning, neutralizing backdoors without requiring clean reference models or trigger knowledge--only a small clean dataset. Experiments across Llama-2 (7B, 13B), CodeLlama (7B, 13B), and Mistral-7B demonstrate CROW's effectiveness: it achieves significant reductions in attack success rates across diverse backdoor strategies (sentiment steering, targeted refusal, code injection) while preserving generative performance. CROW's architecture-agnostic design enables practical deployment.



## **25. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

cs.AI

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.07736v2) [paper-pdf](http://arxiv.org/pdf/2506.07736v2)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.



## **26. Design Patterns for Securing LLM Agents against Prompt Injections**

cs.LG

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.08837v2) [paper-pdf](http://arxiv.org/pdf/2506.08837v2)

**Authors**: Luca Beurer-Kellner, Beat Buesser Ana-Maria Creţu, Edoardo Debenedetti, Daniel Dobos, Daniel Fabian, Marc Fischer, David Froelicher, Kathrin Grosse, Daniel Naeff, Ezinwanne Ozoani, Andrew Paverd, Florian Tramèr, Václav Volhejn

**Abstract**: As AI agents powered by Large Language Models (LLMs) become increasingly versatile and capable of addressing a broad spectrum of tasks, ensuring their security has become a critical challenge. Among the most pressing threats are prompt injection attacks, which exploit the agent's resilience on natural language inputs -- an especially dangerous threat when agents are granted tool access or handle sensitive information. In this work, we propose a set of principled design patterns for building AI agents with provable resistance to prompt injection. We systematically analyze these patterns, discuss their trade-offs in terms of utility and security, and illustrate their real-world applicability through a series of case studies.



## **27. GenBreak: Red Teaming Text-to-Image Generators Using Large Language Models**

cs.CR

27 pages, 7 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.10047v1) [paper-pdf](http://arxiv.org/pdf/2506.10047v1)

**Authors**: Zilong Wang, Xiang Zheng, Xiaosen Wang, Bo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Text-to-image (T2I) models such as Stable Diffusion have advanced rapidly and are now widely used in content creation. However, these models can be misused to generate harmful content, including nudity or violence, posing significant safety risks. While most platforms employ content moderation systems, underlying vulnerabilities can still be exploited by determined adversaries. Recent research on red-teaming and adversarial attacks against T2I models has notable limitations: some studies successfully generate highly toxic images but use adversarial prompts that are easily detected and blocked by safety filters, while others focus on bypassing safety mechanisms but fail to produce genuinely harmful outputs, neglecting the discovery of truly high-risk prompts. Consequently, there remains a lack of reliable tools for evaluating the safety of defended T2I models. To address this gap, we propose GenBreak, a framework that fine-tunes a red-team large language model (LLM) to systematically explore underlying vulnerabilities in T2I generators. Our approach combines supervised fine-tuning on curated datasets with reinforcement learning via interaction with a surrogate T2I model. By integrating multiple reward signals, we guide the LLM to craft adversarial prompts that enhance both evasion capability and image toxicity, while maintaining semantic coherence and diversity. These prompts demonstrate strong effectiveness in black-box attacks against commercial T2I generators, revealing practical and concerning safety weaknesses.



## **28. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

cs.CV

31 pages, 8 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.05982v2) [paper-pdf](http://arxiv.org/pdf/2506.05982v2)

**Authors**: Zonglin Wu, Yule Xue, Xin Wei, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.



## **29. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge**

cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09443v1) [paper-pdf](http://arxiv.org/pdf/2506.09443v1)

**Authors**: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable intelligence across various tasks, which has inspired the development and widespread adoption of LLM-as-a-Judge systems for automated model testing, such as red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising concerns about their robustness and, consequently, their trustworthiness. Existing evaluation methods adopted by LLM-based judges are often piecemeal and lack a unified framework for comprehensive assessment. Furthermore, prompt template and model selections for improving judge robustness have been rarely explored, and their performance in real-world settings remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. RobustJudge investigates the impact of attack methods and defense strategies (RQ1), explores the influence of prompt template and model selection (RQ2), and assesses the robustness of real-world LLM-as-a-Judge applications (RQ3).Our main findings are: (1) LLM-as-a-Judge systems are still vulnerable to a range of adversarial attacks, including Combined Attack and PAIR, while defense mechanisms such as Re-tokenization and LLM-based Detectors offer improved protection; (2) Robustness is highly sensitive to the choice of prompt template and judge models. Our proposed prompt template optimization method can improve robustness, and JudgeLM-13B demonstrates strong performance as a robust open-source judge; (3) Applying RobustJudge to Alibaba's PAI platform reveals previously unreported vulnerabilities. The source code of RobustJudge is provided at https://github.com/S3IC-Lab/RobustJudge.



## **30. Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding**

cs.AI

To appear in ACL 2025

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2406.15481v3) [paper-pdf](http://arxiv.org/pdf/2406.15481v3)

**Authors**: Haneul Yoo, Yongjin Yang, Hwaran Lee

**Abstract**: As large language models (LLMs) have advanced rapidly, concerns regarding their safety have become prominent. In this paper, we discover that code-switching in red-teaming queries can effectively elicit undesirable behaviors of LLMs, which are common practices in natural language. We introduce a simple yet effective framework, CSRT, to synthesize codeswitching red-teaming queries and investigate the safety and multilingual understanding of LLMs comprehensively. Through extensive experiments with ten state-of-the-art LLMs and code-switching queries combining up to 10 languages, we demonstrate that the CSRT significantly outperforms existing multilingual red-teaming techniques, achieving 46.7% more attacks than standard attacks in English and being effective in conventional safety domains. We also examine the multilingual ability of those LLMs to generate and understand codeswitching texts. Additionally, we validate the extensibility of the CSRT by generating codeswitching attack prompts with monolingual data. We finally conduct detailed ablation studies exploring code-switching and propound unintended correlation between resource availability of languages and safety alignment in existing multilingual LLMs.



## **31. Automatic Pseudo-Harmful Prompt Generation for Evaluating False Refusals in Large Language Models**

cs.CL

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2409.00598v2) [paper-pdf](http://arxiv.org/pdf/2409.00598v2)

**Authors**: Bang An, Sicheng Zhu, Ruiyi Zhang, Michael-Andrei Panaitescu-Liess, Yuancheng Xu, Furong Huang

**Abstract**: Safety-aligned large language models (LLMs) sometimes falsely refuse pseudo-harmful prompts, like "how to kill a mosquito," which are actually harmless. Frequent false refusals not only frustrate users but also provoke a public backlash against the very values alignment seeks to protect. In this paper, we propose the first method to auto-generate diverse, content-controlled, and model-dependent pseudo-harmful prompts. Using this method, we construct an evaluation dataset called PHTest, which is ten times larger than existing datasets, covers more false refusal patterns, and separately labels controversial prompts. We evaluate 20 LLMs on PHTest, uncovering new insights due to its scale and labeling. Our findings reveal a trade-off between minimizing false refusals and improving safety against jailbreak attacks. Moreover, we show that many jailbreak defenses significantly increase the false refusal rates, thereby undermining usability. Our method and dataset can help developers evaluate and fine-tune safer and more usable LLMs. Our code and dataset are available at https://github.com/umd-huang-lab/FalseRefusal



## **32. Detecting State Manipulation Vulnerabilities in Smart Contracts Using LLM and Static Analysis**

cs.SE

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.08561v2) [paper-pdf](http://arxiv.org/pdf/2506.08561v2)

**Authors**: Hao Wu, Haijun Wang, Shangwang Li, Yin Wu, Ming Fan, Yitao Zhao, Ting Liu

**Abstract**: An increasing number of DeFi protocols are gaining popularity, facilitating transactions among multiple anonymous users. State Manipulation is one of the notorious attacks in DeFi smart contracts, with price variable being the most commonly exploited state variable-attackers manipulate token prices to gain illicit profits. In this paper, we propose PriceSleuth, a novel method that leverages the Large Language Model (LLM) and static analysis to detect Price Manipulation (PM) attacks proactively. PriceSleuth firstly identifies core logic function related to price calculation in DeFi contracts. Then it guides LLM to locate the price calculation code statements. Secondly, PriceSleuth performs backward dependency analysis of price variables, instructing LLM in detecting potential price manipulation. Finally, PriceSleuth utilizes propagation analysis of price variables to assist LLM in detecting whether these variables are maliciously exploited. We presented preliminary experimental results to substantiate the effectiveness of PriceSleuth . And we outline future research directions for PriceSleuth.



## **33. Your Agent Can Defend Itself against Backdoor Attacks**

cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.08336v2) [paper-pdf](http://arxiv.org/pdf/2506.08336v2)

**Authors**: Li Changjiang, Liang Jiacheng, Cao Bochuan, Chen Jinghui, Wang Ting

**Abstract**: Despite their growing adoption across domains, large language model (LLM)-powered agents face significant security risks from backdoor attacks during training and fine-tuning. These compromised agents can subsequently be manipulated to execute malicious operations when presented with specific triggers in their inputs or environments. To address this pressing risk, we present ReAgent, a novel defense against a range of backdoor attacks on LLM-based agents. Intuitively, backdoor attacks often result in inconsistencies among the user's instruction, the agent's planning, and its execution. Drawing on this insight, ReAgent employs a two-level approach to detect potential backdoors. At the execution level, ReAgent verifies consistency between the agent's thoughts and actions; at the planning level, ReAgent leverages the agent's capability to reconstruct the instruction based on its thought trajectory, checking for consistency between the reconstructed instruction and the user's instruction. Extensive evaluation demonstrates ReAgent's effectiveness against various backdoor attacks across tasks. For instance, ReAgent reduces the attack success rate by up to 90\% in database operation tasks, outperforming existing defenses by large margins. This work reveals the potential of utilizing compromised agents themselves to mitigate backdoor risks.



## **34. FC-Attack: Jailbreaking Multimodal Large Language Models via Auto-Generated Flowcharts**

cs.CV

13 pages, 7 figures

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2502.21059v2) [paper-pdf](http://arxiv.org/pdf/2502.21059v2)

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Jihui Guo, Xinlei He

**Abstract**: Multimodal Large Language Models (MLLMs) have become powerful and widely adopted in some practical applications. However, recent research has revealed their vulnerability to multimodal jailbreak attacks, whereby the model can be induced to generate harmful content, leading to safety risks. Although most MLLMs have undergone safety alignment, recent research shows that the visual modality is still vulnerable to jailbreak attacks. In our work, we discover that by using flowcharts with partially harmful information, MLLMs can be induced to provide additional harmful details. Based on this, we propose a jailbreak attack method based on auto-generated flowcharts, FC-Attack. Specifically, FC-Attack first fine-tunes a pre-trained LLM to create a step-description generator based on benign datasets. The generator is then used to produce step descriptions corresponding to a harmful query, which are transformed into flowcharts in 3 different shapes (vertical, horizontal, and S-shaped) as visual prompts. These flowcharts are then combined with a benign textual prompt to execute the jailbreak attack on MLLMs. Our evaluations on Advbench show that FC-Attack attains an attack success rate of up to 96% via images and up to 78% via videos across multiple MLLMs. Additionally, we investigate factors affecting the attack performance, including the number of steps and the font styles in the flowcharts. We also find that FC-Attack can improve the jailbreak performance from 4% to 28% in Claude-3.5 by changing the font style. To mitigate the attack, we explore several defenses and find that AdaShield can largely reduce the jailbreak performance but with the cost of utility drop.



## **35. PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**

cs.CR

21 pages, 7 figures

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2411.17453v2) [paper-pdf](http://arxiv.org/pdf/2411.17453v2)

**Authors**: Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang

**Abstract**: Fine-tuning is an essential process to improve the performance of Large Language Models (LLMs) in specific domains, with Parameter-Efficient Fine-Tuning (PEFT) gaining popularity due to its capacity to reduce computational demands through the integration of low-rank adapters. These lightweight adapters, such as LoRA, can be shared and utilized on open-source platforms. However, adversaries could exploit this mechanism to inject backdoors into these adapters, resulting in malicious behaviors like incorrect or harmful outputs, which pose serious security risks to the community. Unfortunately, few current efforts concentrate on analyzing the backdoor patterns or detecting the backdoors in the adapters. To fill this gap, we first construct and release PADBench, a comprehensive benchmark that contains 13,300 benign and backdoored adapters fine-tuned with various datasets, attack strategies, PEFT methods, and LLMs. Moreover, we propose PEFTGuard, the first backdoor detection framework against PEFT-based adapters. Extensive evaluation upon PADBench shows that PEFTGuard outperforms existing detection methods, achieving nearly perfect detection accuracy (100%) in most cases. Notably, PEFTGuard exhibits zero-shot transferability on three aspects, including different attacks, PEFT methods, and adapter ranks. In addition, we consider various adaptive attacks to demonstrate the high robustness of PEFTGuard. We further explore several possible backdoor mitigation defenses, finding fine-mixing to be the most effective method. We envision that our benchmark and method can shed light on future LLM backdoor detection research.



## **36. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

cs.CR

Pre-print

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2412.07192v2) [paper-pdf](http://arxiv.org/pdf/2412.07192v2)

**Authors**: Zachary Coalson, Jeonghyun Woo, Yu Sun, Shiyang Chen, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.



## **37. Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs**

cs.CV

14 pages, 5 figures

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.01064v2) [paper-pdf](http://arxiv.org/pdf/2506.01064v2)

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. Despite their potential impact, the development of effective methods for purifying such adversarial examples has received relatively limited attention. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive "fighting fire with fire" strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code will be made publicly available.



## **38. ASIDE: Architectural Separation of Instructions and Data in Language Models**

cs.LG

Preliminary version accepted to ICLR 2025 Workshop on Building Trust  in Language Models and Applications

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2503.10566v3) [paper-pdf](http://arxiv.org/pdf/2503.10566v3)

**Authors**: Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Alexandra Volkova, Soroush Tabesh, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert

**Abstract**: Despite their remarkable performance, large language models lack elementary safety features, making them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as a root cause of the success of prompt injection attacks. In this work, we propose a new architectural element, ASIDE, that allows language models to clearly separate instructions and data at the level of embeddings. ASIDE applies an orthogonal rotation to the embeddings of data tokens, thus creating clearly distinct representations of instructions and data tokens without introducing any additional parameters. As we demonstrate experimentally across a range of models, instruction-tuning LLMs with ASIDE (1) leads to highly increased instruction-data separation without a loss in model utility and (2) makes the models more robust to prompt injection benchmarks, even without dedicated safety training. Additionally, we provide insights into the mechanism underlying our method through an analysis of the model representations. The source code and training scripts are openly accessible at https://github.com/egozverev/aside.



## **39. On the Ethics of Using LLMs for Offensive Security**

cs.CR

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.08693v1) [paper-pdf](http://arxiv.org/pdf/2506.08693v1)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Large Language Models (LLMs) have rapidly evolved over the past few years and are currently evaluated for their efficacy within the domain of offensive cyber-security. While initial forays showcase the potential of LLMs to enhance security research, they also raise critical ethical concerns regarding the dual-use of offensive security tooling.   This paper analyzes a set of papers that leverage LLMs for offensive security, focusing on how ethical considerations are expressed and justified in their work. The goal is to assess the culture of AI in offensive security research regarding ethics communication, highlighting trends, best practices, and gaps in current discourse.   We provide insights into how the academic community navigates the fine line between innovation and ethical responsibility. Particularly, our results show that 13 of 15 reviewed prototypes (86.6\%) mentioned ethical considerations and are thus aware of the potential dual-use of their research. Main motivation given for the research was allowing broader access to penetration-testing as well as preparing defenders for AI-guided attackers.



## **40. ASRJam: Human-Friendly AI Speech Jamming to Prevent Automated Phone Scams**

cs.CL

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.11125v1) [paper-pdf](http://arxiv.org/pdf/2506.11125v1)

**Authors**: Freddie Grabovski, Gilad Gressel, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs), combined with Text-to-Speech (TTS) and Automatic Speech Recognition (ASR), are increasingly used to automate voice phishing (vishing) scams. These systems are scalable and convincing, posing a significant security threat. We identify the ASR transcription step as the most vulnerable link in the scam pipeline and introduce ASRJam, a proactive defence framework that injects adversarial perturbations into the victim's audio to disrupt the attacker's ASR. This breaks the scam's feedback loop without affecting human callers, who can still understand the conversation. While prior adversarial audio techniques are often unpleasant and impractical for real-time use, we also propose EchoGuard, a novel jammer that leverages natural distortions, such as reverberation and echo, that are disruptive to ASR but tolerable to humans. To evaluate EchoGuard's effectiveness and usability, we conducted a 39-person user study comparing it with three state-of-the-art attacks. Results show that EchoGuard achieved the highest overall utility, offering the best combination of ASR disruption and human listening experience.



## **41. Evaluation empirique de la sécurisation et de l'alignement de ChatGPT et Gemini: analyse comparative des vulnérabilités par expérimentations de jailbreaks**

cs.CR

in French language

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.10029v1) [paper-pdf](http://arxiv.org/pdf/2506.10029v1)

**Authors**: Rafaël Nouailles

**Abstract**: Large Language models (LLMs) are transforming digital usage, particularly in text generation, image creation, information retrieval and code development. ChatGPT, launched by OpenAI in November 2022, quickly became a reference, prompting the emergence of competitors such as Google's Gemini. However, these technological advances raise new cybersecurity challenges, including prompt injection attacks, the circumvention of regulatory measures (jailbreaking), the spread of misinformation (hallucinations) and risks associated with deep fakes. This paper presents a comparative analysis of the security and alignment levels of ChatGPT and Gemini, as well as a taxonomy of jailbreak techniques associated with experiments.



## **42. SPBA: Utilizing Speech Large Language Model for Backdoor Attacks on Speech Classification Models**

cs.SD

Accepted by IJCNN 2025

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.08346v1) [paper-pdf](http://arxiv.org/pdf/2506.08346v1)

**Authors**: Wenhan Yao, Fen Xiao, Xiarun Chen, Jia Liu, YongQiang He, Weiping Wen

**Abstract**: Deep speech classification tasks, including keyword spotting and speaker verification, are vital in speech-based human-computer interaction. Recently, the security of these technologies has been revealed to be susceptible to backdoor attacks. Specifically, attackers use noisy disruption triggers and speech element triggers to produce poisoned speech samples that train models to become vulnerable. However, these methods typically create only a limited number of backdoors due to the inherent constraints of the trigger function. In this paper, we propose that speech backdoor attacks can strategically focus on speech elements such as timbre and emotion, leveraging the Speech Large Language Model (SLLM) to generate diverse triggers. Increasing the number of triggers may disproportionately elevate the poisoning rate, resulting in higher attack costs and a lower success rate per trigger. We introduce the Multiple Gradient Descent Algorithm (MGDA) as a mitigation strategy to address this challenge. The proposed attack is called the Speech Prompt Backdoor Attack (SPBA). Building on this foundation, we conducted attack experiments on two speech classification tasks, demonstrating that SPBA shows significant trigger effectiveness and achieves exceptional performance in attack metrics.



## **43. R.R.: Unveiling LLM Training Privacy through Recollection and Ranking**

cs.CL

13 pages, 9 figures; typos corrected

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2502.12658v2) [paper-pdf](http://arxiv.org/pdf/2502.12658v2)

**Authors**: Wenlong Meng, Zhenyuan Guo, Lenan Wu, Chen Gong, Wenyan Liu, Weixian Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large Language Models (LLMs) pose significant privacy risks, potentially leaking training data due to implicit memorization. Existing privacy attacks primarily focus on membership inference attacks (MIAs) or data extraction attacks, but reconstructing specific personally identifiable information (PII) in LLMs' training data remains challenging. In this paper, we propose R.R. (Recollect and Rank), a novel two-step privacy stealing attack that enables attackers to reconstruct PII entities from scrubbed training data where the PII entities have been masked. In the first stage, we introduce a prompt paradigm named recollection, which instructs the LLM to repeat a masked text but fill in masks. Then we can use PII identifiers to extract recollected PII candidates. In the second stage, we design a new criterion to score each PII candidate and rank them. Motivated by membership inference, we leverage the reference model as a calibration to our criterion. Experiments across three popular PII datasets demonstrate that the R.R. achieves better PII identification performance than baselines. These results highlight the vulnerability of LLMs to PII leakage even when training data has been scrubbed. We release our code and datasets at GitHub.



## **44. Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models**

cs.CR

To be published at ACL 2025 (Main)

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.10024v1) [paper-pdf](http://arxiv.org/pdf/2506.10024v1)

**Authors**: Elena Sofia Ruzzetti, Giancarlo A. Xompero, Davide Venditti, Fabio Massimo Zanzotto

**Abstract**: Large Language Models (LLMs) memorize, and thus, among huge amounts of uncontrolled data, may memorize Personally Identifiable Information (PII), which should not be stored and, consequently, not leaked. In this paper, we introduce Private Memorization Editing (PME), an approach for preventing private data leakage that turns an apparent limitation, that is, the LLMs' memorization ability, into a powerful privacy defense strategy. While attacks against LLMs have been performed exploiting previous knowledge regarding their training data, our approach aims to exploit the same kind of knowledge in order to make a model more robust. We detect a memorized PII and then mitigate the memorization of PII by editing a model knowledge of its training data. We verify that our procedure does not affect the underlying language model while making it more robust against privacy Training Data Extraction attacks. We demonstrate that PME can effectively reduce the number of leaked PII in a number of configurations, in some cases even reducing the accuracy of the privacy attacks to zero.



## **45. TokenBreak: Bypassing Text Classification Models Through Token Manipulation**

cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07948v1) [paper-pdf](http://arxiv.org/pdf/2506.07948v1)

**Authors**: Kasimir Schulz, Kenneth Yeung, Kieran Evans

**Abstract**: Natural Language Processing (NLP) models are used for text-related tasks such as classification and generation. To complete these tasks, input data is first tokenized from human-readable text into a format the model can understand, enabling it to make inferences and understand context. Text classification models can be implemented to guard against threats such as prompt injection attacks against Large Language Models (LLMs), toxic input and cybersecurity risks such as spam emails. In this paper, we introduce TokenBreak: a novel attack that can bypass these protection models by taking advantage of the tokenization strategy they use. This attack technique manipulates input text in such a way that certain models give an incorrect classification. Importantly, the end target (LLM or email recipient) can still understand and respond to the manipulated text and therefore be vulnerable to the very attack the protection model was put in place to prevent. The tokenizer is tied to model architecture, meaning it is possible to predict whether or not a model is vulnerable to attack based on family. We also present a defensive strategy as an added layer of protection that can be implemented without having to retrain the defensive model.



## **46. Adversarial Attack Classification and Robustness Testing for Large Language Models for Code**

cs.SE

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07942v1) [paper-pdf](http://arxiv.org/pdf/2506.07942v1)

**Authors**: Yang Liu, Armstrong Foundjem, Foutse Khomh, Heng Li

**Abstract**: Large Language Models (LLMs) have become vital tools in software development tasks such as code generation, completion, and analysis. As their integration into workflows deepens, ensuring robustness against vulnerabilities especially those triggered by diverse or adversarial inputs becomes increasingly important. Such vulnerabilities may lead to incorrect or insecure code generation when models encounter perturbed task descriptions, code, or comments. Prior research often overlooks the role of natural language in guiding code tasks. This study investigates how adversarial perturbations in natural language inputs including prompts, comments, and descriptions affect LLMs for Code (LLM4Code). It examines the effects of perturbations at the character, word, and sentence levels to identify the most impactful vulnerabilities. We analyzed multiple projects (e.g., ReCode, OpenAttack) and datasets (e.g., HumanEval, MBPP), establishing a taxonomy of adversarial attacks. The first dimension classifies the input type code, prompts, or comments while the second dimension focuses on granularity: character, word, or sentence-level changes. We adopted a mixed-methods approach, combining quantitative performance metrics with qualitative vulnerability analysis. LLM4Code models show varying robustness across perturbation types. Sentence-level attacks were least effective, suggesting models are resilient to broader contextual changes. In contrast, word-level perturbations posed serious challenges, exposing semantic vulnerabilities. Character-level effects varied, showing model sensitivity to subtle syntactic deviations.Our study offers a structured framework for testing LLM4Code robustness and emphasizes the critical role of natural language in adversarial evaluation. Improving model resilience to semantic-level disruptions is essential for secure and reliable code-generation systems.



## **47. SoK: Data Reconstruction Attacks Against Machine Learning Models: Definition, Metrics, and Benchmark**

cs.CR

To Appear in the 34th USENIX Security Symposium, August 13-15, 2025

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07888v1) [paper-pdf](http://arxiv.org/pdf/2506.07888v1)

**Authors**: Rui Wen, Yiyong Liu, Michael Backes, Yang Zhang

**Abstract**: Data reconstruction attacks, which aim to recover the training dataset of a target model with limited access, have gained increasing attention in recent years. However, there is currently no consensus on a formal definition of data reconstruction attacks or appropriate evaluation metrics for measuring their quality. This lack of rigorous definitions and universal metrics has hindered further advancement in this field. In this paper, we address this issue in the vision domain by proposing a unified attack taxonomy and formal definitions of data reconstruction attacks. We first propose a set of quantitative evaluation metrics that consider important criteria such as quantifiability, consistency, precision, and diversity. Additionally, we leverage large language models (LLMs) as a substitute for human judgment, enabling visual evaluation with an emphasis on high-quality reconstructions. Using our proposed taxonomy and metrics, we present a unified framework for systematically evaluating the strengths and limitations of existing attacks and establishing a benchmark for future research. Empirical results, primarily from a memorization perspective, not only validate the effectiveness of our metrics but also offer valuable insights for designing new attacks.



## **48. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2406.12091v4) [paper-pdf](http://arxiv.org/pdf/2406.12091v4)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.



## **49. Representation Bending for Large Language Model Safety**

cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2504.01550v2) [paper-pdf](http://arxiv.org/pdf/2504.01550v2)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.



## **50. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

cs.CL

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2505.17654v2) [paper-pdf](http://arxiv.org/pdf/2505.17654v2)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.



