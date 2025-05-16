# Latest Large Language Model Attack Papers
**update at 2025-05-16 16:49:08**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. S3C2 Summit 2024-09: Industry Secure Software Supply Chain Summit**

cs.CR

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.10538v1) [paper-pdf](http://arxiv.org/pdf/2505.10538v1)

**Authors**: Imranur Rahman, Yasemin Acar, Michel Cukier, William Enck, Christian Kastner, Alexandros Kapravelos, Dominik Wermke, Laurie Williams

**Abstract**: While providing economic and software development value, software supply chains are only as strong as their weakest link. Over the past several years, there has been an exponential increase in cyberattacks, specifically targeting vulnerable links in critical software supply chains. These attacks disrupt the day-to-day functioning and threaten the security of nearly everyone on the internet, from billion-dollar companies and government agencies to hobbyist open-source developers. The ever-evolving threat of software supply chain attacks has garnered interest from the software industry and the US government in improving software supply chain security.   On September 20, 2024, three researchers from the NSF-backed Secure Software Supply Chain Center (S3C2) conducted a Secure Software Supply Chain Summit with a diverse set of 12 practitioners from 9 companies. The goals of the Summit were to: (1) to enable sharing between individuals from different companies regarding practical experiences and challenges with software supply chain security, (2) to help form new collaborations, (3) to share our observations from our previous summits with industry, and (4) to learn about practitioners' challenges to inform our future research direction. The summit consisted of discussions of six topics relevant to the companies represented, including updating vulnerable dependencies, component and container choice, malicious commits, building infrastructure, large language models, and reducing entire classes of vulnerabilities.



## **2. MapExplorer: New Content Generation from Low-Dimensional Visualizations**

cs.AI

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2412.18673v2) [paper-pdf](http://arxiv.org/pdf/2412.18673v2)

**Authors**: Xingjian Zhang, Ziyang Xiong, Shixuan Liu, Yutong Xie, Tolga Ergen, Dongsub Shim, Hua Xu, Honglak Lee, Qiaozhu Me

**Abstract**: Low-dimensional visualizations, or "projection maps," are widely used in scientific and creative domains to interpret large-scale and complex datasets. These visualizations not only aid in understanding existing knowledge spaces but also implicitly guide exploration into unknown areas. Although techniques such as t-SNE and UMAP can generate these maps, there exists no systematic method for leveraging them to generate new content. To address this, we introduce MapExplorer, a novel knowledge discovery task that translates coordinates within any projection map into coherent, contextually aligned textual content. This allows users to interactively explore and uncover insights embedded in the maps. To evaluate the performance of MapExplorer methods, we propose Atometric, a fine-grained metric inspired by ROUGE that quantifies logical coherence and alignment between generated and reference text. Experiments on diverse datasets demonstrate the versatility of MapExplorer in generating scientific hypotheses, crafting synthetic personas, and devising strategies for attacking large language models-even with simple baseline methods. By bridging visualization and generation, our work highlights the potential of MapExplorer to enable intuitive human-AI collaboration in large-scale data exploration.



## **3. Scaling Laws for Black box Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2411.16782v2) [paper-pdf](http://arxiv.org/pdf/2411.16782v2)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.



## **4. Dark LLMs: The Growing Threat of Unaligned AI Models**

cs.CL

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.10066v1) [paper-pdf](http://arxiv.org/pdf/2505.10066v1)

**Authors**: Michael Fire, Yitzhak Elbazis, Adi Wasenstein, Lior Rokach

**Abstract**: Large Language Models (LLMs) rapidly reshape modern life, advancing fields from healthcare to education and beyond. However, alongside their remarkable capabilities lies a significant threat: the susceptibility of these models to jailbreaking. The fundamental vulnerability of LLMs to jailbreak attacks stems from the very data they learn from. As long as this training data includes unfiltered, problematic, or 'dark' content, the models can inherently learn undesirable patterns or weaknesses that allow users to circumvent their intended safety controls. Our research identifies the growing threat posed by dark LLMs models deliberately designed without ethical guardrails or modified through jailbreak techniques. In our research, we uncovered a universal jailbreak attack that effectively compromises multiple state-of-the-art models, enabling them to answer almost any question and produce harmful outputs upon request. The main idea of our attack was published online over seven months ago. However, many of the tested LLMs were still vulnerable to this attack. Despite our responsible disclosure efforts, responses from major LLM providers were often inadequate, highlighting a concerning gap in industry practices regarding AI safety. As model training becomes more accessible and cheaper, and as open-source LLMs proliferate, the risk of widespread misuse escalates. Without decisive intervention, LLMs may continue democratizing access to dangerous knowledge, posing greater risks than anticipated.



## **5. Large Language Models for Cyber Security: A Systematic Literature Review**

cs.CR

56 pages,6 figures

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2405.04760v4) [paper-pdf](http://arxiv.org/pdf/2405.04760v4)

**Authors**: Hanxiang Xu, Shenao Wang, Ningke Li, Kailong Wang, Yanjie Zhao, Kai Chen, Ting Yu, Yang Liu, Haoyu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in various domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity (LLM4Security). By comprehensively collecting over 30K relevant papers and systematically analyzing 127 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to a wide range of cybersecurity tasks, including vulnerability detection, malware analysis, network intrusion detection, and phishing detection. Second, we find that the datasets used for training and evaluating LLMs in these tasks are often limited in size and diversity, highlighting the need for more comprehensive and representative datasets. Third, we identify several promising techniques for adapting LLMs to specific cybersecurity domains, such as fine-tuning, transfer learning, and domain-specific pre-training. Finally, we discuss the main challenges and opportunities for future research in LLM4Security, including the need for more interpretable and explainable models, the importance of addressing data privacy and security concerns, and the potential for leveraging LLMs for proactive defense and threat hunting. Overall, our survey provides a comprehensive overview of the current state-of-the-art in LLM4Security and identifies several promising directions for future research.



## **6. PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization**

cs.CR

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.09921v1) [paper-pdf](http://arxiv.org/pdf/2505.09921v1)

**Authors**: Yidan Wang, Yanan Cao, Yubing Ren, Fang Fang, Zheng Lin, Binxing Fang

**Abstract**: Large Language Models (LLMs) excel in various domains but pose inherent privacy risks. Existing methods to evaluate privacy leakage in LLMs often use memorized prefixes or simple instructions to extract data, both of which well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM safety mechanisms to generate harmful content, but their role in privacy scenarios remains underexplored. In this paper, we examine the effectiveness of jailbreak attacks in extracting sensitive information, bridging privacy leakage and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework targeting Personally Identifiable Information (PII) and addressing the limitations of current jailbreak methods. Specifically, PIG identifies PII entities and their types in privacy queries, uses in-context learning to build a privacy context, and iteratively updates it with three gradient-based strategies to elicit target PII. We evaluate PIG and existing jailbreak methods using two privacy-related datasets. Experiments on four white-box and two black-box LLMs show that PIG outperforms baseline methods and achieves state-of-the-art (SoTA) results. The results underscore significant privacy risks in LLMs, emphasizing the need for stronger safeguards. Our code is availble at \href{https://github.com/redwyd/PrivacyJailbreak}{https://github.com/redwyd/PrivacyJailbreak}.



## **7. Adversarial Attack on Large Language Models using Exponentiated Gradient Descent**

cs.LG

Accepted to International Joint Conference on Neural Networks (IJCNN)  2025

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09820v1) [paper-pdf](http://arxiv.org/pdf/2505.09820v1)

**Authors**: Sajib Biswas, Mao Nishino, Samuel Jacob Chacko, Xiuwen Liu

**Abstract**: As Large Language Models (LLMs) are widely used, understanding them systematically is key to improving their safety and realizing their full potential. Although many models are aligned using techniques such as reinforcement learning from human feedback (RLHF), they are still vulnerable to jailbreaking attacks. Some of the existing adversarial attack methods search for discrete tokens that may jailbreak a target model while others try to optimize the continuous space represented by the tokens of the model's vocabulary. While techniques based on the discrete space may prove to be inefficient, optimization of continuous token embeddings requires projections to produce discrete tokens, which might render them ineffective. To fully utilize the constraints and the structures of the space, we develop an intrinsic optimization technique using exponentiated gradient descent with the Bregman projection method to ensure that the optimized one-hot encoding always stays within the probability simplex. We prove the convergence of the technique and implement an efficient algorithm that is effective in jailbreaking several widely used LLMs. We demonstrate the efficacy of the proposed technique using five open-source LLMs on four openly available datasets. The results show that the technique achieves a higher success rate with great efficiency compared to three other state-of-the-art jailbreaking techniques. The source code for our implementation is available at: https://github.com/sbamit/Exponentiated-Gradient-Descent-LLM-Attack



## **8. Adversarial Suffix Filtering: a Defense Pipeline for LLMs**

cs.LG

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09602v1) [paper-pdf](http://arxiv.org/pdf/2505.09602v1)

**Authors**: David Khachaturov, Robert Mullins

**Abstract**: Large Language Models (LLMs) are increasingly embedded in autonomous systems and public-facing environments, yet they remain susceptible to jailbreak vulnerabilities that may undermine their security and trustworthiness. Adversarial suffixes are considered to be the current state-of-the-art jailbreak, consistently outperforming simpler methods and frequently succeeding even in black-box settings. Existing defenses rely on access to the internal architecture of models limiting diverse deployment, increase memory and computation footprints dramatically, or can be bypassed with simple prompt engineering methods. We introduce $\textbf{Adversarial Suffix Filtering}$ (ASF), a lightweight novel model-agnostic defensive pipeline designed to protect LLMs against adversarial suffix attacks. ASF functions as an input preprocessor and sanitizer that detects and filters adversarially crafted suffixes in prompts, effectively neutralizing malicious injections. We demonstrate that ASF provides comprehensive defense capabilities across both black-box and white-box attack settings, reducing the attack efficacy of state-of-the-art adversarial suffix generation methods to below 4%, while only minimally affecting the target model's capabilities in non-adversarial scenarios.



## **9. I Know What You Said: Unveiling Hardware Cache Side-Channels in Local Large Language Model Inference**

cs.CR

Submitted for review in January 22, 2025

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.06738v2) [paper-pdf](http://arxiv.org/pdf/2505.06738v2)

**Authors**: Zibo Gao, Junjie Hu, Feng Guo, Yixin Zhang, Yinglong Han, Siyuan Liu, Haiyang Li, Zhiqiang Lv

**Abstract**: Large Language Models (LLMs) that can be deployed locally have recently gained popularity for privacy-sensitive tasks, with companies such as Meta, Google, and Intel playing significant roles in their development. However, the security of local LLMs through the lens of hardware cache side-channels remains unexplored. In this paper, we unveil novel side-channel vulnerabilities in local LLM inference: token value and token position leakage, which can expose both the victim's input and output text, thereby compromising user privacy. Specifically, we found that adversaries can infer the token values from the cache access patterns of the token embedding operation, and deduce the token positions from the timing of autoregressive decoding phases. To demonstrate the potential of these leaks, we design a novel eavesdropping attack framework targeting both open-source and proprietary LLM inference systems. The attack framework does not directly interact with the victim's LLM and can be executed without privilege.   We evaluate the attack on a range of practical local LLM deployments (e.g., Llama, Falcon, and Gemma), and the results show that our attack achieves promising accuracy. The restored output and input text have an average edit distance of 5.2% and 17.3% to the ground truth, respectively. Furthermore, the reconstructed texts achieve average cosine similarity scores of 98.7% (input) and 98.0% (output).



## **10. FaceShield: Explainable Face Anti-Spoofing with Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09415v1) [paper-pdf](http://arxiv.org/pdf/2505.09415v1)

**Authors**: Hongyang Wang, Yichen Shi, Zhuofu Tao, Yuhao Gao, Liepiao Zhang, Xun Lin, Jun Feng, Xiaochen Yuan, Zitong Yu, Xiaochun Cao

**Abstract**: Face anti-spoofing (FAS) is crucial for protecting facial recognition systems from presentation attacks. Previous methods approached this task as a classification problem, lacking interpretability and reasoning behind the predicted results. Recently, multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and decision-making in visual tasks. However, there is currently no universal and comprehensive MLLM and dataset specifically designed for FAS task. To address this gap, we propose FaceShield, a MLLM for FAS, along with the corresponding pre-training and supervised fine-tuning (SFT) datasets, FaceShield-pre10K and FaceShield-sft45K. FaceShield is capable of determining the authenticity of faces, identifying types of spoofing attacks, providing reasoning for its judgments, and detecting attack areas. Specifically, we employ spoof-aware vision perception (SAVP) that incorporates both the original image and auxiliary information based on prior knowledge. We then use an prompt-guided vision token masking (PVTM) strategy to random mask vision tokens, thereby improving the model's generalization ability. We conducted extensive experiments on three benchmark datasets, demonstrating that FaceShield significantly outperforms previous deep learning models and general MLLMs on four FAS tasks, i.e., coarse-grained classification, fine-grained classification, reasoning, and attack localization. Our instruction datasets, protocols, and codes will be released soon.



## **11. What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks**

cs.CR

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2411.03343v2) [paper-pdf](http://arxiv.org/pdf/2411.03343v2)

**Authors**: Nathalie Kirch, Constantin Weisser, Severin Field, Helen Yannakoudakis, Stephen Casper

**Abstract**: Jailbreaks have been a central focus of research regarding the safety and reliability of large language models (LLMs), yet the mechanisms underlying these attacks remain poorly understood. While previous studies have predominantly relied on linear methods to detect jailbreak attempts and model refusals, we take a different approach by examining both linear and non-linear features in prompts that lead to successful jailbreaks. First, we introduce a novel dataset comprising 10,800 jailbreak attempts spanning 35 diverse attack methods. Leveraging this dataset, we train probes to classify successful from unsuccessful jailbreaks using the latent representations corresponding to prompt tokens. Notably, we find that even when probes achieve high accuracy in predicting the success of jailbreaks, their performance often fails to generalize to unseen attack methods. This reveals that different jailbreaking strategies exploit different non-linear, non-universal features. Next, we demonstrate that non-linear probes provide a powerful tool for steering model behavior. Specifically, we use these probes to guide targeted latent space perturbations, enabling us to effectively modulate the model's robustness against jailbreaks. Overall, our findings challenge the assumption that jailbreaks can be fully understood through linear or simple universal prompt features alone, highlighting the importance of a nuanced understanding of the mechanisms behind LLM vulnerabilities.



## **12. Improving Network Threat Detection by Knowledge Graph, Large Language Model, and Imbalanced Learning**

cs.LG

Accepted by "Combining AI and OR/MS for Better Trustworthy Decision  Making" Bridge Program co-organized by AAAI and INFORMS as poster and demo

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2501.16393v2) [paper-pdf](http://arxiv.org/pdf/2501.16393v2)

**Authors**: Lili Zhang, Quanyan Zhu, Herman Ray, Ying Xie

**Abstract**: Network threat detection has been challenging due to the complexities of attack activities and the limitation of historical threat data to learn from. To help enhance the existing practices of using analytics, machine learning, and artificial intelligence methods to detect the network threats, we propose an integrated modelling framework, where Knowledge Graph is used to analyze the users' activity patterns, Imbalanced Learning techniques are used to prune and weigh Knowledge Graph, and LLM is used to retrieve and interpret the users' activities from Knowledge Graph. The proposed framework is applied to Agile Threat Detection through Online Sequential Learning. The preliminary results show the improved threat capture rate by 3%-4% and the increased interpretabilities of risk predictions based on the users' activities.



## **13. Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction**

cs.CL

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.05084v2) [paper-pdf](http://arxiv.org/pdf/2505.05084v2)

**Authors**: Xiaowei Zhu, Yubing Ren, Yanan Cao, Xixun Lin, Fang Fang, Yangxi Li

**Abstract**: The rapid advancement of large language models has raised significant concerns regarding their potential misuse by malicious actors. As a result, developing effective detectors to mitigate these risks has become a critical priority. However, most existing detection methods focus excessively on detection accuracy, often neglecting the societal risks posed by high false positive rates (FPRs). This paper addresses this issue by leveraging Conformal Prediction (CP), which effectively constrains the upper bound of FPRs. While directly applying CP constrains FPRs, it also leads to a significant reduction in detection performance. To overcome this trade-off, this paper proposes a Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction (MCP), which both enforces the FPR constraint and improves detection performance. This paper also introduces RealDet, a high-quality dataset that spans a wide range of domains, ensuring realistic calibration and enabling superior detection performance when combined with MCP. Empirical evaluations demonstrate that MCP effectively constrains FPRs, significantly enhances detection performance, and increases robustness against adversarial attacks across multiple detectors and datasets.



## **14. Securing RAG: A Risk Assessment and Mitigation Framework**

cs.CR

8 pages, 3 figures, Sara Ott and Lukas Ammann contributed equally

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08728v1) [paper-pdf](http://arxiv.org/pdf/2505.08728v1)

**Authors**: Lukas Ammann, Sara Ott, Christoph R. Landolt, Marco P. Lehmann

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as the de facto industry standard for user-facing NLP applications, offering the ability to integrate data without re-training or fine-tuning Large Language Models (LLMs). This capability enhances the quality and accuracy of responses but also introduces novel security and privacy challenges, particularly when sensitive data is integrated. With the rapid adoption of RAG, securing data and services has become a critical priority. This paper first reviews the vulnerabilities of RAG pipelines, and outlines the attack surface from data pre-processing and data storage management to integration with LLMs. The identified risks are then paired with corresponding mitigations in a structured overview. In a second step, the paper develops a framework that combines RAG-specific security considerations, with existing general security guidelines, industry standards, and best practices. The proposed framework aims to guide the implementation of robust, compliant, secure, and trustworthy RAG systems.



## **15. Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs**

cs.CR

7 Pages, 6 Figures

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.04806v2) [paper-pdf](http://arxiv.org/pdf/2505.04806v2)

**Authors**: Chetan Pathade

**Abstract**: Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.



## **16. LM-Scout: Analyzing the Security of Language Model Integration in Android Apps**

cs.CR

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08204v1) [paper-pdf](http://arxiv.org/pdf/2505.08204v1)

**Authors**: Muhammad Ibrahim, Gűliz Seray Tuncay, Z. Berkay Celik, Aravind Machiry, Antonio Bianchi

**Abstract**: Developers are increasingly integrating Language Models (LMs) into their mobile apps to provide features such as chat-based assistants. To prevent LM misuse, they impose various restrictions, including limits on the number of queries, input length, and allowed topics. However, if the LM integration is insecure, attackers can bypass these restrictions and gain unrestricted access to the LM, potentially harming developers' reputations and leading to significant financial losses.   This paper presents the first systematic study of insecure usage of LMs by Android apps. We first manually analyze a preliminary dataset of apps to investigate LM integration methods, construct a taxonomy that categorizes the LM usage restrictions implemented by the apps, and determine how to bypass them. Alarmingly, we can bypass restrictions in 127 out of 181 apps. Then, we develop LM-Scout, a fully automated tool to detect on a large-scale vulnerable usage of LMs in 2,950 mobile apps. LM-Scout shows that, in many cases (i.e., 120 apps), it is possible to find and exploit such security issues automatically. Finally, we identify the root causes for the identified issues and offer recommendations for secure LM integration.



## **17. A Large-Scale Empirical Analysis of Custom GPTs' Vulnerabilities in the OpenAI Ecosystem**

cs.CR

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08148v1) [paper-pdf](http://arxiv.org/pdf/2505.08148v1)

**Authors**: Sunday Oyinlola Ogundoyin, Muhammad Ikram, Hassan Jameel Asghar, Benjamin Zi Hao Zhao, Dali Kaafar

**Abstract**: Millions of users leverage generative pretrained transformer (GPT)-based language models developed by leading model providers for a wide range of tasks. To support enhanced user interaction and customization, many platforms-such as OpenAI-now enable developers to create and publish tailored model instances, known as custom GPTs, via dedicated repositories or application stores. These custom GPTs empower users to browse and interact with specialized applications designed to meet specific needs. However, as custom GPTs see growing adoption, concerns regarding their security vulnerabilities have intensified. Existing research on these vulnerabilities remains largely theoretical, often lacking empirical, large-scale, and statistically rigorous assessments of associated risks.   In this study, we analyze 14,904 custom GPTs to assess their susceptibility to seven exploitable threats, such as roleplay-based attacks, system prompt leakage, phishing content generation, and malicious code synthesis, across various categories and popularity tiers within the OpenAI marketplace. We introduce a multi-metric ranking system to examine the relationship between a custom GPT's popularity and its associated security risks.   Our findings reveal that over 95% of custom GPTs lack adequate security protections. The most prevalent vulnerabilities include roleplay-based vulnerabilities (96.51%), system prompt leakage (92.20%), and phishing (91.22%). Furthermore, we demonstrate that OpenAI's foundational models exhibit inherent security weaknesses, which are often inherited or amplified in custom GPTs. These results highlight the urgent need for enhanced security measures and stricter content moderation to ensure the safe deployment of GPT-based applications.



## **18. LiteLMGuard: Seamless and Lightweight On-Device Prompt Filtering for Safeguarding Small Language Models against Quantization-induced Risks and Vulnerabilities**

cs.CR

14 pages, 18 figures, and 4 tables

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.05619v2) [paper-pdf](http://arxiv.org/pdf/2505.05619v2)

**Authors**: Kalyan Nakka, Jimmy Dani, Ausmit Mondal, Nitesh Saxena

**Abstract**: The growing adoption of Large Language Models (LLMs) has influenced the development of their lighter counterparts-Small Language Models (SLMs)-to enable on-device deployment across smartphones and edge devices. These SLMs offer enhanced privacy, reduced latency, server-free functionality, and improved user experience. However, due to resource constraints of on-device environment, SLMs undergo size optimization through compression techniques like quantization, which can inadvertently introduce fairness, ethical and privacy risks. Critically, quantized SLMs may respond to harmful queries directly, without requiring adversarial manipulation, raising significant safety and trust concerns.   To address this, we propose LiteLMGuard (LLMG), an on-device prompt guard that provides real-time, prompt-level defense for quantized SLMs. Additionally, our prompt guard is designed to be model-agnostic such that it can be seamlessly integrated with any SLM, operating independently of underlying architectures. Our LLMG formalizes prompt filtering as a deep learning (DL)-based prompt answerability classification task, leveraging semantic understanding to determine whether a query should be answered by any SLM. Using our curated dataset, Answerable-or-Not, we trained and fine-tuned several DL models and selected ELECTRA as the candidate, with 97.75% answerability classification accuracy.   Our safety effectiveness evaluations demonstrate that LLMG defends against over 87% of harmful prompts, including both direct instruction and jailbreak attack strategies. We further showcase its ability to mitigate the Open Knowledge Attacks, where compromised SLMs provide unsafe responses without adversarial prompting. In terms of prompt filtering effectiveness, LLMG achieves near state-of-the-art filtering accuracy of 94%, with an average latency of 135 ms, incurring negligible overhead for users.



## **19. SCA: Improve Semantic Consistent in Unrestricted Adversarial Attacks via DDPM Inversion**

cs.CV

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2410.02240v6) [paper-pdf](http://arxiv.org/pdf/2410.02240v6)

**Authors**: Zihao Pan, Lifeng Chen, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Systems based on deep neural networks are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often result in substantial semantic distortions in the denoised output and suffer from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes a Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our code can be found at https://github.com/Pan-Zihao/SCA.



## **20. Concept-Level Explainability for Auditing & Steering LLM Responses**

cs.CL

9 pages, 7 figures, Submission to Neurips 2025

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07610v1) [paper-pdf](http://arxiv.org/pdf/2505.07610v1)

**Authors**: Kenza Amara, Rita Sevastjanova, Mennatallah El-Assady

**Abstract**: As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior.



## **21. SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models**

cs.CR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07584v1) [paper-pdf](http://arxiv.org/pdf/2505.07584v1)

**Authors**: Huining Cui, Wei Liu

**Abstract**: The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security.



## **22. SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models**

cs.CV

Accepted at CVPR 2025 Workshop EVAL-FoMo-2

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2504.04893v3) [paper-pdf](http://arxiv.org/pdf/2504.04893v3)

**Authors**: Justus Westerhoff, Erblina Purelku, Jakob Hackstein, Jonas Loos, Leo Pinetzki, Lorenz Hufe

**Abstract**: Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper along with the code for evaluations at www.bliss.berlin/research/scam.



## **23. GRADA: Graph-based Reranker against Adversarial Documents Attack**

cs.IR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07546v1) [paper-pdf](http://arxiv.org/pdf/2505.07546v1)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.



## **24. No Query, No Access**

cs.CL

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07258v1) [paper-pdf](http://arxiv.org/pdf/2505.07258v1)

**Authors**: Wenqiang Wang, Siyuan Liang, Yangshijie Zhang, Xiaojun Jia, Hao Lin, Xiaochun Cao

**Abstract**: Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary.   Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at https://anonymous.4open.science/r/VDBA-Victim-Data-based-Adversarial-Attack-36EC/



## **25. One Trigger Token Is Enough: A Defense Strategy for Balancing Safety and Usability in Large Language Models**

cs.CR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07167v1) [paper-pdf](http://arxiv.org/pdf/2505.07167v1)

**Authors**: Haoran Gu, Handing Wang, Yi Mei, Mengjie Zhang, Yaochu Jin

**Abstract**: Large Language Models (LLMs) have been extensively used across diverse domains, including virtual assistants, automated code generation, and scientific research. However, they remain vulnerable to jailbreak attacks, which manipulate the models into generating harmful responses despite safety alignment. Recent studies have shown that current safety-aligned LLMs often undergo the shallow safety alignment, where the first few tokens largely determine whether the response will be harmful. Through comprehensive observations, we find that safety-aligned LLMs and various defense strategies generate highly similar initial tokens in their refusal responses, which we define as safety trigger tokens. Building on this insight, we propose \texttt{D-STT}, a simple yet effective defense algorithm that identifies and explicitly decodes safety trigger tokens of the given safety-aligned LLM to trigger the model's learned safety patterns. In this process, the safety trigger is constrained to a single token, which effectively preserves model usability by introducing minimum intervention in the decoding process. Extensive experiments across diverse jailbreak attacks and benign prompts demonstrate that \ours significantly reduces output harmfulness while preserving model usability and incurring negligible response time overhead, outperforming ten baseline methods.



## **26. Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks**

cs.LG

ICML 2025 Accpeted

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.05190v2) [paper-pdf](http://arxiv.org/pdf/2505.05190v2)

**Authors**: Yixin Cheng, Hongcheng Guo, Yangming Li, Leonid Sigal

**Abstract**: Text watermarking aims to subtly embed statistical signals into text by controlling the Large Language Model (LLM)'s sampling process, enabling watermark detectors to verify that the output was generated by the specified model. The robustness of these watermarking algorithms has become a key factor in evaluating their effectiveness. Current text watermarking algorithms embed watermarks in high-entropy tokens to ensure text quality. In this paper, we reveal that this seemingly benign design can be exploited by attackers, posing a significant risk to the robustness of the watermark. We introduce a generic efficient paraphrasing attack, the Self-Information Rewrite Attack (SIRA), which leverages the vulnerability by calculating the self-information of each token to identify potential pattern tokens and perform targeted attack. Our work exposes a widely prevalent vulnerability in current watermarking algorithms. The experimental results show SIRA achieves nearly 100% attack success rates on seven recent watermarking methods with only 0.88 USD per million tokens cost. Our approach does not require any access to the watermark algorithms or the watermarked LLM and can seamlessly transfer to any LLM as the attack model, even mobile-level models. Our findings highlight the urgent need for more robust watermarking.



## **27. Unleashing the potential of prompt engineering for large language models**

cs.CL

v6 - Metadata updated (title, journal ref, DOI). PDF identical to v5  (original submission). Please cite the peer-reviewed Version of Record in  "Patterns" (DOI: 10.1016/j.patter.2025.101260)

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2310.14735v6) [paper-pdf](http://arxiv.org/pdf/2310.14735v6)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). The development of Artificial Intelligence (AI), from its inception in the 1950s to the emergence of advanced neural networks and deep learning architectures, has made a breakthrough in LLMs, with models such as GPT-4o and Claude-3, and in Vision-Language Models (VLMs), with models such as CLIP and ALIGN. Prompt engineering is the process of structuring inputs, which has emerged as a crucial technique to maximize the utility and accuracy of these models. This paper explores both foundational and advanced methodologies of prompt engineering, including techniques such as self-consistency, chain-of-thought, and generated knowledge, which significantly enhance model performance. Additionally, it examines the prompt method of VLMs through innovative approaches such as Context Optimization (CoOp), Conditional Context Optimization (CoCoOp), and Multimodal Prompt Learning (MaPLe). Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is also addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review also reflects the essential role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.



## **28. IM-BERT: Enhancing Robustness of BERT through the Implicit Euler Method**

cs.CL

Accepted to EMNLP 2024 Main

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.06889v1) [paper-pdf](http://arxiv.org/pdf/2505.06889v1)

**Authors**: Mihyeon Kim, Juhyoung Park, Youngbin Kim

**Abstract**: Pre-trained Language Models (PLMs) have achieved remarkable performance on diverse NLP tasks through pre-training and fine-tuning. However, fine-tuning the model with a large number of parameters on limited downstream datasets often leads to vulnerability to adversarial attacks, causing overfitting of the model on standard datasets.   To address these issues, we propose IM-BERT from the perspective of a dynamic system by conceptualizing a layer of BERT as a solution of Ordinary Differential Equations (ODEs). Under the situation of initial value perturbation, we analyze the numerical stability of two main numerical ODE solvers: the explicit and implicit Euler approaches.   Based on these analyses, we introduce a numerically robust IM-connection incorporating BERT's layers. This strategy enhances the robustness of PLMs against adversarial attacks, even in low-resource scenarios, without introducing additional parameters or adversarial training strategies.   Experimental results on the adversarial GLUE (AdvGLUE) dataset validate the robustness of IM-BERT under various conditions. Compared to the original BERT, IM-BERT exhibits a performance improvement of approximately 8.3\%p on the AdvGLUE dataset. Furthermore, in low-resource scenarios, IM-BERT outperforms BERT by achieving 5.9\%p higher accuracy.



## **29. Benign Samples Matter! Fine-tuning On Outlier Benign Samples Severely Breaks Safety**

cs.LG

26 pages, 13 figures

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.06843v1) [paper-pdf](http://arxiv.org/pdf/2505.06843v1)

**Authors**: Zihan Guan, Mengxuan Hu, Ronghang Zhu, Sheng Li, Anil Vullikanti

**Abstract**: Recent studies have uncovered a troubling vulnerability in the fine-tuning stage of large language models (LLMs): even fine-tuning on entirely benign datasets can lead to a significant increase in the harmfulness of LLM outputs. Building on this finding, our red teaming study takes this threat one step further by developing a more effective attack. Specifically, we analyze and identify samples within benign datasets that contribute most to safety degradation, then fine-tune LLMs exclusively on these samples. We approach this problem from an outlier detection perspective and propose Self-Inf-N, to detect and extract outliers for fine-tuning. Our findings reveal that fine-tuning LLMs on 100 outlier samples selected by Self-Inf-N in the benign datasets severely compromises LLM safety alignment. Extensive experiments across seven mainstream LLMs demonstrate that our attack exhibits high transferability across different architectures and remains effective in practical scenarios. Alarmingly, our results indicate that most existing mitigation strategies fail to defend against this attack, underscoring the urgent need for more robust alignment safeguards. Codes are available at https://github.com/GuanZihan/Benign-Samples-Matter.



## **30. Diversity Helps Jailbreak Large Language Models**

cs.CL

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2411.04223v3) [paper-pdf](http://arxiv.org/pdf/2411.04223v3)

**Authors**: Weiliang Zhao, Daniel Ben-Levi, Wei Hao, Junfeng Yang, Chengzhi Mao

**Abstract**: We have uncovered a powerful jailbreak technique that leverages large language models' ability to diverge from prior context, enabling them to bypass safety constraints and generate harmful outputs. By simply instructing the LLM to deviate and obfuscate previous attacks, our method dramatically outperforms existing approaches, achieving up to a 62.83% higher success rate in compromising ten leading chatbots, including GPT-4, Gemini, and Llama, while using only 12.9% of the queries. This revelation exposes a critical flaw in current LLM safety training, suggesting that existing methods may merely mask vulnerabilities rather than eliminate them. Our findings sound an urgent alarm for the need to revolutionize testing methodologies to ensure robust and reliable LLM security.



## **31. Practical Reasoning Interruption Attacks on Reasoning Large Language Models**

cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06643v1) [paper-pdf](http://arxiv.org/pdf/2505.06643v1)

**Authors**: Yu Cui, Cong Zuo

**Abstract**: Reasoning large language models (RLLMs) have demonstrated outstanding performance across a variety of tasks, yet they also expose numerous security vulnerabilities. Most of these vulnerabilities have centered on the generation of unsafe content. However, recent work has identified a distinct "thinking-stopped" vulnerability in DeepSeek-R1: under adversarial prompts, the model's reasoning process ceases at the system level and produces an empty final answer. Building upon this vulnerability, researchers developed a novel prompt injection attack, termed reasoning interruption attack, and also offered an initial analysis of its root cause. Through extensive experiments, we verify the previous analyses, correct key errors based on three experimental findings, and present a more rigorous explanation of the fundamental causes driving the vulnerability. Moreover, existing attacks typically require over 2,000 tokens, impose significant overhead, reduce practicality, and are easily detected. To overcome these limitations, we propose the first practical reasoning interruption attack. It succeeds with just 109 tokens by exploiting our newly uncovered "reasoning token overflow" (RTO) effect to overwrite the model's final answer, forcing it to return an invalid response. Experimental results demonstrate that our proposed attack is highly effective. Furthermore, we discover that the method for triggering RTO differs between the official DeepSeek-R1 release and common unofficial deployments. As a broadened application of RTO, we also construct a novel jailbreak attack that enables the transfer of unsafe content within the reasoning tokens into final answer, thereby exposing it to the user. Our work carries significant implications for enhancing the security of RLLMs.



## **32. POISONCRAFT: Practical Poisoning of Retrieval-Augmented Generation for Large Language Models**

cs.CR

12 pages, 7 tables and 3 figures

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06579v1) [paper-pdf](http://arxiv.org/pdf/2505.06579v1)

**Authors**: Yangguang Shao, Xinjie Lin, Haozheng Luo, Chengshang Hou, Gang Xiong, Jiahao Yu, Junzheng Shi

**Abstract**: Large language models (LLMs) have achieved remarkable success in various domains, primarily due to their strong capabilities in reasoning and generating human-like text. Despite their impressive performance, LLMs are susceptible to hallucinations, which can lead to incorrect or misleading outputs. This is primarily due to the lack of up-to-date knowledge or domain-specific information. Retrieval-augmented generation (RAG) is a promising approach to mitigate hallucinations by leveraging external knowledge sources. However, the security of RAG systems has not been thoroughly studied. In this paper, we study a poisoning attack on RAG systems named POISONCRAFT, which can mislead the model to refer to fraudulent websites. Compared to existing poisoning attacks on RAG systems, our attack is more practical as it does not require access to the target user query's info or edit the user query. It not only ensures that injected texts can be retrieved by the model, but also ensures that the LLM will be misled to refer to the injected texts in its response. We demonstrate the effectiveness of POISONCRAFTacross different datasets, retrievers, and language models in RAG pipelines, and show that it remains effective when transferred across retrievers, including black-box systems. Moreover, we present a case study revealing how the attack influences both the retrieval behavior and the step-by-step reasoning trace within the generation model, and further evaluate the robustness of POISONCRAFTunder multiple defense mechanisms. These results validate the practicality of our threat model and highlight a critical security risk for RAG systems deployed in real-world applications. We release our code\footnote{https://github.com/AndyShaw01/PoisonCraft} to support future research on the security and robustness of RAG systems in real-world settings.



## **33. Fun-tuning: Characterizing the Vulnerability of Proprietary LLMs to Optimization-based Prompt Injection Attacks via the Fine-Tuning Interface**

cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2501.09798v2) [paper-pdf](http://arxiv.org/pdf/2501.09798v2)

**Authors**: Andrey Labunets, Nishit V. Pandya, Ashish Hooda, Xiaohan Fu, Earlence Fernandes

**Abstract**: We surface a new threat to closed-weight Large Language Models (LLMs) that enables an attacker to compute optimization-based prompt injections. Specifically, we characterize how an attacker can leverage the loss-like information returned from the remote fine-tuning interface to guide the search for adversarial prompts. The fine-tuning interface is hosted by an LLM vendor and allows developers to fine-tune LLMs for their tasks, thus providing utility, but also exposes enough information for an attacker to compute adversarial prompts. Through an experimental analysis, we characterize the loss-like values returned by the Gemini fine-tuning API and demonstrate that they provide a useful signal for discrete optimization of adversarial prompts using a greedy search algorithm. Using the PurpleLlama prompt injection benchmark, we demonstrate attack success rates between 65% and 82% on Google's Gemini family of LLMs. These attacks exploit the classic utility-security tradeoff - the fine-tuning interface provides a useful feature for developers but also exposes the LLMs to powerful attacks.



## **34. System Prompt Poisoning: Persistent Attacks on Large Language Models Beyond User Injection**

cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06493v1) [paper-pdf](http://arxiv.org/pdf/2505.06493v1)

**Authors**: Jiawei Guo, Haipeng Cai

**Abstract**: Large language models (LLMs) have gained widespread adoption across diverse applications due to their impressive generative capabilities. Their plug-and-play nature enables both developers and end users to interact with these models through simple prompts. However, as LLMs become more integrated into various systems in diverse domains, concerns around their security are growing. Existing studies mainly focus on threats arising from user prompts (e.g. prompt injection attack) and model output (e.g. model inversion attack), while the security of system prompts remains largely overlooked. This work bridges the critical gap. We introduce system prompt poisoning, a new attack vector against LLMs that, unlike traditional user prompt injection, poisons system prompts hence persistently impacts all subsequent user interactions and model responses. We systematically investigate four practical attack strategies in various poisoning scenarios. Through demonstration on both generative and reasoning LLMs, we show that system prompt poisoning is highly feasible without requiring jailbreak techniques, and effective across a wide range of tasks, including those in mathematics, coding, logical reasoning, and natural language processing. Importantly, our findings reveal that the attack remains effective even when user prompts employ advanced prompting techniques like chain-of-thought (CoT). We also show that such techniques, including CoT and retrieval-augmentation-generation (RAG), which are proven to be effective for improving LLM performance in a wide range of tasks, are significantly weakened in their effectiveness by system prompt poisoning.



## **35. Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions**

cs.CL

This paper is accepted by NAACL 2025 findings. Link to the paper  presentation: https://youtu.be/IhaxwbZOcaU

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2410.18966v3) [paper-pdf](http://arxiv.org/pdf/2410.18966v3)

**Authors**: Yujuan Fu, Ozlem Uzuner, Meliha Yetisgen, Fei Xia

**Abstract**: Large language models (LLMs) have demonstrated great performance across various benchmarks, showing potential as general-purpose task solvers. However, as LLMs are typically trained on vast amounts of data, a significant concern in their evaluation is data contamination, where overlap between training data and evaluation datasets inflates performance assessments. Multiple approaches have been developed to identify data contamination. These approaches rely on specific assumptions that may not hold universally across different settings. To bridge this gap, we systematically review 50 papers on data contamination detection, categorize the underlying assumptions, and assess whether they have been rigorously validated. We identify and analyze eight categories of assumptions and test three of them as case studies. Our case studies focus on detecting direct, instance-level data contamination, which is also referred to as Membership Inference Attacks (MIA). Our analysis reveals that MIA approaches based on these three assumptions can have similar performance to random guessing, on datasets used in LLM pretraining, suggesting that current LLMs might learn data distributions rather than memorizing individual instances. Meanwhile, MIA can easily fail when there are data distribution shifts between the seen and unseen instances.



## **36. LATENT: LLM-Augmented Trojan Insertion and Evaluation Framework for Analog Netlist Topologies**

cs.CR

Accepted for presentation at IEEE International Conference on  LLM-Aided Design (ICLAD), 2025

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.06364v1) [paper-pdf](http://arxiv.org/pdf/2505.06364v1)

**Authors**: Jayeeta Chaudhuri, Arjun Chaudhuri, Krishnendu Chakrabarty

**Abstract**: Analog and mixed-signal (A/MS) integrated circuits (ICs) are integral to safety-critical applications. However, the globalization and outsourcing of A/MS ICs to untrusted third-party foundries expose them to security threats, particularly analog Trojans. Unlike digital Trojans which have been extensively studied, analog Trojans remain largely unexplored. There has been only limited research on their diversity and stealth in analog designs, where a Trojan is activated only during a narrow input voltage range. Effective defense techniques require a clear understanding of the attack vectors; however, the lack of diverse analog Trojan instances limits robust advances in detection strategies. To address this gap, we present LATENT, the first large language model (LLM)-driven framework for crafting stealthy, circuit-specific analog Trojans. LATENT incorporates LLM as an autonomous agent to intelligently insert and refine Trojan components within analog designs based on iterative feedback from a detection model. This feedback loop ensures that the inserted Trojans remain stealthy while successfully evading detection. Experimental results demonstrate that our generated Trojan designs exhibit an average Trojan-activation range of 15.74%, ensuring they remain inactive under most operating voltages, while causing a significant performance degradation of 11.3% upon activation.



## **37. AgentXploit: End-to-End Redteaming of Black-Box AI Agents**

cs.CR

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.05849v1) [paper-pdf](http://arxiv.org/pdf/2505.05849v1)

**Authors**: Zhun Wang, Vincent Siu, Zhe Ye, Tianneng Shi, Yuzhou Nie, Xuandong Zhao, Chenguang Wang, Wenbo Guo, Dawn Song

**Abstract**: The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentXploit, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentXploit on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentXploit exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.



## **38. Unified Attacks to Large Language Model Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation**

cs.CL

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2504.17480v3) [paper-pdf](http://arxiv.org/pdf/2504.17480v3)

**Authors**: Xin Yi, Yue Li, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.



## **39. Towards the Worst-case Robustness of Large Language Models**

cs.LG

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2501.19040v2) [paper-pdf](http://arxiv.org/pdf/2501.19040v2)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of large language models to adversarial attacks, where adversaries craft specific input sequences to induce harmful, violent, private, or incorrect outputs. In this work, we study their worst-case robustness, i.e., whether an adversarial example exists that leads to such undesirable outputs. We upper bound the worst-case robustness using stronger white-box attacks, indicating that most current deterministic defenses achieve nearly 0\% worst-case robustness. We propose a general tight lower bound for randomized smoothing using fractional knapsack solvers or 0-1 knapsack solvers, and using them to bound the worst-case robustness of all stochastic defenses. Based on these solvers, we provide theoretical lower bounds for several previous empirical defenses. For example, we certify the robustness of a specific case, smoothing using a uniform kernel, against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.



## **40. Jailbreaking and Mitigation of Vulnerabilities in Large Language Models**

cs.CR

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2410.15236v2) [paper-pdf](http://arxiv.org/pdf/2410.15236v2)

**Authors**: Benji Peng, Keyu Chen, Qian Niu, Ziqian Bi, Ming Liu, Pohsun Feng, Tianyang Wang, Lawrence K. Q. Yan, Yizhu Wen, Yichao Zhang, Caitlyn Heqi Yin

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence by advancing natural language understanding and generation, enabling applications across fields beyond healthcare, software engineering, and conversational systems. Despite these advancements in the past few years, LLMs have shown considerable vulnerabilities, particularly to prompt injection and jailbreaking attacks. This review analyzes the state of research on these vulnerabilities and presents available defense strategies. We roughly categorize attack approaches into prompt-based, model-based, multimodal, and multilingual, covering techniques such as adversarial prompting, backdoor injections, and cross-modality exploits. We also review various defense mechanisms, including prompt filtering, transformation, alignment techniques, multi-agent defenses, and self-regulation, evaluating their strengths and shortcomings. We also discuss key metrics and benchmarks used to assess LLM safety and robustness, noting challenges like the quantification of attack success in interactive contexts and biases in existing datasets. Identifying current research gaps, we suggest future directions for resilient alignment strategies, advanced defenses against evolving attacks, automation of jailbreak detection, and consideration of ethical and societal impacts. This review emphasizes the need for continued research and cooperation within the AI community to enhance LLM security and ensure their safe deployment.



## **41. Defending against Indirect Prompt Injection by Instruction Detection**

cs.CR

13 pages, 4 figures

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.06311v1) [paper-pdf](http://arxiv.org/pdf/2505.06311v1)

**Authors**: Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, Fangzhao Wu

**Abstract**: The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. We recognize that the success of IPI attacks fundamentally relies in the presence of instructions embedded within external content, which can alter the behavioral state of LLMs. Can effectively detecting such state changes help us defend against IPI attacks? In this paper, we propose a novel approach that takes external data as input and leverages the behavioral state of LLMs during both forward and backward propagation to detect potential IPI attacks. Specifically, we demonstrate that the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. By effectively combining these features, our approach achieves a detection accuracy of 99.60\% in the in-domain setting and 96.90\% in the out-of-domain setting, while reducing the attack success rate to just 0.12\% on the BIPIA benchmark.



## **42. Stealthy LLM-Driven Data Poisoning Attacks Against Embedding-Based Retrieval-Augmented Recommender Systems**

cs.IR

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05196v1) [paper-pdf](http://arxiv.org/pdf/2505.05196v1)

**Authors**: Fatemeh Nazary, Yashar Deldjoo, Tommaso Di Noia, Eugenio Di Sciascio

**Abstract**: We present a systematic study of provider-side data poisoning in retrieval-augmented recommender systems (RAG-based). By modifying only a small fraction of tokens within item descriptions -- for instance, adding emotional keywords or borrowing phrases from semantically related items -- an attacker can significantly promote or demote targeted items. We formalize these attacks under token-edit and semantic-similarity constraints, and we examine their effectiveness in both promotion (long-tail items) and demotion (short-head items) scenarios. Our experiments on MovieLens, using two large language model (LLM) retrieval modules, show that even subtle attacks shift final rankings and item exposures while eluding naive detection. The results underscore the vulnerability of RAG-based pipelines to small-scale metadata rewrites and emphasize the need for robust textual consistency checks and provenance tracking to thwart stealthy provider-side poisoning.



## **43. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

cs.CV

ICML 2025

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05528v1) [paper-pdf](http://arxiv.org/pdf/2505.05528v1)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.



## **44. Safeguard-by-Development: A Privacy-Enhanced Development Paradigm for Multi-Agent Collaboration Systems**

cs.CR

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04799v1) [paper-pdf](http://arxiv.org/pdf/2505.04799v1)

**Authors**: Jian Cui, Zichuan Li, Luyi Xing, Xiaojing Liao

**Abstract**: Multi-agent collaboration systems (MACS), powered by large language models (LLMs), solve complex problems efficiently by leveraging each agent's specialization and communication between agents. However, the inherent exchange of information between agents and their interaction with external environments, such as LLM, tools, and users, inevitably introduces significant risks of sensitive data leakage, including vulnerabilities to attacks like prompt injection and reconnaissance. Existing MACS fail to enable privacy controls, making it challenging to manage sensitive information securely. In this paper, we take the first step to address the MACS's data leakage threat at the system development level through a privacy-enhanced development paradigm, Maris. Maris enables rigorous message flow control within MACS by embedding reference monitors into key multi-agent conversation components. We implemented Maris as an integral part of AutoGen, a widely adopted open-source multi-agent development framework. Then, we evaluate Maris for its effectiveness and performance overhead on privacy-critical MACS use cases, including healthcare, supply chain optimization, and personalized recommendation system. The result shows that Maris achieves satisfactory effectiveness, performance overhead and practicability for adoption.



## **45. A Proposal for Evaluating the Operational Risk for ChatBots based on Large Language Models**

cs.CR

21 pages

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04784v1) [paper-pdf](http://arxiv.org/pdf/2505.04784v1)

**Authors**: Pedro Pinacho-Davidson, Fernando Gutierrez, Pablo Zapata, Rodolfo Vergara, Pablo Aqueveque

**Abstract**: The emergence of Generative AI (Gen AI) and Large Language Models (LLMs) has enabled more advanced chatbots capable of human-like interactions. However, these conversational agents introduce a broader set of operational risks that extend beyond traditional cybersecurity considerations. In this work, we propose a novel, instrumented risk-assessment metric that simultaneously evaluates potential threats to three key stakeholders: the service-providing organization, end users, and third parties. Our approach incorporates the technical complexity required to induce erroneous behaviors in the chatbot--ranging from non-induced failures to advanced prompt-injection attacks--as well as contextual factors such as the target industry, user age range, and vulnerability severity. To validate our metric, we leverage Garak, an open-source framework for LLM vulnerability testing. We further enhance Garak to capture a variety of threat vectors (e.g., misinformation, code hallucinations, social engineering, and malicious code generation). Our methodology is demonstrated in a scenario involving chatbots that employ retrieval-augmented generation (RAG), showing how the aggregated risk scores guide both short-term mitigation and longer-term improvements in model design and deployment. The results underscore the importance of multi-dimensional risk assessments in operationalizing secure, reliable AI-driven conversational systems.



## **46. ACE: A Security Architecture for LLM-Integrated App Systems**

cs.CR

21 pages, 13 figures; clarify relation to indirect prompt injection  attacks

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2504.20984v2) [paper-pdf](http://arxiv.org/pdf/2504.20984v2)

**Authors**: Evan Li, Tushin Mallick, Evan Rose, William Robertson, Alina Oprea, Cristina Nita-Rotaru

**Abstract**: LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution.   In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that our system is secure against attacks from the INJECAGENT benchmark, a standard benchmark for control flow integrity in the face of indirect prompt injection attacks, and our newly introduced attacks. Our architecture represents a significant advancement towards hardening LLM-based systems containing system facilities of varying levels of trustworthiness.



## **47. Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization**

cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04578v1) [paper-pdf](http://arxiv.org/pdf/2505.04578v1)

**Authors**: Wenjun Cao

**Abstract**: Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models.



## **48. An LLM-based Self-Evolving Security Framework for 6G Space-Air-Ground Integrated Networks**

cs.CR

Accepted by IEEE Communications Magazine

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.03161v2) [paper-pdf](http://arxiv.org/pdf/2505.03161v2)

**Authors**: Qi Qin, Xinye Cao, Guoshun Nan, Sihan Chen, Rushan Li, Li Su, Haitao Du, Qimei Cui, Pengxuan Mao, Xiaofeng Tao, Tony Q. S. Quek

**Abstract**: Recently emerged 6G space-air-ground integrated networks (SAGINs), which integrate satellites, aerial networks, and terrestrial communications, offer ubiquitous coverage for various mobile applications. However, the highly dynamic, open, and heterogeneous nature of SAGINs poses severe security issues. Forming a defense line of SAGINs suffers from two preliminary challenges: 1) accurately understanding massive unstructured multi-dimensional threat information to generate defense strategies against various malicious attacks, 2) rapidly adapting to potential unknown threats to yield more effective security strategies. To tackle the above two challenges, we propose a novel security framework for SAGINs based on Large Language Models (LLMs), which consists of two key ingredients LLM-6GNG and 6G-INST. Our proposed LLM-6GNG leverages refined chain-of-thought (CoT) reasoning and dynamic multi-agent mechanisms to analyze massive unstructured multi-dimensional threat data and generate comprehensive security strategies, thus addressing the first challenge. Our proposed 6G-INST relies on a novel self-evolving method to automatically update LLM-6GNG, enabling it to accommodate unknown threats under dynamic communication environments, thereby addressing the second challenge. Additionally, we prototype the proposed framework with ns-3, OpenAirInterface (OAI), and software-defined radio (SDR). Experiments on three benchmarks demonstrate the effectiveness of our framework. The results show that our framework produces highly accurate security strategies that remain robust against a variety of unknown attacks. We will release our code to contribute to the community.



## **49. OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models**

cs.CL

18 pages, 2 figures

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04416v1) [paper-pdf](http://arxiv.org/pdf/2505.04416v1)

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose OBLIVIATE, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA), it ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including the Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: forget quality (new document-level memorization score), model utility, and fluency. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios.



## **50. The Aloe Family Recipe for Open and Specialized Healthcare LLMs**

cs.CL

arXiv admin note: substantial text overlap with arXiv:2405.01886

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04388v1) [paper-pdf](http://arxiv.org/pdf/2505.04388v1)

**Authors**: Dario Garcia-Gasulla, Jordi Bayarri-Planas, Ashwin Kumar Gururajan, Enrique Lopez-Cuena, Adrian Tormos, Daniel Hinjos, Pablo Bernabeu-Perez, Anna Arias-Duart, Pablo Agustin Martin-Torres, Marta Gonzalez-Mallo, Sergio Alvarez-Napagao, Eduard Ayguadé-Parra, Ulises Cortés

**Abstract**: Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license.   Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results.   Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models.   Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare.



