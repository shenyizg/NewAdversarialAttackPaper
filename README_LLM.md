# Latest Large Language Model Attack Papers
**update at 2025-04-17 11:37:33**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails**

cs.CR

12 pages, 5 figures, 6 tables

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11168v2) [paper-pdf](http://arxiv.org/pdf/2504.11168v2)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.



## **2. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2402.09063v2) [paper-pdf](http://arxiv.org/pdf/2402.09063v2)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.



## **3. LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks**

cs.CL

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.10185v2) [paper-pdf](http://arxiv.org/pdf/2504.10185v2)

**Authors**: Soumyadeep Pal, Changsheng Wang, James Diffenderfer, Bhavya Kailkhura, Sijia Liu

**Abstract**: Large language model unlearning has become a critical challenge in ensuring safety and controlled model behavior by removing undesired data-model influences from the pretrained model while preserving general utility. Significant recent efforts have been dedicated to developing LLM unlearning benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine Unlearning Six-way Evaluation), facilitating standardized unlearning performance assessment and method comparison. Despite their usefulness, we uncover for the first time a novel coreset effect within these benchmarks. Specifically, we find that LLM unlearning achieved with the original (full) forget set can be effectively maintained using a significantly smaller subset (functioning as a "coreset"), e.g., as little as 5% of the forget set, even when selected at random. This suggests that LLM unlearning in these benchmarks can be performed surprisingly easily, even in an extremely low-data regime. We demonstrate that this coreset effect remains strong, regardless of the LLM unlearning method used, such as NPO (Negative Preference Optimization) and RMU (Representation Misdirection Unlearning), the popular ones in these benchmarks. The surprisingly strong coreset effect is also robust across various data selection methods, ranging from random selection to more sophisticated heuristic approaches. We explain the coreset effect in LLM unlearning through a keyword-based perspective, showing that keywords extracted from the forget set alone contribute significantly to unlearning effectiveness and indicating that current unlearning is driven by a compact set of high-impact tokens rather than the entire dataset. We further justify the faithfulness of coreset-unlearned models along additional dimensions, such as mode connectivity and robustness to jailbreaking attacks. Codes are available at https://github.com/OPTML-Group/MU-Coreset.



## **4. Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation**

cs.CL

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.12108v1) [paper-pdf](http://arxiv.org/pdf/2504.12108v1)

**Authors**: Shizhan Cai, Liang Ding, Dacheng Tao

**Abstract**: The rapid development of Large Language Models (LLMs) has intensified concerns about content traceability and potential misuse. Existing watermarking schemes for sampled text often face trade-offs between maintaining text quality and ensuring robust detection against various attacks. To address these issues, we propose a novel watermarking scheme that improves both detectability and text quality by introducing a cumulative watermark entropy threshold. Our approach is compatible with and generalizes existing sampling functions, enhancing adaptability. Experimental results across multiple LLMs show that our scheme significantly outperforms existing methods, achieving over 80\% improvements on widely-used datasets, e.g., MATH and GSM8K, while maintaining high detection accuracy.



## **5. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2410.02644v3) [paper-pdf](http://arxiv.org/pdf/2410.02644v3)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.



## **6. Progent: Programmable Privilege Control for LLM Agents**

cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11703v1) [paper-pdf](http://arxiv.org/pdf/2504.11703v1)

**Authors**: Tianneng Shi, Jingxuan He, Zhun Wang, Linyu Wu, Hongwei Li, Wenbo Guo, Dawn Song

**Abstract**: LLM agents are an emerging form of AI systems where large language models (LLMs) serve as the central component, utilizing a diverse set of tools to complete user-assigned tasks. Despite their great potential, LLM agents pose significant security risks. When interacting with the external world, they may encounter malicious commands from attackers, leading to the execution of dangerous actions. A promising way to address this is by enforcing the principle of least privilege: allowing only essential actions for task completion while blocking unnecessary ones. However, achieving this is challenging, as it requires covering diverse agent scenarios while preserving both security and utility.   We introduce Progent, the first privilege control mechanism for LLM agents. At its core is a domain-specific language for flexibly expressing privilege control policies applied during agent execution. These policies provide fine-grained constraints over tool calls, deciding when tool calls are permissible and specifying fallbacks if they are not. This enables agent developers and users to craft suitable policies for their specific use cases and enforce them deterministically to guarantee security. Thanks to its modular design, integrating Progent does not alter agent internals and requires only minimal changes to agent implementation, enhancing its practicality and potential for widespread adoption. To automate policy writing, we leverage LLMs to generate policies based on user queries, which are then updated dynamically for improved security and utility. Our extensive evaluation shows that it enables strong security while preserving high utility across three distinct scenarios or benchmarks: AgentDojo, ASB, and AgentPoison. Furthermore, we perform an in-depth analysis, showcasing the effectiveness of its core components and the resilience of its automated policy generation against adaptive attacks.



## **7. Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' "Typo" Correction**

cs.CR

Length: 13 pages Figures: 5 figures Tables: 7 tables Keywords:  Acoustic side-channel attacks, machine learning, Visual Transformers, Large  Language Models (LLMs), security Conference: Accepted at the 19th USENIX WOOT  Conference on Offensive Technologies (WOOT '25). Licensing: This paper is  submitted under the CC BY Creative Commons Attribution license. arXiv admin  note: text overlap with arXiv:2502.09782

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11622v1) [paper-pdf](http://arxiv.org/pdf/2504.11622v1)

**Authors**: Seyyed Ali Ayati, Jin Hyun Park, Yichen Cai, Marcus Botacin

**Abstract**: The large integration of microphones into devices increases the opportunities for Acoustic Side-Channel Attacks (ASCAs), as these can be used to capture keystrokes' audio signals that might reveal sensitive information. However, the current State-Of-The-Art (SOTA) models for ASCAs, including Convolutional Neural Networks (CNNs) and hybrid models, such as CoAtNet, still exhibit limited robustness under realistic noisy conditions. Solving this problem requires either: (i) an increased model's capacity to infer contextual information from longer sequences, allowing the model to learn that an initially noisily typed word is the same as a futurely collected non-noisy word, or (ii) an approach to fix misidentified information from the contexts, as one does not type random words, but the ones that best fit the conversation context. In this paper, we demonstrate that both strategies are viable and complementary solutions for making ASCAs practical. We observed that no existing solution leverages advanced transformer architectures' power for these tasks and propose that: (i) Visual Transformers (VTs) are the candidate solutions for capturing long-term contextual information and (ii) transformer-powered Large Language Models (LLMs) are the candidate solutions to fix the ``typos'' (mispredictions) the model might make. Thus, we here present the first-of-its-kind approach that integrates VTs and LLMs for ASCAs.   We first show that VTs achieve SOTA performance in classifying keystrokes when compared to the previous CNN benchmark. Second, we demonstrate that LLMs can mitigate the impact of real-world noise. Evaluations on the natural sentences revealed that: (i) incorporating LLMs (e.g., GPT-4o) in our ASCA pipeline boosts the performance of error-correction tasks; and (ii) the comparable performance can be attained by a lightweight, fine-tuned smaller LLM (67 times smaller than GPT-4o), using...



## **8. Lateral Phishing With Large Language Models: A Large Organization Comparative Study**

cs.CR

Accepted for publication in IEEE Access. This version includes  revisions following peer review

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2401.09727v2) [paper-pdf](http://arxiv.org/pdf/2401.09727v2)

**Authors**: Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nicole Beebe, Nishant Vishwamitra, Peyman Najafirad

**Abstract**: The emergence of Large Language Models (LLMs) has heightened the threat of phishing emails by enabling the generation of highly targeted, personalized, and automated attacks. Traditionally, many phishing emails have been characterized by typos, errors, and poor language. These errors can be mitigated by LLMs, potentially lowering the barrier for attackers. Despite this, there is a lack of large-scale studies comparing the effectiveness of LLM-generated lateral phishing emails to those crafted by humans. Current literature does not adequately address the comparative effectiveness of LLM and human-generated lateral phishing emails in a real-world, large-scale organizational setting, especially considering the potential for LLMs to generate more convincing and error-free phishing content. To address this gap, we conducted a pioneering study within a large university, targeting its workforce of approximately 9,000 individuals including faculty, staff, administrators, and student workers. Our results indicate that LLM-generated lateral phishing emails are as effective as those written by communications professionals, emphasizing the critical threat posed by LLMs in leading phishing campaigns. We break down the results of the overall phishing experiment, comparing vulnerability between departments and job roles. Furthermore, to gather qualitative data, we administered a detailed questionnaire, revealing insights into the reasons and motivations behind vulnerable employee's actions. This study contributes to the understanding of cyber security threats in educational institutions and provides a comprehensive comparison of LLM and human-generated phishing emails' effectiveness, considering the potential for LLMs to generate more convincing content. The findings highlight the need for enhanced user education and system defenses to mitigate the growing threat of AI-powered phishing attacks.



## **9. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2410.02240v5) [paper-pdf](http://arxiv.org/pdf/2410.02240v5)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.



## **10. The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections**

cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11281v1) [paper-pdf](http://arxiv.org/pdf/2504.11281v1)

**Authors**: Chaoran Chen, Zhiping Zhang, Bingcan Guo, Shang Ma, Ibrahim Khalilov, Simret A Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li

**Abstract**: A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.



## **11. Exploring Backdoor Attack and Defense for LLM-empowered Recommendations**

cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11182v1) [paper-pdf](http://arxiv.org/pdf/2504.11182v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner.



## **12. QAVA: Query-Agnostic Visual Attack to Large Vision-Language Models**

cs.CV

Accepted by NAACL 2025 main

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11038v1) [paper-pdf](http://arxiv.org/pdf/2504.11038v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Yu Wang

**Abstract**: In typical multimodal tasks, such as Visual Question Answering (VQA), adversarial attacks targeting a specific image and question can lead large vision-language models (LVLMs) to provide incorrect answers. However, it is common for a single image to be associated with multiple questions, and LVLMs may still answer other questions correctly even for an adversarial image attacked by a specific question. To address this, we introduce the query-agnostic visual attack (QAVA), which aims to create robust adversarial examples that generate incorrect responses to unspecified and unknown questions. Compared to traditional adversarial attacks focused on specific images and questions, QAVA significantly enhances the effectiveness and efficiency of attacks on images when the question is unknown, achieving performance comparable to attacks on known target questions. Our research broadens the scope of visual adversarial attacks on LVLMs in practical settings, uncovering previously overlooked vulnerabilities, particularly in the context of visual adversarial threats. The code is available at https://github.com/btzyd/qava.



## **13. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

cs.CR

7 pages; In submission

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2412.21051v2) [paper-pdf](http://arxiv.org/pdf/2412.21051v2)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Yuyu Zhao

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided a large number of benefits in daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks. Recent advancements in generative foundation models (GFMs), particularly in the large language models (LLMs), offer promising solutions for security intelligence. By exploiting the powerful abilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel proactive defense architecture that defeats various threats in a proactive manner. LLM-PD can efficiently make a decision through comprehensive data analysis and sequential reasoning, as well as dynamically creating and deploying actionable defense mechanisms on the target cloud. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. The experimental results demonstrate its remarkable ability in terms of defense effectiveness and efficiency, particularly highlighting an outstanding success rate when compared with other existing methods.



## **14. Adversarial Prompt Distillation for Vision-Language Models**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.15244v2) [paper-pdf](http://arxiv.org/pdf/2411.15244v2)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.



## **15. The Jailbreak Tax: How Useful are Your Jailbreak Outputs?**

cs.LG

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10694v1) [paper-pdf](http://arxiv.org/pdf/2504.10694v1)

**Authors**: Kristina Nikolić, Luze Sun, Jie Zhang, Florian Tramèr

**Abstract**: Jailbreak attacks bypass the guardrails of large language models to produce harmful outputs. In this paper, we ask whether the model outputs produced by existing jailbreaks are actually useful. For example, when jailbreaking a model to give instructions for building a bomb, does the jailbreak yield good instructions? Since the utility of most unsafe answers (e.g., bomb instructions) is hard to evaluate rigorously, we build new jailbreak evaluation sets with known ground truth answers, by aligning models to refuse questions related to benign and easy-to-evaluate topics (e.g., biology or math). Our evaluation of eight representative jailbreaks across five utility benchmarks reveals a consistent drop in model utility in jailbroken responses, which we term the jailbreak tax. For example, while all jailbreaks we tested bypass guardrails in models aligned to refuse to answer math, this comes at the expense of a drop of up to 92% in accuracy. Overall, our work proposes the jailbreak tax as a new important metric in AI safety, and introduces benchmarks to evaluate existing and future jailbreaks. We make the benchmark available at https://github.com/ethz-spylab/jailbreak-tax



## **16. Look Before You Leap: Enhancing Attention and Vigilance Regarding Harmful Content with GuidelineLLM**

cs.CL

AAAI 2025

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2412.10423v2) [paper-pdf](http://arxiv.org/pdf/2412.10423v2)

**Authors**: Shaoqing Zhang, Zhuosheng Zhang, Kehai Chen, Rongxiang Weng, Muyun Yang, Tiejun Zhao, Min Zhang

**Abstract**: Despite being empowered with alignment mechanisms, large language models (LLMs) are increasingly vulnerable to emerging jailbreak attacks that can compromise their alignment mechanisms. This vulnerability poses significant risks to real-world applications. Existing work faces challenges in both training efficiency and generalization capabilities (i.e., Reinforcement Learning from Human Feedback and Red-Teaming). Developing effective strategies to enable LLMs to resist continuously evolving jailbreak attempts represents a significant challenge. To address this challenge, we propose a novel defensive paradigm called GuidelineLLM, which assists LLMs in recognizing queries that may have harmful content. Before LLMs respond to a query, GuidelineLLM first identifies potential risks associated with the query, summarizes these risks into guideline suggestions, and then feeds these guidelines to the responding LLMs. Importantly, our approach eliminates the necessity for additional safety fine-tuning of the LLMs themselves; only the GuidelineLLM requires fine-tuning. This characteristic enhances the general applicability of GuidelineLLM across various LLMs. Experimental results demonstrate that GuidelineLLM can significantly reduce the attack success rate (ASR) against LLM (an average reduction of 34.17\% ASR) while maintaining the usefulness of LLM in handling benign queries. The code is available at https://github.com/sqzhang-lazy/GuidelineLLM.



## **17. Using Large Language Models for Template Detection from Security Event Logs**

cs.CR

Accepted for publication in International Journal of Information  Security

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2409.05045v3) [paper-pdf](http://arxiv.org/pdf/2409.05045v3)

**Authors**: Risto Vaarandi, Hayretdin Bahsi

**Abstract**: In modern IT systems and computer networks, real-time and offline event log analysis is a crucial part of cyber security monitoring. In particular, event log analysis techniques are essential for the timely detection of cyber attacks and for assisting security experts with the analysis of past security incidents. The detection of line patterns or templates from unstructured textual event logs has been identified as an important task of event log analysis since detected templates represent event types in the event log and prepare the logs for downstream online or offline security monitoring tasks. During the last two decades, a number of template mining algorithms have been proposed. However, many proposed algorithms rely on traditional data mining techniques, and the usage of Large Language Models (LLMs) has received less attention so far. Also, most approaches that harness LLMs are supervised, and unsupervised LLM-based template mining remains an understudied area. The current paper addresses this research gap and investigates the application of LLMs for unsupervised detection of templates from unstructured security event logs.



## **18. Benchmarking Practices in LLM-driven Offensive Security: Testbeds, Metrics, and Experiment Design**

cs.CR

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10112v1) [paper-pdf](http://arxiv.org/pdf/2504.10112v1)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Large Language Models (LLMs) have emerged as a powerful approach for driving offensive penetration-testing tooling. This paper analyzes the methodology and benchmarking practices used for evaluating Large Language Model (LLM)-driven attacks, focusing on offensive uses of LLMs in cybersecurity. We review 16 research papers detailing 15 prototypes and their respective testbeds.   We detail our findings and provide actionable recommendations for future research, emphasizing the importance of extending existing testbeds, creating baselines, and including comprehensive metrics and qualitative analysis. We also note the distinction between security research and practice, suggesting that CTF-based challenges may not fully represent real-world penetration testing scenarios.



## **19. From Vulnerabilities to Remediation: A Systematic Literature Review of LLMs in Code Security**

cs.CR

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2412.15004v3) [paper-pdf](http://arxiv.org/pdf/2412.15004v3)

**Authors**: Enna Basic, Alberto Giaretta

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for automating various programming tasks, including security-related ones, such as detecting and fixing vulnerabilities. Despite their promising capabilities, when required to produce or modify pre-existing code, LLMs could introduce vulnerabilities unbeknown to the programmer. When analyzing code, they could miss clear vulnerabilities or signal nonexistent ones. In this Systematic Literature Review (SLR), we aim to investigate both the security benefits and potential drawbacks of using LLMs for a variety of code-related tasks. In particular, first we focus on the types of vulnerabilities that could be introduced by LLMs, when used for producing code. Second, we analyze the capabilities of LLMs to detect and fix vulnerabilities, in any given code, and how the prompting strategy of choice impacts their performance in these two tasks. Last, we provide an in-depth analysis on how data poisoning attacks on LLMs can impact performance in the aforementioned tasks.



## **20. Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models?**

cs.CR

Accepted to CVPR 2025, codes in process

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10000v1) [paper-pdf](http://arxiv.org/pdf/2504.10000v1)

**Authors**: Yanbo Wang, Jiyang Guan, Jian Liang, Ran He

**Abstract**: Multi-modal large language models (MLLMs) have made significant progress, yet their safety alignment remains limited. Typically, current open-source MLLMs rely on the alignment inherited from their language module to avoid harmful generations. However, the lack of safety measures specifically designed for multi-modal inputs creates an alignment gap, leaving MLLMs vulnerable to vision-domain attacks such as typographic manipulation. Current methods utilize a carefully designed safety dataset to enhance model defense capability, while the specific knowledge or patterns acquired from the high-quality dataset remain unclear. Through comparison experiments, we find that the alignment gap primarily arises from data distribution biases, while image content, response quality, or the contrastive behavior of the dataset makes little contribution to boosting multi-modal safety. To further investigate this and identify the key factors in improving MLLM safety, we propose finetuning MLLMs on a small set of benign instruct-following data with responses replaced by simple, clear rejection sentences. Experiments show that, without the need for labor-intensive collection of high-quality malicious data, model safety can still be significantly improved, as long as a specific fraction of rejection data exists in the finetuning set, indicating the security alignment is not lost but rather obscured during multi-modal pretraining or instruction finetuning. Simply correcting the underlying data bias could narrow the safety gap in the vision domain.



## **21. StruPhantom: Evolutionary Injection Attacks on Black-Box Tabular Agents Powered by Large Language Models**

cs.CR

Work in Progress

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.09841v1) [paper-pdf](http://arxiv.org/pdf/2504.09841v1)

**Authors**: Yang Feng, Xudong Pan

**Abstract**: The proliferation of autonomous agents powered by large language models (LLMs) has revolutionized popular business applications dealing with tabular data, i.e., tabular agents. Although LLMs are observed to be vulnerable against prompt injection attacks from external data sources, tabular agents impose strict data formats and predefined rules on the attacker's payload, which are ineffective unless the agent navigates multiple layers of structural data to incorporate the payload. To address the challenge, we present a novel attack termed StruPhantom which specifically targets black-box LLM-powered tabular agents. Our attack designs an evolutionary optimization procedure which continually refines attack payloads via the proposed constrained Monte Carlo Tree Search augmented by an off-topic evaluator. StruPhantom helps systematically explore and exploit the weaknesses of target applications to achieve goal hijacking. Our evaluation validates the effectiveness of StruPhantom across various LLM-based agents, including those on real-world platforms, and attack scenarios. Our attack achieves over 50% higher success rates than baselines in enforcing the application's response to contain phishing links or malicious codes.



## **22. An Investigation of Large Language Models and Their Vulnerabilities in Spam Detection**

cs.CR

10 pages; presented at HotSoS'2025 as a work in progress paper

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.09776v1) [paper-pdf](http://arxiv.org/pdf/2504.09776v1)

**Authors**: Qiyao Tang, Xiangyang Li

**Abstract**: Spam messages continue to present significant challenges to digital users, cluttering inboxes and posing security risks. Traditional spam detection methods, including rules-based, collaborative, and machine learning approaches, struggle to keep up with the rapidly evolving tactics employed by spammers. This project studies new spam detection systems that leverage Large Language Models (LLMs) fine-tuned with spam datasets. More importantly, we want to understand how LLM-based spam detection systems perform under adversarial attacks that purposefully modify spam emails and data poisoning attacks that exploit the differences between the training data and the massages in detection, to which traditional machine learning models are shown to be vulnerable. This experimentation employs two LLM models of GPT2 and BERT and three spam datasets of Enron, LingSpam, and SMSspamCollection for extensive training and testing tasks. The results show that, while they can function as effective spam filters, the LLM models are susceptible to the adversarial and data poisoning attacks. This research provides very useful insights for future applications of LLM models for information security.



## **23. ControlNET: A Firewall for RAG-based LLM System**

cs.CR

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09593v1) [paper-pdf](http://arxiv.org/pdf/2504.09593v1)

**Authors**: Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.



## **24. AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender**

cs.CR

17 pages, 6 figures, 9 tables

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09466v1) [paper-pdf](http://arxiv.org/pdf/2504.09466v1)

**Authors**: Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu

**Abstract**: Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.



## **25. SaRO: Enhancing LLM Safety through Reasoning-based Alignment**

cs.CL

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09420v1) [paper-pdf](http://arxiv.org/pdf/2504.09420v1)

**Authors**: Yutao Mou, Yuxiao Luo, Shikun Zhang, Wei Ye

**Abstract**: Current safety alignment techniques for large language models (LLMs) face two key challenges: (1) under-generalization, which leaves models vulnerable to novel jailbreak attacks, and (2) over-alignment, which leads to the excessive refusal of benign instructions. Our preliminary investigation reveals semantic overlap between jailbreak/harmful queries and normal prompts in embedding space, suggesting that more effective safety alignment requires a deeper semantic understanding. This motivates us to incorporate safety-policy-driven reasoning into the alignment process. To this end, we propose the Safety-oriented Reasoning Optimization Framework (SaRO), which consists of two stages: (1) Reasoning-style Warmup (RW) that enables LLMs to internalize long-chain reasoning through supervised fine-tuning, and (2) Safety-oriented Reasoning Process Optimization (SRPO) that promotes safety reflection via direct preference optimization (DPO). Extensive experiments demonstrate the superiority of SaRO over traditional alignment methods.



## **26. Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities**

cs.CR

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2502.05209v2) [paper-pdf](http://arxiv.org/pdf/2502.05209v2)

**Authors**: Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, Dylan Hadfield-Menell

**Abstract**: Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the attack success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.



## **27. Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities**

stat.ML

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2501.02406v3) [paper-pdf](http://arxiv.org/pdf/2501.02406v3)

**Authors**: Tara Radvand, Mojtaba Abdolmaleki, Mohamed Mostagir, Ambuj Tewari

**Abstract**: Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. We answer the following question: Given a piece of text, can we identify whether it was produced by LLM $A$ or $B$ (where $B$ can be a human)? We model LLM-generated text as a sequential stochastic process with complete dependence on history and design zero-shot statistical tests to distinguish between (i) the text generated by two different sets of LLMs $A$ (in-house) and $B$ (non-sanctioned) and also (ii) LLM-generated and human-generated texts. We prove that our tests' type I and type II errors decrease exponentially as text length increases. For designing our tests for a given string, we demonstrate that if the string is generated by the evaluator model $A$, the log-perplexity of the string under $A$ converges to the average entropy of the string under $A$, except with an exponentially small probability in the string length. We also show that if $B$ generates the text, except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. For our experiments: First, we present experiments using open-source LLMs to support our theoretical results, and then we provide experiments in a black-box setting with adversarial attacks. Practically, our work enables guaranteed finding of the origin of harmful or false LLM-generated text, which can be useful for combating misinformation and compliance with emerging AI regulations.



## **28. Feature-Aware Malicious Output Detection and Mitigation**

cs.CL

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2504.09191v1) [paper-pdf](http://arxiv.org/pdf/2504.09191v1)

**Authors**: Weilong Dong, Peiguang Li, Yu Tian, Xinyi Zeng, Fengdi Li, Sirui Wang

**Abstract**: The rapid advancement of large language models (LLMs) has brought significant benefits to various domains while introducing substantial risks. Despite being fine-tuned through reinforcement learning, LLMs lack the capability to discern malicious content, limiting their defense against jailbreak. To address these safety concerns, we propose a feature-aware method for harmful response rejection (FMM), which detects the presence of malicious features within the model's feature space and adaptively adjusts the model's rejection mechanism. By employing a simple discriminator, we detect potential malicious traits during the decoding phase. Upon detecting features indicative of toxic tokens, FMM regenerates the current token. By employing activation patching, an additional rejection vector is incorporated during the subsequent token generation, steering the model towards a refusal response. Experimental results demonstrate the effectiveness of our approach across multiple language models and diverse attack techniques, while crucially maintaining the models' standard generation capabilities.



## **29. Privacy Preservation in Gen AI Applications**

cs.CR

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2504.09095v1) [paper-pdf](http://arxiv.org/pdf/2504.09095v1)

**Authors**: Swetha S, Ram Sundhar K Shaju, Rakshana M, Ganesh R, Balavedhaa S, Thiruvaazhi U

**Abstract**: The ability of machines to comprehend and produce language that is similar to that of humans has revolutionized sectors like customer service, healthcare, and finance thanks to the quick advances in Natural Language Processing (NLP), which are fueled by Generative Artificial Intelligence (AI) and Large Language Models (LLMs). However, because LLMs trained on large datasets may unintentionally absorb and reveal Personally Identifiable Information (PII) from user interactions, these capabilities also raise serious privacy concerns. Deep neural networks' intricacy makes it difficult to track down or stop the inadvertent storing and release of private information, which raises serious concerns about the privacy and security of AI-driven data. This study tackles these issues by detecting Generative AI weaknesses through attacks such as data extraction, model inversion, and membership inference. A privacy-preserving Generative AI application that is resistant to these assaults is then developed. It ensures privacy without sacrificing functionality by using methods to identify, alter, or remove PII before to dealing with LLMs. In order to determine how well cloud platforms like Microsoft Azure, Google Cloud, and AWS provide privacy tools for protecting AI applications, the study also examines these technologies. In the end, this study offers a fundamental privacy paradigm for generative AI systems, focusing on data security and moral AI implementation, and opening the door to a more secure and conscientious use of these tools.



## **30. Detecting Instruction Fine-tuning Attack on Language Models with Influence Function**

cs.LG

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2504.09026v1) [paper-pdf](http://arxiv.org/pdf/2504.09026v1)

**Authors**: Jiawei Li

**Abstract**: Instruction fine-tuning attacks pose a significant threat to large language models (LLMs) by subtly embedding poisoned data in fine-tuning datasets, which can trigger harmful or unintended responses across a range of tasks. This undermines model alignment and poses security risks in real-world deployment. In this work, we present a simple and effective approach to detect and mitigate such attacks using influence functions, a classical statistical tool adapted for machine learning interpretation. Traditionally, the high computational costs of influence functions have limited their application to large models and datasets. The recent Eigenvalue-Corrected Kronecker-Factored Approximate Curvature (EK-FAC) approximation method enables efficient influence score computation, making it feasible for large-scale analysis.   We are the first to apply influence functions for detecting language model instruction fine-tuning attacks on large-scale datasets, as both the instruction fine-tuning attack on language models and the influence calculation approximation technique are relatively new. Our large-scale empirical evaluation of influence functions on 50,000 fine-tuning examples and 32 tasks reveals a strong association between influence scores and sentiment. Building on this, we introduce a novel sentiment transformation combined with influence functions to detect and remove critical poisons -- poisoned data points that skew model predictions. Removing these poisons (only 1% of total data) recovers model performance to near-clean levels, demonstrating the effectiveness and efficiency of our approach. Artifact is available at https://github.com/lijiawei20161002/Poison-Detection.   WARNING: This paper contains offensive data examples.



## **31. Robust Steganography from Large Language Models**

cs.CR

36 pages, 9 figures

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08977v1) [paper-pdf](http://arxiv.org/pdf/2504.08977v1)

**Authors**: Neil Perry, Sanket Gupte, Nishant Pitta, Lior Rotem

**Abstract**: Recent steganographic schemes, starting with Meteor (CCS'21), rely on leveraging large language models (LLMs) to resolve a historically-challenging task of disguising covert communication as ``innocent-looking'' natural-language communication. However, existing methods are vulnerable to ``re-randomization attacks,'' where slight changes to the communicated text, that might go unnoticed, completely destroy any hidden message. This is also a vulnerability in more traditional encryption-based stegosystems, where adversaries can modify the randomness of an encryption scheme to destroy the hidden message while preserving an acceptable covertext to ordinary users. In this work, we study the problem of robust steganography. We introduce formal definitions of weak and strong robust LLM-based steganography, corresponding to two threat models in which natural language serves as a covertext channel resistant to realistic re-randomization attacks. We then propose two constructions satisfying these notions. We design and implement our steganographic schemes that embed arbitrary secret messages into natural language text generated by LLMs, ensuring recoverability even under adversarial paraphrasing and rewording attacks. To support further research and real-world deployment, we release our implementation and datasets for public use.



## **32. Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups**

cs.CL

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.06160v3) [paper-pdf](http://arxiv.org/pdf/2504.06160v3)

**Authors**: Rijul Magu, Arka Dutta, Sean Kim, Ashiqur R. KhudaBukhsh, Munmun De Choudhury

**Abstract**: Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from sociological foundations of stigmatization theory, our stigmatization analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation.



## **33. MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits**

cs.CR

27 pages, 21 figures, and 2 Tables. Cleans up the TeX source

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.03767v2) [paper-pdf](http://arxiv.org/pdf/2504.03767v2)

**Authors**: Brandon Radosevich, John Halloran

**Abstract**: To reduce development overhead and enable seamless integration between potential components comprising any given generative AI application, the Model Context Protocol (MCP) (Anthropic, 2024) has recently been released and subsequently widely adopted. The MCP is an open protocol that standardizes API calls to large language models (LLMs), data sources, and agentic tools. By connecting multiple MCP servers, each defined with a set of tools, resources, and prompts, users are able to define automated workflows fully driven by LLMs. However, we show that the current MCP design carries a wide range of security risks for end users. In particular, we demonstrate that industry-leading LLMs may be coerced into using MCP tools to compromise an AI developer's system through various attacks, such as malicious code execution, remote access control, and credential theft. To proactively mitigate these and related attacks, we introduce a safety auditing tool, MCPSafetyScanner, the first agentic tool to assess the security of an arbitrary MCP server. MCPScanner uses several agents to (a) automatically determine adversarial samples given an MCP server's tools and resources; (b) search for related vulnerabilities and remediations based on those samples; and (c) generate a security report detailing all findings. Our work highlights serious security issues with general-purpose agentic workflows while also providing a proactive tool to audit MCP server safety and address detected vulnerabilities before deployment.   The described MCP server auditing tool, MCPSafetyScanner, is freely available at: https://github.com/johnhalloran321/mcpSafetyScanner



## **34. SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models**

cs.CV

Submitted to CVPR 2025 Workshop EVAL-FoMo-2

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.04893v2) [paper-pdf](http://arxiv.org/pdf/2504.04893v2)

**Authors**: Justus Westerhoff, Erblina Purelku, Jakob Hackstein, Leo Pinetzki, Lorenz Hufe

**Abstract**: Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper under https://huggingface.co/datasets/BLISS-e-V/SCAM, along with the code for evaluations at https://github.com/Bliss-e-V/SCAM.



## **35. X-Guard: Multilingual Guard Agent for Content Moderation**

cs.CR

34 pages, 15 figures

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08848v1) [paper-pdf](http://arxiv.org/pdf/2504.08848v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Ph. D

**Abstract**: Large Language Models (LLMs) have rapidly become integral to numerous applications in critical domains where reliability is paramount. Despite significant advances in safety frameworks and guardrails, current protective measures exhibit crucial vulnerabilities, particularly in multilingual contexts. Existing safety systems remain susceptible to adversarial attacks in low-resource languages and through code-switching techniques, primarily due to their English-centric design. Furthermore, the development of effective multilingual guardrails is constrained by the scarcity of diverse cross-lingual training data. Even recent solutions like Llama Guard-3, while offering multilingual support, lack transparency in their decision-making processes. We address these challenges by introducing X-Guard agent, a transparent multilingual safety agent designed to provide content moderation across diverse linguistic contexts. X-Guard effectively defends against both conventional low-resource language attacks and sophisticated code-switching attacks. Our approach includes: curating and enhancing multiple open-source safety datasets with explicit evaluation rationales; employing a jury of judges methodology to mitigate individual judge LLM provider biases; creating a comprehensive multilingual safety dataset spanning 132 languages with 5 million data points; and developing a two-stage architecture combining a custom-finetuned mBART-50 translation module with an evaluation X-Guard 3B model trained through supervised finetuning and GRPO training. Our empirical evaluations demonstrate X-Guard's effectiveness in detecting unsafe content across multiple languages while maintaining transparency throughout the safety evaluation process. Our work represents a significant advancement in creating robust, transparent, and linguistically inclusive safety systems for LLMs and its integrated systems.



## **36. GenXSS: an AI-Driven Framework for Automated Detection of XSS Attacks in WAFs**

cs.CR

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08176v1) [paper-pdf](http://arxiv.org/pdf/2504.08176v1)

**Authors**: Vahid Babaey, Arun Ravindran

**Abstract**: The increasing reliance on web services has led to a rise in cybersecurity threats, particularly Cross-Site Scripting (XSS) attacks, which target client-side layers of web applications by injecting malicious scripts. Traditional Web Application Firewalls (WAFs) struggle to detect highly obfuscated and complex attacks, as their rules require manual updates. This paper presents a novel generative AI framework that leverages Large Language Models (LLMs) to enhance XSS mitigation. The framework achieves two primary objectives: (1) generating sophisticated and syntactically validated XSS payloads using in-context learning, and (2) automating defense mechanisms by testing these attacks against a vulnerable application secured by a WAF, classifying bypassing attacks, and generating effective WAF security rules. Experimental results using GPT-4o demonstrate the framework's effectiveness generating 264 XSS payloads, 83% of which were validated, with 80% bypassing ModSecurity WAF equipped with an industry standard security rule set developed by the Open Web Application Security Project (OWASP) to protect against web vulnerabilities. Through rule generation, 86% of previously successful attacks were blocked using only 15 new rules. In comparison, Google Gemini Pro achieved a lower bypass rate of 63%, highlighting performance differences across LLMs.



## **37. Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge**

cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07887v1) [paper-pdf](http://arxiv.org/pdf/2504.07887v1)

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models.



## **38. PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization**

cs.CR

Accepted at SIGIR 2025

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07717v1) [paper-pdf](http://arxiv.org/pdf/2504.07717v1)

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications, e.g., medical question-answering, mathematical sciences, and code generation. However, they also exhibit inherent limitations, such as outdated knowledge and susceptibility to hallucinations. Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address these issues, but it also introduces new vulnerabilities. Recent efforts have focused on the security of RAG-based LLMs, yet existing attack methods face three critical challenges: (1) their effectiveness declines sharply when only a limited number of poisoned texts can be injected into the knowledge database, (2) they lack sufficient stealth, as the attacks are often detectable by anomaly detection systems, which compromises their effectiveness, and (3) they rely on heuristic approaches to generate poisoned texts, lacking formal optimization frameworks and theoretic guarantees, which limits their effectiveness and applicability. To address these issues, we propose coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack that introduces a small number of poisoned texts into the knowledge database while embedding a backdoor trigger within the prompt. When activated, the trigger causes the LLM to generate pre-designed responses to targeted queries, while maintaining normal behavior in other contexts. This ensures both high effectiveness and stealth. We formulate the attack generation process as a bilevel optimization problem leveraging a principled optimization framework to develop optimal poisoned texts and triggers. Extensive experiments across diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving a high attack success rate even with a limited number of poisoned texts and significantly improved stealth compared to existing methods.



## **39. Agent That Debugs: Dynamic State-Guided Vulnerability Repair**

cs.SE

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07634v1) [paper-pdf](http://arxiv.org/pdf/2504.07634v1)

**Authors**: Zhengyao Liu, Yunlong Ma, Jingxuan Xu, Junchen Ai, Xiang Gao, Hailong Sun, Abhik Roychoudhury

**Abstract**: In recent years, more vulnerabilities have been discovered every day, while manual vulnerability repair requires specialized knowledge and is time-consuming. As a result, many detected or even published vulnerabilities remain unpatched, thereby increasing the exposure of software systems to attacks. Recent advancements in agents based on Large Language Models have demonstrated their increasing capabilities in code understanding and generation, which can be promising to achieve automated vulnerability repair. However, the effectiveness of agents based on static information retrieval is still not sufficient for patch generation. To address the challenge, we propose a program repair agent called VulDebugger that fully utilizes both static and dynamic context, and it debugs programs in a manner akin to humans. The agent inspects the actual state of the program via the debugger and infers expected states via constraints that need to be satisfied. By continuously comparing the actual state with the expected state, it deeply understands the root causes of the vulnerabilities and ultimately accomplishes repairs. We experimentally evaluated VulDebugger on 50 real-life projects. With 60.00% successfully fixed, VulDebugger significantly outperforms state-of-the-art approaches for vulnerability repair.



## **40. Defense against Prompt Injection Attacks via Mixture of Encodings**

cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07467v1) [paper-pdf](http://arxiv.org/pdf/2504.07467v1)

**Authors**: Ruiyi Zhang, David Sullivan, Kyle Jackson, Pengtao Xie, Mei Chen

**Abstract**: Large Language Models (LLMs) have emerged as a dominant approach for a wide range of NLP tasks, with their access to external information further enhancing their capabilities. However, this introduces new vulnerabilities, known as prompt injection attacks, where external content embeds malicious instructions that manipulate the LLM's output. Recently, the Base64 defense has been recognized as one of the most effective methods for reducing success rate of prompt injection attacks. Despite its efficacy, this method can degrade LLM performance on certain NLP tasks. To address this challenge, we propose a novel defense mechanism: mixture of encodings, which utilizes multiple character encodings, including Base64. Extensive experimental results show that our method achieves one of the lowest attack success rates under prompt injection attacks, while maintaining high performance across all NLP tasks, outperforming existing character encoding-based defense methods. This underscores the effectiveness of our mixture of encodings strategy for both safety and task performance metrics.



## **41. Achilles Heel of Distributed Multi-Agent Systems**

cs.MA

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07461v1) [paper-pdf](http://arxiv.org/pdf/2504.07461v1)

**Authors**: Yiting Zhang, Yijiang Li, Tianwei Zhao, Kaijie Zhu, Haohan Wang, Nuno Vasconcelos

**Abstract**: Multi-agent system (MAS) has demonstrated exceptional capabilities in addressing complex challenges, largely due to the integration of multiple large language models (LLMs). However, the heterogeneity of LLMs, the scalability of quantities of LLMs, and local computational constraints pose significant challenges to hosting these models locally. To address these issues, we propose a new framework termed Distributed Multi-Agent System (DMAS). In DMAS, heterogeneous third-party agents function as service providers managed remotely by a central MAS server and each agent offers its services through API interfaces. However, the distributed nature of DMAS introduces several concerns about trustworthiness. In this paper, we study the Achilles heel of distributed multi-agent systems, identifying four critical trustworthiness challenges: free riding, susceptibility to malicious attacks, communication inefficiencies, and system instability. Extensive experiments across seven frameworks and four datasets reveal significant vulnerabilities of the DMAS. These attack strategies can lead to a performance degradation of up to 80% and attain a 100% success rate in executing free riding and malicious attacks. We envision our work will serve as a useful red-teaming tool for evaluating future multi-agent systems and spark further research on trustworthiness challenges in distributed multi-agent systems.



## **42. Code Generation with Small Language Models: A Deep Evaluation on Codeforces**

cs.SE

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2504.07343v1) [paper-pdf](http://arxiv.org/pdf/2504.07343v1)

**Authors**: Débora Souza, Rohit Gheyi, Lucas Albuquerque, Gustavo Soares, Márcio Ribeiro

**Abstract**: Large Language Models (LLMs) have demonstrated capabilities in code generation, potentially boosting developer productivity. However, their widespread adoption remains limited by high computational costs, significant energy demands, and security risks such as data leakage and adversarial attacks. As a lighter-weight alternative, Small Language Models (SLMs) offer faster inference, lower deployment overhead, and better adaptability to domain-specific tasks, making them an attractive option for real-world applications. While prior research has benchmarked LLMs on competitive programming tasks, such evaluations often focus narrowly on metrics like Elo scores or pass rates, overlooking deeper insights into model behavior, failure patterns, and problem diversity. Furthermore, the potential of SLMs to tackle complex tasks such as competitive programming remains underexplored. In this study, we benchmark five open SLMs - LLAMA 3.2 3B, GEMMA 2 9B, GEMMA 3 12B, DEEPSEEK-R1 14B, and PHI-4 14B - across 280 Codeforces problems spanning Elo ratings from 800 to 2100 and covering 36 distinct topics. All models were tasked with generating Python solutions. PHI-4 14B achieved the best performance among SLMs, with a pass@3 of 63.6%, approaching the proprietary O3-MINI-HIGH (86.8%). In addition, we evaluated PHI-4 14B on C++ and found that combining outputs from both Python and C++ increases its aggregated pass@3 to 73.6%. A qualitative analysis of PHI-4 14B's incorrect outputs revealed that some failures were due to minor implementation issues - such as handling edge cases or correcting variable initialization - rather than deeper reasoning flaws.



## **43. LLM Safeguard is a Double-Edged Sword: Exploiting False Positives for Denial-of-Service Attacks**

cs.CR

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2410.02916v3) [paper-pdf](http://arxiv.org/pdf/2410.02916v3)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern for large language models (LLMs) in open deployment, motivating the development of safeguard methods that enforce ethical and responsible use through safety alignment or guardrail mechanisms. Jailbreak attacks that exploit the \emph{false negatives} of safeguard methods have emerged as a prominent research focus in the field of LLM security. However, we found that the malicious attackers could also exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a denial-of-service (DoS) affecting LLM users. To bridge the knowledge gap of this overlooked threat, we explore multiple attack methods that include inserting a short adversarial prompt into user prompt templates and corrupting the LLM on the server by poisoned fine-tuning. In both ways, the attack triggers safeguard rejections of user requests from the client. Our evaluation demonstrates the severity of this threat across multiple scenarios. For instance, in the scenario of white-box adversarial prompt injection, the attacker can use our optimization process to automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97% of user requests on Llama Guard 3. These findings reveal a new dimension in LLM safeguard evaluation -- adversarial robustness to false positives.



## **44. SafeMLRM: Demystifying Safety in Multi-modal Large Reasoning Models**

cs.LG

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2504.08813v1) [paper-pdf](http://arxiv.org/pdf/2504.08813v1)

**Authors**: Junfeng Fang, Yukai Wang, Ruipeng Wang, Zijun Yao, Kun Wang, An Zhang, Xiang Wang, Tat-Seng Chua

**Abstract**: The rapid advancement of multi-modal large reasoning models (MLRMs) -- enhanced versions of multimodal language models (MLLMs) equipped with reasoning capabilities -- has revolutionized diverse applications. However, their safety implications remain underexplored. While prior work has exposed critical vulnerabilities in unimodal reasoning models, MLRMs introduce distinct risks from cross-modal reasoning pathways. This work presents the first systematic safety analysis of MLRMs through large-scale empirical studies comparing MLRMs with their base MLLMs. Our experiments reveal three critical findings: (1) The Reasoning Tax: Acquiring reasoning capabilities catastrophically degrades inherited safety alignment. MLRMs exhibit 37.44% higher jailbreaking success rates than base MLLMs under adversarial attacks. (2) Safety Blind Spots: While safety degradation is pervasive, certain scenarios (e.g., Illegal Activity) suffer 25 times higher attack rates -- far exceeding the average 3.4 times increase, revealing scenario-specific vulnerabilities with alarming cross-model and datasets consistency. (3) Emergent Self-Correction: Despite tight reasoning-answer safety coupling, MLRMs demonstrate nascent self-correction -- 16.9% of jailbroken reasoning steps are overridden by safe answers, hinting at intrinsic safeguards. These findings underscore the urgency of scenario-aware safety auditing and mechanisms to amplify MLRMs' self-correction potential. To catalyze research, we open-source OpenSafeMLRM, the first toolkit for MLRM safety evaluation, providing unified interface for mainstream models, datasets, and jailbreaking methods. Our work calls for immediate efforts to harden reasoning-augmented AI, ensuring its transformative potential aligns with ethical safeguards.



## **45. JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model**

cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.03770v2) [paper-pdf](http://arxiv.org/pdf/2504.03770v2)

**Authors**: Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, Yue Zhao

**Abstract**: Multimodal large language models (MLLMs) excel in vision-language tasks but also pose significant risks of generating harmful content, particularly through jailbreak attacks. Jailbreak attacks refer to intentional manipulations that bypass safety mechanisms in models, leading to the generation of inappropriate or unsafe content. Detecting such attacks is critical to ensuring the responsible deployment of MLLMs. Existing jailbreak detection methods face three primary challenges: (1) Many rely on model hidden states or gradients, limiting their applicability to white-box models, where the internal workings of the model are accessible; (2) They involve high computational overhead from uncertainty-based analysis, which limits real-time detection, and (3) They require fully labeled harmful datasets, which are often scarce in real-world settings. To address these issues, we introduce a test-time adaptive framework called JAILDAM. Our method leverages a memory-based approach guided by policy-driven unsafe knowledge representations, eliminating the need for explicit exposure to harmful data. By dynamically updating unsafe knowledge during test-time, our framework improves generalization to unseen jailbreak strategies while maintaining efficiency. Experiments on multiple VLM jailbreak benchmarks demonstrate that JAILDAM delivers state-of-the-art performance in harmful content detection, improving both accuracy and speed.



## **46. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

cs.IR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05804v1) [paper-pdf](http://arxiv.org/pdf/2504.05804v1)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems.



## **47. Separator Injection Attack: Uncovering Dialogue Biases in Large Language Models Caused by Role Separators**

cs.CL

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05689v1) [paper-pdf](http://arxiv.org/pdf/2504.05689v1)

**Authors**: Xitao Li, Haijun Wang, Jiang Wu, Ting Liu

**Abstract**: Conversational large language models (LLMs) have gained widespread attention due to their instruction-following capabilities. To ensure conversational LLMs follow instructions, role separators are employed to distinguish between different participants in a conversation. However, incorporating role separators introduces potential vulnerabilities. Misusing roles can lead to prompt injection attacks, which can easily misalign the model's behavior with the user's intentions, raising significant security concerns. Although various prompt injection attacks have been proposed, recent research has largely overlooked the impact of role separators on safety. This highlights the critical need to thoroughly understand the systemic weaknesses in dialogue systems caused by role separators. This paper identifies modeling weaknesses caused by role separators. Specifically, we observe a strong positional bias associated with role separators, which is inherent in the format of dialogue modeling and can be triggered by the insertion of role separators. We further develop the Separators Injection Attack (SIA), a new orthometric attack based on role separators. The experiment results show that SIA is efficient and extensive in manipulating model behavior with an average gain of 18.2% for manual methods and enhances the attack success rate to 100% with automatic methods.



## **48. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05652v1) [paper-pdf](http://arxiv.org/pdf/2504.05652v1)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Jie-Zhang

**Abstract**: Large Language Models (LLMs) have become increasingly integral to a wide range of applications. However, they still remain the threat of jailbreak attacks, where attackers manipulate designed prompts to make the models elicit malicious outputs. Analyzing jailbreak methods can help us delve into the weakness of LLMs and improve it. In this paper, We reveal a vulnerability in large language models (LLMs), which we term Defense Threshold Decay (DTD), by analyzing the attention weights of the model's output on input and subsequent output on prior output: as the model generates substantial benign content, its attention weights shift from the input to prior output, making it more susceptible to jailbreak attacks. To demonstrate the exploitability of DTD, we propose a novel jailbreak attack method, Sugar-Coated Poison (SCP), which induces the model to generate substantial benign content through benign input and adversarial reasoning, subsequently producing malicious content. To mitigate such attacks, we introduce a simple yet effective defense strategy, POSD, which significantly reduces jailbreak success rates while preserving the model's generalization capabilities.



## **49. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments**

cs.CV

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2412.00114v2) [paper-pdf](http://arxiv.org/pdf/2412.00114v2)

**Authors**: Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent. Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration. The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image. This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications. Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.



## **50. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05050v1) [paper-pdf](http://arxiv.org/pdf/2504.05050v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.



