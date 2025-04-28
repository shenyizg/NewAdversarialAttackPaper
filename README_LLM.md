# Latest Large Language Model Attack Papers
**update at 2025-04-28 17:19:48**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. ThreMoLIA: Threat Modeling of Large Language Model-Integrated Applications**

cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18369v1) [paper-pdf](http://arxiv.org/pdf/2504.18369v1)

**Authors**: Felix Viktor Jedrzejewski, Davide Fucci, Oleksandr Adamov

**Abstract**: Large Language Models (LLMs) are currently being integrated into industrial software applications to help users perform more complex tasks in less time. However, these LLM-Integrated Applications (LIA) expand the attack surface and introduce new kinds of threats. Threat modeling is commonly used to identify these threats and suggest mitigations. However, it is a time-consuming practice that requires the involvement of a security practitioner. Our goals are to 1) provide a method for performing threat modeling for LIAs early in their lifecycle, (2) develop a threat modeling tool that integrates existing threat models, and (3) ensure high-quality threat modeling. To achieve the goals, we work in collaboration with our industry partner. Our proposed way of performing threat modeling will benefit industry by requiring fewer security experts' participation and reducing the time spent on this activity. Our proposed tool combines LLMs and Retrieval Augmented Generation (RAG) and uses sources such as existing threat models and application architecture repositories to continuously create and update threat models. We propose to evaluate the tool offline -- i.e., using benchmarking -- and online with practitioners in the field. We conducted an early evaluation using ChatGPT on a simple LIA and obtained results that encouraged us to proceed with our research efforts.



## **2. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.14348v3) [paper-pdf](http://arxiv.org/pdf/2504.14348v3)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.



## **3. NoEsis: Differentially Private Knowledge Transfer in Modular LLM Adaptation**

cs.CR

ICLR 2025 MCDC workshop

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18147v1) [paper-pdf](http://arxiv.org/pdf/2504.18147v1)

**Authors**: Rob Romijnders, Stefanos Laskaridis, Ali Shahin Shamsabadi, Hamed Haddadi

**Abstract**: Large Language Models (LLM) are typically trained on vast amounts of data from various sources. Even when designed modularly (e.g., Mixture-of-Experts), LLMs can leak privacy on their sources. Conversely, training such models in isolation arguably prohibits generalization. To this end, we propose a framework, NoEsis, which builds upon the desired properties of modularity, privacy, and knowledge transfer. NoEsis integrates differential privacy with a hybrid two-staged parameter-efficient fine-tuning that combines domain-specific low-rank adapters, acting as experts, with common prompt tokens, acting as a knowledge-sharing backbone. Results from our evaluation on CodeXGLUE showcase that NoEsis can achieve provable privacy guarantees with tangible knowledge transfer across domains, and empirically show protection against Membership Inference Attacks. Finally, on code completion tasks, NoEsis bridges at least 77% of the accuracy gap between the non-shared and the non-private baseline.



## **4. Automating Function-Level TARA for Automotive Full-Lifecycle Security**

cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18083v1) [paper-pdf](http://arxiv.org/pdf/2504.18083v1)

**Authors**: Yuqiao Yang, Yongzhao Zhang, Wenhao Liu, Jun Li, Pengtao Shi, DingYu Zhong, Jie Yang, Ting Chen, Sheng Cao, Yuntao Ren, Yongyue Wu, Xiaosong Zhang

**Abstract**: As modern vehicles evolve into intelligent and connected systems, their growing complexity introduces significant cybersecurity risks. Threat Analysis and Risk Assessment (TARA) has therefore become essential for managing these risks under mandatory regulations. However, existing TARA automation methods rely on static threat libraries, limiting their utility in the detailed, function-level analyses demanded by industry. This paper introduces DefenseWeaver, the first system that automates function-level TARA using component-specific details and large language models (LLMs). DefenseWeaver dynamically generates attack trees and risk evaluations from system configurations described in an extended OpenXSAM++ format, then employs a multi-agent framework to coordinate specialized LLM roles for more robust analysis. To further adapt to evolving threats and diverse standards, DefenseWeaver incorporates Low-Rank Adaptation (LoRA) fine-tuning and Retrieval-Augmented Generation (RAG) with expert-curated TARA reports. We validated DefenseWeaver through deployment in four automotive security projects, where it identified 11 critical attack paths, verified through penetration testing, and subsequently reported and remediated by the relevant automakers and suppliers. Additionally, DefenseWeaver demonstrated cross-domain adaptability, successfully applying to unmanned aerial vehicles (UAVs) and marine navigation systems. In comparison to human experts, DefenseWeaver outperformed manual attack tree generation across six assessment scenarios. Integrated into commercial cybersecurity platforms such as UAES and Xiaomi, DefenseWeaver has generated over 8,200 attack trees. These results highlight its ability to significantly reduce processing time, and its scalability and transformative impact on cybersecurity across industries.



## **5. DREAM: Disentangling Risks to Enhance Safety Alignment in Multimodal Large Language Models**

cs.CL

[NAACL 2025] The first four authors contribute equally, 23 pages,  repo at https://github.com/Kizna1ver/DREAM

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18053v1) [paper-pdf](http://arxiv.org/pdf/2504.18053v1)

**Authors**: Jianyu Liu, Hangyu Guo, Ranjie Duan, Xingyuan Bu, Yancheng He, Shilong Li, Hui Huang, Jiaheng Liu, Yucheng Wang, Chenchen Jing, Xingwei Qu, Xiao Zhang, Yingshui Tan, Yanan Wu, Jihao Gu, Yangguang Li, Jianke Zhu

**Abstract**: Multimodal Large Language Models (MLLMs) pose unique safety challenges due to their integration of visual and textual data, thereby introducing new dimensions of potential attacks and complex risk combinations. In this paper, we begin with a detailed analysis aimed at disentangling risks through step-by-step reasoning within multimodal inputs. We find that systematic multimodal risk disentanglement substantially enhances the risk awareness of MLLMs. Via leveraging the strong discriminative abilities of multimodal risk disentanglement, we further introduce \textbf{DREAM} (\textit{\textbf{D}isentangling \textbf{R}isks to \textbf{E}nhance Safety \textbf{A}lignment in \textbf{M}LLMs}), a novel approach that enhances safety alignment in MLLMs through supervised fine-tuning and iterative Reinforcement Learning from AI Feedback (RLAIF). Experimental results show that DREAM significantly boosts safety during both inference and training phases without compromising performance on normal tasks (namely oversafety), achieving a 16.17\% improvement in the SIUO safe\&effective score compared to GPT-4V. The data and code are available at https://github.com/Kizna1ver/DREAM.



## **6. Unified Attacks to Large Language Model Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation**

cs.CL

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17480v1) [paper-pdf](http://arxiv.org/pdf/2504.17480v1)

**Authors**: Xin Yi, Shunfan Zhengc, Linlin Wanga, Xiaoling Wang, Liang He

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.



## **7. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

cs.CR

17 pages, 11 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2411.11114v2) [paper-pdf](http://arxiv.org/pdf/2411.11114v2)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Wenhui Zhang, Qinglong Wang, Rui Zheng

**Abstract**: Despite the outstanding performance of Large language Models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses. Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explained typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of jailbreak attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives~(which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on five mainstream LLMs under seven jailbreak strategies. Our evaluation reveals that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. This manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals. Notably, we find a strong and consistent correlation between representation deception and activation shift of key circuits across diverse jailbreak methods and multiple LLMs.



## **8. GraphRAG under Fire**

cs.LG

13 pages

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2501.14050v2) [paper-pdf](http://arxiv.org/pdf/2501.14050v2)

**Authors**: Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, Ting Wang

**Abstract**: GraphRAG advances retrieval-augmented generation (RAG) by structuring external knowledge as multi-scale knowledge graphs, enabling language models to integrate both broad context and granular details in their generation. While GraphRAG has demonstrated success across domains, its security implications remain largely unexplored. To bridge this gap, this work examines GraphRAG's vulnerability to poisoning attacks, uncovering an intriguing security paradox: compared to conventional RAG, GraphRAG's graph-based indexing and retrieval enhance resilience against simple poisoning attacks; yet, the same features also create new attack surfaces. We present GRAGPoison, a novel attack that exploits shared relations in the underlying knowledge graph to craft poisoning text capable of compromising multiple queries simultaneously. GRAGPoison employs three key strategies: i) relation injection to introduce false knowledge, ii) relation enhancement to amplify poisoning influence, and iii) narrative generation to embed malicious content within coherent text. Empirical evaluation across diverse datasets and models shows that GRAGPoison substantially outperforms existing attacks in terms of effectiveness (up to 98\% success rate) and scalability (using less than 68\% poisoning text) on various GraphRAG-based systems. We also explore potential defensive measures and their limitations, identifying promising directions for future research.



## **9. CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent**

cs.CR

Accepted by KDD 2024;

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.13192v2) [paper-pdf](http://arxiv.org/pdf/2504.13192v2)

**Authors**: Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, Feiran Huang

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems (RecSys) have brought significant advances in personalized user experience and have attracted considerable attention. Despite the impressive progress, the research question regarding the safety vulnerability of LLM-empowered RecSys still remains largely under-investigated. Given the security and privacy concerns, it is more practical to focus on attacking the black-box RecSys, where attackers can only observe the system's inputs and outputs. However, traditional attack approaches employing reinforcement learning (RL) agents are not effective for attacking LLM-empowered RecSys due to the limited capabilities in processing complex textual inputs, planning, and reasoning. On the other hand, LLMs provide unprecedented opportunities to serve as attack agents to attack RecSys because of their impressive capability in simulating human-like decision-making processes. Therefore, in this paper, we propose a novel attack framework called CheatAgent by harnessing the human-like capabilities of LLMs, where an LLM-based agent is developed to attack LLM-Empowered RecSys. Specifically, our method first identifies the insertion position for maximum impact with minimal input modification. After that, the LLM agent is designed to generate adversarial perturbations to insert at target positions. To further improve the quality of generated perturbations, we utilize the prompt tuning technique to improve attacking strategies via feedback from the victim RecSys iteratively. Extensive experiments across three real-world datasets demonstrate the effectiveness of our proposed attacking method.



## **10. Automatically Generating Rules of Malicious Software Packages via Large Language Model**

cs.SE

14 pages, 11 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17198v1) [paper-pdf](http://arxiv.org/pdf/2504.17198v1)

**Authors**: XiangRui Zhang, HaoYu Chen, Yongzhong He, Wenjia Niu, Qiang Li

**Abstract**: Today's security tools predominantly rely on predefined rules crafted by experts, making them poorly adapted to the emergence of software supply chain attacks. To tackle this limitation, we propose a novel tool, RuleLLM, which leverages large language models (LLMs) to automate rule generation for OSS ecosystems. RuleLLM extracts metadata and code snippets from malware as its input, producing YARA and Semgrep rules that can be directly deployed in software development. Specifically, the rule generation task involves three subtasks: crafting rules, refining rules, and aligning rules. To validate RuleLLM's effectiveness, we implemented a prototype system and conducted experiments on the dataset of 1,633 malicious packages. The results are promising that RuleLLM generated 763 rules (452 YARA and 311 Semgrep) with a precision of 85.2\% and a recall of 91.8\%, outperforming state-of-the-art (SOTA) tools and scored-based approaches. We further analyzed generated rules and proposed a rule taxonomy: 11 categories and 38 subcategories.



## **11. Robo-Troj: Attacking LLM-based Task Planners**

cs.RO

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.17070v1) [paper-pdf](http://arxiv.org/pdf/2504.17070v1)

**Authors**: Mohaiminul Al Nahian, Zainab Altaweel, David Reitano, Sabbir Ahmed, Saumitra Lohokare, Shiqi Zhang, Adnan Siraj Rakin

**Abstract**: Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems.



## **12. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

cs.LG

END (will never be modified again!!) :Jumps-Diffusion and stock  market: Better quantify uncertainty in financial simulations

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2407.14573v7) [paper-pdf](http://arxiv.org/pdf/2407.14573v7)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.



## **13. Safety Pretraining: Toward the Next Generation of Safe AI**

cs.LG

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16980v1) [paper-pdf](http://arxiv.org/pdf/2504.16980v1)

**Authors**: Pratyush Maini, Sachin Goyal, Dylan Sam, Alex Robey, Yash Savani, Yiding Jiang, Andy Zou, Zacharcy C. Lipton, J. Zico Kolter

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes settings, the risk of generating harmful or toxic content remains a central challenge. Post-hoc alignment methods are brittle: once unsafe patterns are learned during pretraining, they are hard to remove. We present a data-centric pretraining framework that builds safety into the model from the start. Our contributions include: (i) a safety classifier trained on 10,000 GPT-4 labeled examples, used to filter 600B tokens; (ii) the largest synthetic safety dataset to date (100B tokens) generated via recontextualization of harmful web data; (iii) RefuseWeb and Moral Education datasets that convert harmful prompts into refusal dialogues and web-style educational material; (iv) Harmfulness-Tag annotations injected during pretraining to flag unsafe content and steer away inference from harmful generations; and (v) safety evaluations measuring base model behavior before instruction tuning. Our safety-pretrained models reduce attack success rates from 38.8% to 8.4% with no performance degradation on standard LLM safety benchmarks.



## **14. Context-Enhanced Vulnerability Detection Based on Large Language Model**

cs.SE

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16877v1) [paper-pdf](http://arxiv.org/pdf/2504.16877v1)

**Authors**: Yixin Yang, Bowen Xu, Xiang Gao, Hailong Sun

**Abstract**: Vulnerability detection is a critical aspect of software security. Accurate detection is essential to prevent potential security breaches and protect software systems from malicious attacks. Recently, vulnerability detection methods leveraging deep learning and large language models (LLMs) have garnered increasing attention. However, existing approaches often focus on analyzing individual files or functions, which limits their ability to gather sufficient contextual information. Analyzing entire repositories to gather context introduces significant noise and computational overhead. To address these challenges, we propose a context-enhanced vulnerability detection approach that combines program analysis with LLMs. Specifically, we use program analysis to extract contextual information at various levels of abstraction, thereby filtering out irrelevant noise. The abstracted context along with source code are provided to LLM for vulnerability detection. We investigate how different levels of contextual granularity improve LLM-based vulnerability detection performance. Our goal is to strike a balance between providing sufficient detail to accurately capture vulnerabilities and minimizing unnecessary complexity that could hinder model performance. Based on an extensive study using GPT-4, DeepSeek, and CodeLLaMA with various prompting strategies, our key findings includes: (1) incorporating abstracted context significantly enhances vulnerability detection effectiveness; (2) different models benefit from distinct levels of abstraction depending on their code understanding capabilities; and (3) capturing program behavior through program analysis for general LLM-based code analysis tasks can be a direction that requires further attention.



## **15. aiXamine: Simplified LLM Safety and Security**

cs.CR

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.14985v2) [paper-pdf](http://arxiv.org/pdf/2504.14985v2)

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.



## **16. Amplified Vulnerabilities: Structured Jailbreak Attacks on LLM-based Multi-Agent Debate**

cs.CR

33 pages, 5 figures

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16489v1) [paper-pdf](http://arxiv.org/pdf/2504.16489v1)

**Authors**: Senmao Qi, Yifei Zou, Peng Li, Ziyi Lin, Xiuzhen Cheng, Dongxiao Yu

**Abstract**: Multi-Agent Debate (MAD), leveraging collaborative interactions among Large Language Models (LLMs), aim to enhance reasoning capabilities in complex tasks. However, the security implications of their iterative dialogues and role-playing characteristics, particularly susceptibility to jailbreak attacks eliciting harmful content, remain critically underexplored. This paper systematically investigates the jailbreak vulnerabilities of four prominent MAD frameworks built upon leading commercial LLMs (GPT-4o, GPT-4, GPT-3.5-turbo, and DeepSeek) without compromising internal agents. We introduce a novel structured prompt-rewriting framework specifically designed to exploit MAD dynamics via narrative encapsulation, role-driven escalation, iterative refinement, and rhetorical obfuscation. Our extensive experiments demonstrate that MAD systems are inherently more vulnerable than single-agent setups. Crucially, our proposed attack methodology significantly amplifies this fragility, increasing average harmfulness from 28.14% to 80.34% and achieving attack success rates as high as 80% in certain scenarios. These findings reveal intrinsic vulnerabilities in MAD architectures and underscore the urgent need for robust, specialized defenses prior to real-world deployment.



## **17. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

cs.CL

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2405.20770v4) [paper-pdf](http://arxiv.org/pdf/2405.20770v4)

**Authors**: Guang Lin, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.



## **18. Attention Tracker: Detecting Prompt Injection Attacks in LLMs**

cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2411.00348v2) [paper-pdf](http://arxiv.org/pdf/2411.00348v2)

**Authors**: Kuo-Han Hung, Ching-Yun Ko, Ambrish Rawat, I-Hsin Chung, Winston H. Hsu, Pin-Yu Chen

**Abstract**: Large Language Models (LLMs) have revolutionized various domains but remain vulnerable to prompt injection attacks, where malicious inputs manipulate the model into ignoring original instructions and executing designated action. In this paper, we investigate the underlying mechanisms of these attacks by analyzing the attention patterns within LLMs. We introduce the concept of the distraction effect, where specific attention heads, termed important heads, shift focus from the original instruction to the injected instruction. Building on this discovery, we propose Attention Tracker, a training-free detection method that tracks attention patterns on instruction to detect prompt injection attacks without the need for additional LLM inference. Our method generalizes effectively across diverse models, datasets, and attack types, showing an AUROC improvement of up to 10.0% over existing methods, and performs well even on small LLMs. We demonstrate the robustness of our approach through extensive evaluations and provide insights into safeguarding LLM-integrated systems from prompt injection vulnerabilities.



## **19. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2408.09093v3) [paper-pdf](http://arxiv.org/pdf/2408.09093v3)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **20. Red Team Diffuser: Exposing Toxic Continuation Vulnerabilities in Vision-Language Models via Reinforcement Learning**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2503.06223v2) [paper-pdf](http://arxiv.org/pdf/2503.06223v2)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The growing deployment of large Vision-Language Models (VLMs) exposes critical safety gaps in their alignment mechanisms. While existing jailbreak studies primarily focus on VLMs' susceptibility to harmful instructions, we reveal a fundamental yet overlooked vulnerability: toxic text continuation, where VLMs produce highly toxic completions when prompted with harmful text prefixes paired with semantically adversarial images. To systematically study this threat, we propose Red Team Diffuser (RTD), the first red teaming diffusion model that coordinates adversarial image generation and toxic continuation through reinforcement learning. Our key innovations include dynamic cross-modal attack and stealth-aware optimization. For toxic text prefixes from an LLM safety benchmark, we conduct greedy search to identify optimal image prompts that maximally induce toxic completions. The discovered image prompts then drive RL-based diffusion model fine-tuning, producing semantically aligned adversarial images that boost toxicity rates. Stealth-aware optimization introduces joint adversarial rewards that balance toxicity maximization (via Detoxify classifier) and stealthiness (via BERTScore), circumventing traditional noise-based adversarial patterns. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% over text-only baselines on the original attack set and 8.91% on an unseen set, proving generalization capability. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. Our findings expose two critical flaws in current VLM alignment: (1) failure to prevent toxic continuation from harmful prefixes, and (2) overlooking cross-modal attack vectors. These results necessitate a paradigm shift toward multimodal red teaming in safety evaluations.



## **21. Research on Cloud Platform Network Traffic Monitoring and Anomaly Detection System based on Large Language Models**

cs.NI

Proceedings of 2025 IEEE 7th International Conference on  Communications, Information System and Computer Engineering (CISCE 2025)

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.17807v1) [paper-pdf](http://arxiv.org/pdf/2504.17807v1)

**Authors**: Ze Yang, Yihong Jin, Juntian Liu, Xinhe Xu, Yihan Zhang, Shuyang Ji

**Abstract**: The rapidly evolving cloud platforms and the escalating complexity of network traffic demand proper network traffic monitoring and anomaly detection to ensure network security and performance. This paper introduces a large language model (LLM)-based network traffic monitoring and anomaly detection system. In addition to existing models such as autoencoders and decision trees, we harness the power of large language models for processing sequence data from network traffic, which allows us a better capture of underlying complex patterns, as well as slight fluctuations in the dataset. We show for a given detection task, the need for a hybrid model that incorporates the attention mechanism of the transformer architecture into a supervised learning framework in order to achieve better accuracy. A pre-trained large language model analyzes and predicts the probable network traffic, and an anomaly detection layer that considers temporality and context is added. Moreover, we present a novel transfer learning-based methodology to enhance the model's effectiveness to quickly adapt to unknown network structures and adversarial conditions without requiring extensive labeled datasets. Actual results show that the designed model outperforms traditional methods in detection accuracy and computational efficiency, effectively identify various network anomalies such as zero-day attacks and traffic congestion pattern, and significantly reduce the false positive rate.



## **22. Exploring the Role of Large Language Models in Cybersecurity: A Systematic Survey**

cs.CR

20 pages, 3 figures

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15622v1) [paper-pdf](http://arxiv.org/pdf/2504.15622v1)

**Authors**: Shuang Tian, Tao Zhang, Jiqiang Liu, Jiacheng Wang, Xuangou Wu, Xiaoqiang Zhu, Ruichen Zhang, Weiting Zhang, Zhenhui Yuan, Shiwen Mao, Dong In Kim

**Abstract**: With the rapid development of technology and the acceleration of digitalisation, the frequency and complexity of cyber security threats are increasing. Traditional cybersecurity approaches, often based on static rules and predefined scenarios, are struggling to adapt to the rapidly evolving nature of modern cyberattacks. There is an urgent need for more adaptive and intelligent defence strategies. The emergence of Large Language Model (LLM) provides an innovative solution to cope with the increasingly severe cyber threats, and its potential in analysing complex attack patterns, predicting threats and assisting real-time response has attracted a lot of attention in the field of cybersecurity, and exploring how to effectively use LLM to defend against cyberattacks has become a hot topic in the current research field. This survey examines the applications of LLM from the perspective of the cyber attack lifecycle, focusing on the three phases of defense reconnaissance, foothold establishment, and lateral movement, and it analyzes the potential of LLMs in Cyber Threat Intelligence (CTI) tasks. Meanwhile, we investigate how LLM-based security solutions are deployed and applied in different network scenarios. It also summarizes the internal and external risk issues faced by LLM during its application. Finally, this survey also points out the facing risk issues and possible future research directions in this domain.



## **23. Diversity Helps Jailbreak Large Language Models**

cs.CL

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2411.04223v2) [paper-pdf](http://arxiv.org/pdf/2411.04223v2)

**Authors**: Weiliang Zhao, Daniel Ben-Levi, Wei Hao, Junfeng Yang, Chengzhi Mao

**Abstract**: We have uncovered a powerful jailbreak technique that leverages large language models' ability to diverge from prior context, enabling them to bypass safety constraints and generate harmful outputs. By simply instructing the LLM to deviate and obfuscate previous attacks, our method dramatically outperforms existing approaches, achieving up to a 62.83% higher success rate in compromising ten leading chatbots, including GPT-4, Gemini, and Llama, while using only 12.9% of the queries. This revelation exposes a critical flaw in current LLM safety training, suggesting that existing methods may merely mask vulnerabilities rather than eliminate them. Our findings sound an urgent alarm for the need to revolutionize testing methodologies to ensure robust and reliable LLM security.



## **24. Trading Devil RL: Backdoor attack via Stock market, Bayesian Optimization and Reinforcement Learning**

cs.LG

End of data poisoning research!: Navier-stokes equations (3D;  update); Reinforcement Learning (RL); HFT (High Frequency Trading); Limit  Order Markets and backdoor attack detection

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2412.17908v3) [paper-pdf](http://arxiv.org/pdf/2412.17908v3)

**Authors**: Orson Mengara

**Abstract**: With the rapid development of generative artificial intelligence, particularly large language models a number of sub-fields of deep learning have made significant progress and are now very useful in everyday applications. For example,financial institutions simulate a wide range of scenarios for various models created by their research teams using reinforcement learning, both before production and after regular operations. In this work, we propose a backdoor attack that focuses solely on data poisoning and a method of detection by dynamic systems and statistical analysis of the distribution of data. This particular backdoor attack is classified as an attack without prior consideration or trigger, and we name it FinanceLLMsBackRL. Our aim is to examine the potential effects of large language models that use reinforcement learning systems for text production or speech recognition, finance, physics, or the ecosystem of contemporary artificial intelligence models.



## **25. LAMD: Context-driven Android Malware Detection and Classification with LLMs**

cs.CR

accepted by 2025 46th IEEE Symposium on Security and Privacy  Workshops (SPW)

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2502.13055v2) [paper-pdf](http://arxiv.org/pdf/2502.13055v2)

**Authors**: Xingzhi Qian, Xinran Zheng, Yiling He, Shuo Yang, Lorenzo Cavallaro

**Abstract**: The rapid growth of mobile applications has escalated Android malware threats. Although there are numerous detection methods, they often struggle with evolving attacks, dataset biases, and limited explainability. Large Language Models (LLMs) offer a promising alternative with their zero-shot inference and reasoning capabilities. However, applying LLMs to Android malware detection presents two key challenges: (1)the extensive support code in Android applications, often spanning thousands of classes, exceeds LLMs' context limits and obscures malicious behavior within benign functionality; (2)the structural complexity and interdependencies of Android applications surpass LLMs' sequence-based reasoning, fragmenting code analysis and hindering malicious intent inference. To address these challenges, we propose LAMD, a practical context-driven framework to enable LLM-based Android malware detection. LAMD integrates key context extraction to isolate security-critical code regions and construct program structures, then applies tier-wise code reasoning to analyze application behavior progressively, from low-level instructions to high-level semantics, providing final prediction and explanation. A well-designed factual consistency verification mechanism is equipped to mitigate LLM hallucinations from the first tier. Evaluation in real-world settings demonstrates LAMD's effectiveness over conventional detectors, establishing a feasible basis for LLM-driven malware analysis in dynamic threat landscapes.



## **26. ASIDE: Architectural Separation of Instructions and Data in Language Models**

cs.LG

ICLR 2025 Workshop on Building Trust in Language Models and  Applications

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2503.10566v2) [paper-pdf](http://arxiv.org/pdf/2503.10566v2)

**Authors**: Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Alexandra Volkova, Soroush Tabesh, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert

**Abstract**: Despite their remarkable performance, large language models lack elementary safety features, and this makes them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as a root cause for the success of prompt injection attacks. In this work, we propose a method, ASIDE, that allows the model to clearly separate between instructions and data on the level of embeddings. ASIDE applies a fixed orthogonal rotation to the embeddings of data tokens, thus creating distinct representations of instructions and data tokens without introducing any additional parameters. We demonstrate the effectiveness of our method by instruct-tuning LLMs with ASIDE and showing (1) highly increased instruction-data separation scores without a loss in model capabilities and (2) competitive results on prompt injection benchmarks, even without dedicated safety training. Additionally, we study the working mechanism behind our method through an analysis of model representations.



## **27. MR. Guard: Multilingual Reasoning Guardrail using Curriculum Learning**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15241v1) [paper-pdf](http://arxiv.org/pdf/2504.15241v1)

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual setting, where multilingual safety-aligned data are often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we propose an approach to build a multilingual guardrail with reasoning. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-guided Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail consistently outperforms recent baselines across both in-domain and out-of-domain languages. The multilingual reasoning capability of our guardrail enables it to generate multilingual explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.



## **28. HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2502.14744v3) [paper-pdf](http://arxiv.org/pdf/2502.14744v3)

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.



## **29. RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15047v1) [paper-pdf](http://arxiv.org/pdf/2504.15047v1)

**Authors**: Quy-Anh Dang, Chris Ngo, Truong-Son Hy

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but are susceptible to adversarial prompts that exploit vulnerabilities to produce unsafe or biased outputs. Existing red-teaming methods often face scalability challenges, resource-intensive requirements, or limited diversity in attack strategies. We propose RainbowPlus, a novel red-teaming framework rooted in evolutionary computation, enhancing adversarial prompt generation through an adaptive quality-diversity (QD) search that extends classical evolutionary algorithms like MAP-Elites with innovations tailored for language models. By employing a multi-element archive to store diverse high-quality prompts and a comprehensive fitness function to evaluate multiple prompts concurrently, RainbowPlus overcomes the constraints of single-prompt archives and pairwise comparisons in prior QD methods like Rainbow Teaming. Experiments comparing RainbowPlus to QD methods across six benchmark datasets and four open-source LLMs demonstrate superior attack success rate (ASR) and diversity (Diverse-Score $\approx 0.84$), generating up to 100 times more unique prompts (e.g., 10,418 vs. 100 for Ministral-8B-Instruct-2410). Against nine state-of-the-art methods on the HarmBench dataset with twelve LLMs (ten open-source, two closed-source), RainbowPlus achieves an average ASR of 81.1%, surpassing AutoDAN-Turbo by 3.9%, and is 9 times faster (1.45 vs. 13.50 hours). Our open-source implementation fosters further advancements in LLM safety, offering a scalable tool for vulnerability assessment. Code and resources are publicly available at https://github.com/knoveleng/rainbowplus, supporting reproducibility and future research in LLM red-teaming.



## **30. Risks of Practicing Large Language Models in Smart Grid: Threat Modeling and Validation**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2405.06237v3) [paper-pdf](http://arxiv.org/pdf/2405.06237v3)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Sun

**Abstract**: Large language models (LLMs) represent significant breakthroughs in artificial intelligence and hold potential for applications within smart grids. However, as demonstrated in previous literature, AI technologies are susceptible to various types of attacks. It is crucial to investigate and evaluate the risks associated with LLMs before deploying them in critical infrastructure like smart grids. In this paper, we systematically evaluated the risks of LLMs and identified two major types of attacks relevant to potential smart grid LLM applications, presenting the corresponding threat models. We validated these attacks using popular LLMs and real smart grid data. Our validation demonstrates that attackers are capable of injecting bad data and retrieving domain knowledge from LLMs employed in different smart grid applications.



## **31. MCGMark: An Encodable and Robust Online Watermark for Tracing LLM-Generated Malicious Code**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2408.01354v2) [paper-pdf](http://arxiv.org/pdf/2408.01354v2)

**Authors**: Kaiwen Ning, Jiachi Chen, Qingyuan Zhong, Tao Zhang, Yanlin Wang, Wei Li, Jingwen Zhang, Jianxing Yu, Yuming Feng, Weizhe Zhang, Zibin Zheng

**Abstract**: With the advent of large language models (LLMs), numerous software service providers (SSPs) are dedicated to developing LLMs customized for code generation tasks, such as CodeLlama and Copilot. However, these LLMs can be leveraged by attackers to create malicious software, which may pose potential threats to the software ecosystem. For example, they can automate the creation of advanced phishing malware. To address this issue, we first conduct an empirical study and design a prompt dataset, MCGTest, which involves approximately 400 person-hours of work and consists of 406 malicious code generation tasks. Utilizing this dataset, we propose MCGMark, the first robust, code structure-aware, and encodable watermarking approach to trace LLM-generated code. We embed encodable information by controlling the token selection and ensuring the output quality based on probabilistic outliers. Additionally, we enhance the robustness of the watermark by considering the structural features of malicious code, preventing the embedding of the watermark in easily modified positions, such as comments. We validate the effectiveness and robustness of MCGMark on the DeepSeek-Coder. MCGMark achieves an embedding success rate of 88.9% within a maximum output limit of 400 tokens. Furthermore, it also demonstrates strong robustness and has minimal impact on the quality of the output code. Our approach assists SSPs in tracing and holding responsible parties accountable for malicious code generated by LLMs.



## **32. BadApex: Backdoor Attack Based on Adaptive Optimization Mechanism of Black-box Large Language Models**

cs.CL

16 pages, 6 figures

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.13775v2) [paper-pdf](http://arxiv.org/pdf/2504.13775v2)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Ziwei Zhang, Yinghan Zhou, Yiming Xue

**Abstract**: Previous insertion-based and paraphrase-based backdoors have achieved great success in attack efficacy, but they ignore the text quality and semantic consistency between poisoned and clean texts. Although recent studies introduce LLMs to generate poisoned texts and improve the stealthiness, semantic consistency, and text quality, their hand-crafted prompts rely on expert experiences, facing significant challenges in prompt adaptability and attack performance after defenses. In this paper, we propose a novel backdoor attack based on adaptive optimization mechanism of black-box large language models (BadApex), which leverages a black-box LLM to generate poisoned text through a refined prompt. Specifically, an Adaptive Optimization Mechanism is designed to refine an initial prompt iteratively using the generation and modification agents. The generation agent generates the poisoned text based on the initial prompt. Then the modification agent evaluates the quality of the poisoned text and refines a new prompt. After several iterations of the above process, the refined prompt is used to generate poisoned texts through LLMs. We conduct extensive experiments on three dataset with six backdoor attacks and two defenses. Extensive experimental results demonstrate that BadApex significantly outperforms state-of-the-art attacks. It improves prompt adaptability, semantic consistency, and text quality. Furthermore, when two defense methods are applied, the average attack success rate (ASR) still up to 96.75%.



## **33. Detecting Training Data of Large Language Models via Expectation Maximization**

cs.CL

15 pages

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.07582v2) [paper-pdf](http://arxiv.org/pdf/2410.07582v2)

**Authors**: Gyuwan Kim, Yang Li, Evangelia Spiliopoulou, Jie Ma, Miguel Ballesteros, William Yang Wang

**Abstract**: The advancement of large language models has grown parallel to the opacity of their training data. Membership inference attacks (MIAs) aim to determine whether specific data was used to train a model. They offer valuable insights into detecting data contamination and ensuring compliance with privacy and copyright standards. However, MIA for LLMs is challenging due to the massive scale of training data and the inherent ambiguity of membership in texts. Moreover, creating realistic MIA evaluation benchmarks is difficult as training and test data distributions are often unknown. We introduce EM-MIA, a novel membership inference method that iteratively refines membership scores and prefix scores via an expectation-maximization algorithm. Our approach leverages the observation that these scores can improve each other: membership scores help identify effective prefixes for detecting training data, while prefix scores help determine membership. As a result, EM-MIA achieves state-of-the-art results on WikiMIA. To enable comprehensive evaluation, we introduce OLMoMIA, a benchmark built from OLMo resources, which allows controlling task difficulty through varying degrees of overlap between training and test data distributions. Our experiments demonstrate EM-MIA is robust across different scenarios while also revealing fundamental limitations of current MIA approaches when member and non-member distributions are nearly identical.



## **34. Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2503.15547v2) [paper-pdf](http://arxiv.org/pdf/2503.15547v2)

**Authors**: Juhee Kim, Woohyuk Choi, Byoungyoung Lee

**Abstract**: Large Language Models (LLMs) are combined with tools to create powerful LLM agents that provide a wide range of services. Unlike traditional software, LLM agent's behavior is determined at runtime by natural language prompts from either user or tool's data. This flexibility enables a new computing paradigm with unlimited capabilities and programmability, but also introduces new security risks, vulnerable to privilege escalation attacks. Moreover, user prompts are prone to be interpreted in an insecure way by LLM agents, creating non-deterministic behaviors that can be exploited by attackers. To address these security risks, we propose Prompt Flow Integrity (PFI), a system security-oriented solution to prevent privilege escalation in LLM agents. Analyzing the architectural characteristics of LLM agents, PFI features three mitigation techniques -- i.e., agent isolation, secure untrusted data processing, and privilege escalation guardrails. Our evaluation result shows that PFI effectively mitigates privilege escalation attacks while successfully preserving the utility of LLM agents.



## **35. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

cs.SE

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2411.10565v2) [paper-pdf](http://arxiv.org/pdf/2411.10565v2)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.



## **36. REDEditing: Relationship-Driven Precise Backdoor Poisoning on Text-to-Image Diffusion Models**

cs.CR

10 pages, 7 figures

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2504.14554v1) [paper-pdf](http://arxiv.org/pdf/2504.14554v1)

**Authors**: Chongye Guo, Jinhu Fu, Junfeng Fang, Kun Wang, Guorui Feng

**Abstract**: The rapid advancement of generative AI highlights the importance of text-to-image (T2I) security, particularly with the threat of backdoor poisoning. Timely disclosure and mitigation of security vulnerabilities in T2I models are crucial for ensuring the safe deployment of generative models. We explore a novel training-free backdoor poisoning paradigm through model editing, which is recently employed for knowledge updating in large language models. Nevertheless, we reveal the potential security risks posed by model editing techniques to image generation models. In this work, we establish the principles for backdoor attacks based on model editing, and propose a relationship-driven precise backdoor poisoning method, REDEditing. Drawing on the principles of equivalent-attribute alignment and stealthy poisoning, we develop an equivalent relationship retrieval and joint-attribute transfer approach that ensures consistent backdoor image generation through concept rebinding. A knowledge isolation constraint is proposed to preserve benign generation integrity. Our method achieves an 11\% higher attack success rate compared to state-of-the-art approaches. Remarkably, adding just one line of code enhances output naturalness while improving backdoor stealthiness by 24\%. This work aims to heighten awareness regarding this security vulnerability in editable image generation models.



## **37. Breaking the Prompt Wall (I): A Real-World Case Study of Attacking ChatGPT via Lightweight Prompt Injection**

cs.CR

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2504.16125v1) [paper-pdf](http://arxiv.org/pdf/2504.16125v1)

**Authors**: Xiangyu Chang, Guang Dai, Hao Di, Haishan Ye

**Abstract**: This report presents a real-world case study demonstrating how prompt injection can attack large language model platforms such as ChatGPT according to a proposed injection framework. By providing three real-world examples, we show how adversarial prompts can be injected via user inputs, web-based retrieval, and system-level agent instructions. These attacks, though lightweight and low-cost, can cause persistent and misleading behaviors in LLM outputs. Our case study reveals that even commercial-grade LLMs remain vulnerable to subtle manipulations that bypass safety filters and influence user decisions. \textbf{More importantly, we stress that this report is not intended as an attack guide, but as a technical alert. As ethical researchers, we aim to raise awareness and call upon developers, especially those at OpenAI, to treat prompt-level security as a critical design priority.



## **38. Reconstruction of Differentially Private Text Sanitization via Large Language Models**

cs.CR

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2410.12443v2) [paper-pdf](http://arxiv.org/pdf/2410.12443v2)

**Authors**: Shuchao Pang, Zhigang Lu, Haichen Wang, Peng Fu, Yongbin Zhou, Minhui Xue

**Abstract**: Differential privacy (DP) is the de facto privacy standard against privacy leakage attacks, including many recently discovered ones against large language models (LLMs). However, we discovered that LLMs could reconstruct the altered/removed privacy from given DP-sanitized prompts. We propose two attacks (black-box and white-box) based on the accessibility to LLMs and show that LLMs could connect the pair of DP-sanitized text and the corresponding private training data of LLMs by giving sample text pairs as instructions (in the black-box attacks) or fine-tuning data (in the white-box attacks). To illustrate our findings, we conduct comprehensive experiments on modern LLMs (e.g., LLaMA-2, LLaMA-3, ChatGPT-3.5, ChatGPT-4, ChatGPT-4o, Claude-3, Claude-3.5, OPT, GPT-Neo, GPT-J, Gemma-2, and Pythia) using commonly used datasets (such as WikiMIA, Pile-CC, and Pile-Wiki) against both word-level and sentence-level DP. The experimental results show promising recovery rates, e.g., the black-box attacks against the word-level DP over WikiMIA dataset gave 72.18% on LLaMA-2 (70B), 82.39% on LLaMA-3 (70B), 75.35% on Gemma-2, 91.2% on ChatGPT-4o, and 94.01% on Claude-3.5 (Sonnet). More urgently, this study indicates that these well-known LLMs have emerged as a new security risk for existing DP text sanitization approaches in the current environment.



## **39. SHIELD : An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models**

cs.CV

Accepted by Visual Intelligence

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2402.04178v2) [paper-pdf](http://arxiv.org/pdf/2402.04178v2)

**Authors**: Yichen Shi, Yuhao Gao, Yingxin Lai, Hongyang Wang, Jun Feng, Lei He, Jun Wan, Changsheng Chen, Zitong Yu, Xiaochun Cao

**Abstract**: Multimodal large language models (MLLMs) have demonstrated strong capabilities in vision-related tasks, capitalizing on their visual semantic comprehension and reasoning capabilities. However, their ability to detect subtle visual spoofing and forgery clues in face attack detection tasks remains underexplored. In this paper, we introduce a benchmark, SHIELD, to evaluate MLLMs for face spoofing and forgery detection. Specifically, we design true/false and multiple-choice questions to assess MLLM performance on multimodal face data across two tasks. For the face anti-spoofing task, we evaluate three modalities (i.e., RGB, infrared, and depth) under six attack types. For the face forgery detection task, we evaluate GAN-based and diffusion-based data, incorporating visual and acoustic modalities. We conduct zero-shot and few-shot evaluations in standard and chain of thought (COT) settings. Additionally, we propose a novel multi-attribute chain of thought (MA-COT) paradigm for describing and judging various task-specific and task-irrelevant attributes of face images. The findings of this study demonstrate that MLLMs exhibit strong potential for addressing the challenges associated with the security of facial recognition technology applications.



## **40. Jailbreaking as a Reward Misspecification Problem**

cs.LG

Accepted to ICLR 2025. Code:  https://github.com/zhxieml/remiss-jailbreak

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2406.14393v5) [paper-pdf](http://arxiv.org/pdf/2406.14393v5)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.



## **41. Multi-Stage Retrieval for Operational Technology Cybersecurity Compliance Using Large Language Models: A Railway Casestudy**

cs.AI

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.14044v1) [paper-pdf](http://arxiv.org/pdf/2504.14044v1)

**Authors**: Regan Bolton, Mohammadreza Sheikhfathollahi, Simon Parkinson, Dan Basher, Howard Parkinson

**Abstract**: Operational Technology Cybersecurity (OTCS) continues to be a dominant challenge for critical infrastructure such as railways. As these systems become increasingly vulnerable to malicious attacks due to digitalization, effective documentation and compliance processes are essential to protect these safety-critical systems. This paper proposes a novel system that leverages Large Language Models (LLMs) and multi-stage retrieval to enhance the compliance verification process against standards like IEC 62443 and the rail-specific IEC 63452. We first evaluate a Baseline Compliance Architecture (BCA) for answering OTCS compliance queries, then develop an extended approach called Parallel Compliance Architecture (PCA) that incorporates additional context from regulatory standards. Through empirical evaluation comparing OpenAI-gpt-4o and Claude-3.5-haiku models in these architectures, we demonstrate that the PCA significantly improves both correctness and reasoning quality in compliance verification. Our research establishes metrics for response correctness, logical reasoning, and hallucination detection, highlighting the strengths and limitations of using LLMs for compliance verification in railway cybersecurity. The results suggest that retrieval-augmented approaches can significantly improve the efficiency and accuracy of compliance assessments, particularly valuable in an industry facing a shortage of cybersecurity expertise.



## **42. Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?**

cs.SE

The paper has been peer-reviewed and accepted for publication to the  29th International Conference on Evaluation and Assessment in Software  Engineering (EASE 2025)

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13769v1) [paper-pdf](http://arxiv.org/pdf/2504.13769v1)

**Authors**: Motunrayo Ibiyo, Thinakone Louangdy, Phuong T. Nguyen, Claudio Di Sipio, Davide Di Ruscio

**Abstract**: Malicious software packages in open-source ecosystems, such as PyPI, pose growing security risks. Unlike traditional vulnerabilities, these packages are intentionally designed to deceive users, making detection challenging due to evolving attack methods and the lack of structured datasets. In this work, we empirically evaluate the effectiveness of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and few-shot learning for detecting malicious source code. We fine-tune LLMs on curated datasets and integrate YARA rules, GitHub Security Advisories, and malicious code snippets with the aim of enhancing classification accuracy. We came across a counterintuitive outcome: While RAG is expected to boost up the prediction performance, it fails in the performed evaluation, obtaining a mediocre accuracy. In contrast, few-shot learning is more effective as it significantly improves the detection of malicious code, achieving 97% accuracy and 95% balanced accuracy, outperforming traditional RAG approaches. Thus, future work should expand structured knowledge bases, refine retrieval models, and explore hybrid AI-driven cybersecurity solutions.



## **43. DETAM: Defending LLMs Against Jailbreak Attacks via Targeted Attention Modification**

cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13562v1) [paper-pdf](http://arxiv.org/pdf/2504.13562v1)

**Authors**: Yu Li, Han Jiang, Zhihua Wei

**Abstract**: With the widespread adoption of Large Language Models (LLMs), jailbreak attacks have become an increasingly pressing safety concern. While safety-aligned LLMs can effectively defend against normal harmful queries, they remain vulnerable to such attacks. Existing defense methods primarily rely on fine-tuning or input modification, which often suffer from limited generalization and reduced utility. To address this, we introduce DETAM, a finetuning-free defense approach that improves the defensive capabilities against jailbreak attacks of LLMs via targeted attention modification. Specifically, we analyze the differences in attention scores between successful and unsuccessful defenses to identify the attention heads sensitive to jailbreak attacks. During inference, we reallocate attention to emphasize the user's core intention, minimizing interference from attack tokens. Our experimental results demonstrate that DETAM outperforms various baselines in jailbreak defense and exhibits robust generalization across different attacks and models, maintaining its effectiveness even on in-the-wild jailbreak data. Furthermore, in evaluating the model's utility, we incorporated over-defense datasets, which further validate the superior performance of our approach. The code will be released immediately upon acceptance.



## **44. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

cs.CL

WWW'25 research track accepted

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2406.11260v3) [paper-pdf](http://arxiv.org/pdf/2406.11260v3)

**Authors**: Sungwon Park, Sungwon Han, Xing Xie, Jae-Gil Lee, Meeyoung Cha

**Abstract**: The spread of fake news harms individuals and presents a critical social challenge that must be addressed. Although numerous algorithmic and insightful features have been developed to detect fake news, many of these features can be manipulated with style-conversion attacks, especially with the emergence of advanced language models, making it more difficult to differentiate from genuine news. This study proposes adversarial style augmentation, AdStyle, designed to train a fake news detector that remains robust against various style-conversion attacks. The primary mechanism involves the strategic use of LLMs to automatically generate a diverse and coherent array of style-conversion attack prompts, enhancing the generation of particularly challenging prompts for the detector. Experiments indicate that our augmentation strategy significantly improves robustness and detection performance when evaluated on fake news benchmark datasets.



## **45. Large Language Models for Validating Network Protocol Parsers**

cs.SE

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13515v1) [paper-pdf](http://arxiv.org/pdf/2504.13515v1)

**Authors**: Mingwei Zheng, Danning Xie, Xiangyu Zhang

**Abstract**: Network protocol parsers are essential for enabling correct and secure communication between devices. Bugs in these parsers can introduce critical vulnerabilities, including memory corruption, information leakage, and denial-of-service attacks. An intuitive way to assess parser correctness is to compare the implementation with its official protocol standard. However, this comparison is challenging because protocol standards are typically written in natural language, whereas implementations are in source code. Existing methods like model checking, fuzzing, and differential testing have been used to find parsing bugs, but they either require significant manual effort or ignore the protocol standards, limiting their ability to detect semantic violations. To enable more automated validation of parser implementations against protocol standards, we propose PARVAL, a multi-agent framework built on large language models (LLMs). PARVAL leverages the capabilities of LLMs to understand both natural language and code. It transforms both protocol standards and their implementations into a unified intermediate representation, referred to as format specifications, and performs a differential comparison to uncover inconsistencies. We evaluate PARVAL on the Bidirectional Forwarding Detection (BFD) protocol. Our experiments demonstrate that PARVAL successfully identifies inconsistencies between the implementation and its RFC standard, achieving a low false positive rate of 5.6%. PARVAL uncovers seven unique bugs, including five previously unknown issues.



## **46. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.05050v2) [paper-pdf](http://arxiv.org/pdf/2504.05050v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.



## **47. GraphQLer: Enhancing GraphQL Security with Context-Aware API Testing**

cs.CR

Publicly available on: https://github.com/omar2535/GraphQLer

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13358v1) [paper-pdf](http://arxiv.org/pdf/2504.13358v1)

**Authors**: Omar Tsai, Jianing Li, Tsz Tung Cheung, Lejing Huang, Hao Zhu, Jianrui Xiao, Iman Sharafaldin, Mohammad A. Tayebi

**Abstract**: GraphQL is an open-source data query and manipulation language for web applications, offering a flexible alternative to RESTful APIs. However, its dynamic execution model and lack of built-in security mechanisms expose it to vulnerabilities such as unauthorized data access, denial-of-service (DoS) attacks, and injections. Existing testing tools focus on functional correctness, often overlooking security risks stemming from query interdependencies and execution context. This paper presents GraphQLer, the first context-aware security testing framework for GraphQL APIs. GraphQLer constructs a dependency graph to analyze relationships among mutations, queries, and objects, capturing critical interdependencies. It chains related queries and mutations to reveal authentication and authorization flaws, access control bypasses, and resource misuse. Additionally, GraphQLer tracks internal resource usage to uncover data leakage, privilege escalation, and replay attack vectors. We assess GraphQLer on various GraphQL APIs, demonstrating improved testing coverage - averaging a 35% increase, with up to 84% in some cases - compared to top-performing baselines. Remarkably, this is achieved in less time, making GraphQLer suitable for time-sensitive contexts. GraphQLer also successfully detects a known CVE and potential vulnerabilities in large-scale production APIs. These results underline GraphQLer's utility in proactively securing GraphQL APIs through automated, context-aware vulnerability detection.



## **48. GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms**

cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13052v1) [paper-pdf](http://arxiv.org/pdf/2504.13052v1)

**Authors**: Sinan He, An Wang

**Abstract**: Large Language Models (LLMs) have been equipped with safety mechanisms to prevent harmful outputs, but these guardrails can often be bypassed through "jailbreak" prompts. This paper introduces a novel graph-based approach to systematically generate jailbreak prompts through semantic transformations. We represent malicious prompts as nodes in a graph structure with edges denoting different transformations, leveraging Abstract Meaning Representation (AMR) and Resource Description Framework (RDF) to parse user goals into semantic components that can be manipulated to evade safety filters. We demonstrate a particularly effective exploitation vector by instructing LLMs to generate code that realizes the intent described in these semantic graphs, achieving success rates of up to 87% against leading commercial LLMs. Our analysis reveals that contextual framing and abstraction are particularly effective at circumventing safety measures, highlighting critical gaps in current safety alignment techniques that focus primarily on surface-level patterns. These findings provide insights for developing more robust safeguards against structured semantic attacks. Our research contributes both a theoretical framework and practical methodology for systematically stress-testing LLM safety mechanisms.



## **49. From Sands to Mansions: Towards Automated Cyberattack Emulation with Classical Planning and Large Language Models**

cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2407.16928v3) [paper-pdf](http://arxiv.org/pdf/2407.16928v3)

**Authors**: Lingzhi Wang, Zhenyuan Li, Yi Jiang, Zhengkai Wang, Zonghan Guo, Jiahui Wang, Yangyang Wei, Xiangmin Shen, Wei Ruan, Yan Chen

**Abstract**: As attackers continually advance their tools, skills, and techniques during cyberattacks - particularly in modern Advanced Persistence Threats (APT) campaigns - there is a pressing need for a comprehensive and up-to-date cyberattack dataset to support threat-informed defense and enable benchmarking of defense systems in both academia and commercial solutions. However, there is a noticeable scarcity of cyberattack datasets: recent academic studies continue to rely on outdated benchmarks, while cyberattack emulation in industry remains limited due to the significant human effort and expertise required. Creating datasets by emulating advanced cyberattacks presents several challenges, such as limited coverage of attack techniques, the complexity of chaining multiple attack steps, and the difficulty of realistically mimicking actual threat groups. In this paper, we introduce modularized Attack Action and Attack Action Linking Model as a structured way to organizing and chaining individual attack steps into multi-step cyberattacks. Building on this, we propose Aurora, a system that autonomously emulates cyberattacks using third-party attack tools and threat intelligence reports with the help of classical planning and large language models. Aurora can automatically generate detailed attack plans, set up emulation environments, and semi-automatically execute the attacks. We utilize Aurora to create a dataset containing over 1,000 attack chains. To our best knowledge, Aurora is the only system capable of automatically constructing such a large-scale cyberattack dataset with corresponding attack execution scripts and environments. Our evaluation further demonstrates that Aurora outperforms the previous similar work and even the most advanced generative AI models in cyberattack emulation. To support further research, we published the cyberattack dataset and will publish the source code of Aurora.



## **50. ControlNET: A Firewall for RAG-based LLM System**

cs.CR

Project Page: https://ai.zjuicsr.cn/firewall

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.09593v2) [paper-pdf](http://arxiv.org/pdf/2504.09593v2)

**Authors**: Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, Zhan Qin

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.



