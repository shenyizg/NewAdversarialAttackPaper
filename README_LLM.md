# Latest Large Language Model Attack Papers
**update at 2025-04-23 10:03:56**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2408.09093v3) [paper-pdf](http://arxiv.org/pdf/2408.09093v3)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **2. Red Team Diffuser: Exposing Toxic Continuation Vulnerabilities in Vision-Language Models via Reinforcement Learning**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2503.06223v2) [paper-pdf](http://arxiv.org/pdf/2503.06223v2)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The growing deployment of large Vision-Language Models (VLMs) exposes critical safety gaps in their alignment mechanisms. While existing jailbreak studies primarily focus on VLMs' susceptibility to harmful instructions, we reveal a fundamental yet overlooked vulnerability: toxic text continuation, where VLMs produce highly toxic completions when prompted with harmful text prefixes paired with semantically adversarial images. To systematically study this threat, we propose Red Team Diffuser (RTD), the first red teaming diffusion model that coordinates adversarial image generation and toxic continuation through reinforcement learning. Our key innovations include dynamic cross-modal attack and stealth-aware optimization. For toxic text prefixes from an LLM safety benchmark, we conduct greedy search to identify optimal image prompts that maximally induce toxic completions. The discovered image prompts then drive RL-based diffusion model fine-tuning, producing semantically aligned adversarial images that boost toxicity rates. Stealth-aware optimization introduces joint adversarial rewards that balance toxicity maximization (via Detoxify classifier) and stealthiness (via BERTScore), circumventing traditional noise-based adversarial patterns. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% over text-only baselines on the original attack set and 8.91% on an unseen set, proving generalization capability. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. Our findings expose two critical flaws in current VLM alignment: (1) failure to prevent toxic continuation from harmful prefixes, and (2) overlooking cross-modal attack vectors. These results necessitate a paradigm shift toward multimodal red teaming in safety evaluations.



## **3. Exploring the Role of Large Language Models in Cybersecurity: A Systematic Survey**

cs.CR

20 pages, 3 figures

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15622v1) [paper-pdf](http://arxiv.org/pdf/2504.15622v1)

**Authors**: Shuang Tian, Tao Zhang, Jiqiang Liu, Jiacheng Wang, Xuangou Wu, Xiaoqiang Zhu, Ruichen Zhang, Weiting Zhang, Zhenhui Yuan, Shiwen Mao, Dong In Kim

**Abstract**: With the rapid development of technology and the acceleration of digitalisation, the frequency and complexity of cyber security threats are increasing. Traditional cybersecurity approaches, often based on static rules and predefined scenarios, are struggling to adapt to the rapidly evolving nature of modern cyberattacks. There is an urgent need for more adaptive and intelligent defence strategies. The emergence of Large Language Model (LLM) provides an innovative solution to cope with the increasingly severe cyber threats, and its potential in analysing complex attack patterns, predicting threats and assisting real-time response has attracted a lot of attention in the field of cybersecurity, and exploring how to effectively use LLM to defend against cyberattacks has become a hot topic in the current research field. This survey examines the applications of LLM from the perspective of the cyber attack lifecycle, focusing on the three phases of defense reconnaissance, foothold establishment, and lateral movement, and it analyzes the potential of LLMs in Cyber Threat Intelligence (CTI) tasks. Meanwhile, we investigate how LLM-based security solutions are deployed and applied in different network scenarios. It also summarizes the internal and external risk issues faced by LLM during its application. Finally, this survey also points out the facing risk issues and possible future research directions in this domain.



## **4. Diversity Helps Jailbreak Large Language Models**

cs.CL

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2411.04223v2) [paper-pdf](http://arxiv.org/pdf/2411.04223v2)

**Authors**: Weiliang Zhao, Daniel Ben-Levi, Wei Hao, Junfeng Yang, Chengzhi Mao

**Abstract**: We have uncovered a powerful jailbreak technique that leverages large language models' ability to diverge from prior context, enabling them to bypass safety constraints and generate harmful outputs. By simply instructing the LLM to deviate and obfuscate previous attacks, our method dramatically outperforms existing approaches, achieving up to a 62.83% higher success rate in compromising ten leading chatbots, including GPT-4, Gemini, and Llama, while using only 12.9% of the queries. This revelation exposes a critical flaw in current LLM safety training, suggesting that existing methods may merely mask vulnerabilities rather than eliminate them. Our findings sound an urgent alarm for the need to revolutionize testing methodologies to ensure robust and reliable LLM security.



## **5. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.14348v2) [paper-pdf](http://arxiv.org/pdf/2504.14348v2)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.



## **6. Trading Devil RL: Backdoor attack via Stock market, Bayesian Optimization and Reinforcement Learning**

cs.LG

End of data poisoning research!: Navier-stokes equations (3D;  update); Reinforcement Learning (RL); HFT (High Frequency Trading); Limit  Order Markets and backdoor attack detection

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2412.17908v3) [paper-pdf](http://arxiv.org/pdf/2412.17908v3)

**Authors**: Orson Mengara

**Abstract**: With the rapid development of generative artificial intelligence, particularly large language models a number of sub-fields of deep learning have made significant progress and are now very useful in everyday applications. For example,financial institutions simulate a wide range of scenarios for various models created by their research teams using reinforcement learning, both before production and after regular operations. In this work, we propose a backdoor attack that focuses solely on data poisoning and a method of detection by dynamic systems and statistical analysis of the distribution of data. This particular backdoor attack is classified as an attack without prior consideration or trigger, and we name it FinanceLLMsBackRL. Our aim is to examine the potential effects of large language models that use reinforcement learning systems for text production or speech recognition, finance, physics, or the ecosystem of contemporary artificial intelligence models.



## **7. LAMD: Context-driven Android Malware Detection and Classification with LLMs**

cs.CR

accepted by 2025 46th IEEE Symposium on Security and Privacy  Workshops (SPW)

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2502.13055v2) [paper-pdf](http://arxiv.org/pdf/2502.13055v2)

**Authors**: Xingzhi Qian, Xinran Zheng, Yiling He, Shuo Yang, Lorenzo Cavallaro

**Abstract**: The rapid growth of mobile applications has escalated Android malware threats. Although there are numerous detection methods, they often struggle with evolving attacks, dataset biases, and limited explainability. Large Language Models (LLMs) offer a promising alternative with their zero-shot inference and reasoning capabilities. However, applying LLMs to Android malware detection presents two key challenges: (1)the extensive support code in Android applications, often spanning thousands of classes, exceeds LLMs' context limits and obscures malicious behavior within benign functionality; (2)the structural complexity and interdependencies of Android applications surpass LLMs' sequence-based reasoning, fragmenting code analysis and hindering malicious intent inference. To address these challenges, we propose LAMD, a practical context-driven framework to enable LLM-based Android malware detection. LAMD integrates key context extraction to isolate security-critical code regions and construct program structures, then applies tier-wise code reasoning to analyze application behavior progressively, from low-level instructions to high-level semantics, providing final prediction and explanation. A well-designed factual consistency verification mechanism is equipped to mitigate LLM hallucinations from the first tier. Evaluation in real-world settings demonstrates LAMD's effectiveness over conventional detectors, establishing a feasible basis for LLM-driven malware analysis in dynamic threat landscapes.



## **8. ASIDE: Architectural Separation of Instructions and Data in Language Models**

cs.LG

ICLR 2025 Workshop on Building Trust in Language Models and  Applications

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2503.10566v2) [paper-pdf](http://arxiv.org/pdf/2503.10566v2)

**Authors**: Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Alexandra Volkova, Soroush Tabesh, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert

**Abstract**: Despite their remarkable performance, large language models lack elementary safety features, and this makes them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as a root cause for the success of prompt injection attacks. In this work, we propose a method, ASIDE, that allows the model to clearly separate between instructions and data on the level of embeddings. ASIDE applies a fixed orthogonal rotation to the embeddings of data tokens, thus creating distinct representations of instructions and data tokens without introducing any additional parameters. We demonstrate the effectiveness of our method by instruct-tuning LLMs with ASIDE and showing (1) highly increased instruction-data separation scores without a loss in model capabilities and (2) competitive results on prompt injection benchmarks, even without dedicated safety training. Additionally, we study the working mechanism behind our method through an analysis of model representations.



## **9. MR. Guard: Multilingual Reasoning Guardrail using Curriculum Learning**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15241v1) [paper-pdf](http://arxiv.org/pdf/2504.15241v1)

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual setting, where multilingual safety-aligned data are often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we propose an approach to build a multilingual guardrail with reasoning. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-guided Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail consistently outperforms recent baselines across both in-domain and out-of-domain languages. The multilingual reasoning capability of our guardrail enables it to generate multilingual explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.



## **10. HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2502.14744v3) [paper-pdf](http://arxiv.org/pdf/2502.14744v3)

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.



## **11. RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15047v1) [paper-pdf](http://arxiv.org/pdf/2504.15047v1)

**Authors**: Quy-Anh Dang, Chris Ngo, Truong-Son Hy

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but are susceptible to adversarial prompts that exploit vulnerabilities to produce unsafe or biased outputs. Existing red-teaming methods often face scalability challenges, resource-intensive requirements, or limited diversity in attack strategies. We propose RainbowPlus, a novel red-teaming framework rooted in evolutionary computation, enhancing adversarial prompt generation through an adaptive quality-diversity (QD) search that extends classical evolutionary algorithms like MAP-Elites with innovations tailored for language models. By employing a multi-element archive to store diverse high-quality prompts and a comprehensive fitness function to evaluate multiple prompts concurrently, RainbowPlus overcomes the constraints of single-prompt archives and pairwise comparisons in prior QD methods like Rainbow Teaming. Experiments comparing RainbowPlus to QD methods across six benchmark datasets and four open-source LLMs demonstrate superior attack success rate (ASR) and diversity (Diverse-Score $\approx 0.84$), generating up to 100 times more unique prompts (e.g., 10,418 vs. 100 for Ministral-8B-Instruct-2410). Against nine state-of-the-art methods on the HarmBench dataset with twelve LLMs (ten open-source, two closed-source), RainbowPlus achieves an average ASR of 81.1%, surpassing AutoDAN-Turbo by 3.9%, and is 9 times faster (1.45 vs. 13.50 hours). Our open-source implementation fosters further advancements in LLM safety, offering a scalable tool for vulnerability assessment. Code and resources are publicly available at https://github.com/knoveleng/rainbowplus, supporting reproducibility and future research in LLM red-teaming.



## **12. aiXamine: LLM Safety and Security Simplified**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.14985v1) [paper-pdf](http://arxiv.org/pdf/2504.14985v1)

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.



## **13. Risks of Practicing Large Language Models in Smart Grid: Threat Modeling and Validation**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2405.06237v3) [paper-pdf](http://arxiv.org/pdf/2405.06237v3)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Sun

**Abstract**: Large language models (LLMs) represent significant breakthroughs in artificial intelligence and hold potential for applications within smart grids. However, as demonstrated in previous literature, AI technologies are susceptible to various types of attacks. It is crucial to investigate and evaluate the risks associated with LLMs before deploying them in critical infrastructure like smart grids. In this paper, we systematically evaluated the risks of LLMs and identified two major types of attacks relevant to potential smart grid LLM applications, presenting the corresponding threat models. We validated these attacks using popular LLMs and real smart grid data. Our validation demonstrates that attackers are capable of injecting bad data and retrieving domain knowledge from LLMs employed in different smart grid applications.



## **14. MCGMark: An Encodable and Robust Online Watermark for Tracing LLM-Generated Malicious Code**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2408.01354v2) [paper-pdf](http://arxiv.org/pdf/2408.01354v2)

**Authors**: Kaiwen Ning, Jiachi Chen, Qingyuan Zhong, Tao Zhang, Yanlin Wang, Wei Li, Jingwen Zhang, Jianxing Yu, Yuming Feng, Weizhe Zhang, Zibin Zheng

**Abstract**: With the advent of large language models (LLMs), numerous software service providers (SSPs) are dedicated to developing LLMs customized for code generation tasks, such as CodeLlama and Copilot. However, these LLMs can be leveraged by attackers to create malicious software, which may pose potential threats to the software ecosystem. For example, they can automate the creation of advanced phishing malware. To address this issue, we first conduct an empirical study and design a prompt dataset, MCGTest, which involves approximately 400 person-hours of work and consists of 406 malicious code generation tasks. Utilizing this dataset, we propose MCGMark, the first robust, code structure-aware, and encodable watermarking approach to trace LLM-generated code. We embed encodable information by controlling the token selection and ensuring the output quality based on probabilistic outliers. Additionally, we enhance the robustness of the watermark by considering the structural features of malicious code, preventing the embedding of the watermark in easily modified positions, such as comments. We validate the effectiveness and robustness of MCGMark on the DeepSeek-Coder. MCGMark achieves an embedding success rate of 88.9% within a maximum output limit of 400 tokens. Furthermore, it also demonstrates strong robustness and has minimal impact on the quality of the output code. Our approach assists SSPs in tracing and holding responsible parties accountable for malicious code generated by LLMs.



## **15. BadApex: Backdoor Attack Based on Adaptive Optimization Mechanism of Black-box Large Language Models**

cs.CL

16 pages, 6 figures

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.13775v2) [paper-pdf](http://arxiv.org/pdf/2504.13775v2)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Ziwei Zhang, Yinghan Zhou, Yiming Xue

**Abstract**: Previous insertion-based and paraphrase-based backdoors have achieved great success in attack efficacy, but they ignore the text quality and semantic consistency between poisoned and clean texts. Although recent studies introduce LLMs to generate poisoned texts and improve the stealthiness, semantic consistency, and text quality, their hand-crafted prompts rely on expert experiences, facing significant challenges in prompt adaptability and attack performance after defenses. In this paper, we propose a novel backdoor attack based on adaptive optimization mechanism of black-box large language models (BadApex), which leverages a black-box LLM to generate poisoned text through a refined prompt. Specifically, an Adaptive Optimization Mechanism is designed to refine an initial prompt iteratively using the generation and modification agents. The generation agent generates the poisoned text based on the initial prompt. Then the modification agent evaluates the quality of the poisoned text and refines a new prompt. After several iterations of the above process, the refined prompt is used to generate poisoned texts through LLMs. We conduct extensive experiments on three dataset with six backdoor attacks and two defenses. Extensive experimental results demonstrate that BadApex significantly outperforms state-of-the-art attacks. It improves prompt adaptability, semantic consistency, and text quality. Furthermore, when two defense methods are applied, the average attack success rate (ASR) still up to 96.75%.



## **16. Detecting Training Data of Large Language Models via Expectation Maximization**

cs.CL

15 pages

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.07582v2) [paper-pdf](http://arxiv.org/pdf/2410.07582v2)

**Authors**: Gyuwan Kim, Yang Li, Evangelia Spiliopoulou, Jie Ma, Miguel Ballesteros, William Yang Wang

**Abstract**: The advancement of large language models has grown parallel to the opacity of their training data. Membership inference attacks (MIAs) aim to determine whether specific data was used to train a model. They offer valuable insights into detecting data contamination and ensuring compliance with privacy and copyright standards. However, MIA for LLMs is challenging due to the massive scale of training data and the inherent ambiguity of membership in texts. Moreover, creating realistic MIA evaluation benchmarks is difficult as training and test data distributions are often unknown. We introduce EM-MIA, a novel membership inference method that iteratively refines membership scores and prefix scores via an expectation-maximization algorithm. Our approach leverages the observation that these scores can improve each other: membership scores help identify effective prefixes for detecting training data, while prefix scores help determine membership. As a result, EM-MIA achieves state-of-the-art results on WikiMIA. To enable comprehensive evaluation, we introduce OLMoMIA, a benchmark built from OLMo resources, which allows controlling task difficulty through varying degrees of overlap between training and test data distributions. Our experiments demonstrate EM-MIA is robust across different scenarios while also revealing fundamental limitations of current MIA approaches when member and non-member distributions are nearly identical.



## **17. Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2503.15547v2) [paper-pdf](http://arxiv.org/pdf/2503.15547v2)

**Authors**: Juhee Kim, Woohyuk Choi, Byoungyoung Lee

**Abstract**: Large Language Models (LLMs) are combined with tools to create powerful LLM agents that provide a wide range of services. Unlike traditional software, LLM agent's behavior is determined at runtime by natural language prompts from either user or tool's data. This flexibility enables a new computing paradigm with unlimited capabilities and programmability, but also introduces new security risks, vulnerable to privilege escalation attacks. Moreover, user prompts are prone to be interpreted in an insecure way by LLM agents, creating non-deterministic behaviors that can be exploited by attackers. To address these security risks, we propose Prompt Flow Integrity (PFI), a system security-oriented solution to prevent privilege escalation in LLM agents. Analyzing the architectural characteristics of LLM agents, PFI features three mitigation techniques -- i.e., agent isolation, secure untrusted data processing, and privilege escalation guardrails. Our evaluation result shows that PFI effectively mitigates privilege escalation attacks while successfully preserving the utility of LLM agents.



## **18. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

cs.SE

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2411.10565v2) [paper-pdf](http://arxiv.org/pdf/2411.10565v2)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.



## **19. REDEditing: Relationship-Driven Precise Backdoor Poisoning on Text-to-Image Diffusion Models**

cs.CR

10 pages, 7 figures

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2504.14554v1) [paper-pdf](http://arxiv.org/pdf/2504.14554v1)

**Authors**: Chongye Guo, Jinhu Fu, Junfeng Fang, Kun Wang, Guorui Feng

**Abstract**: The rapid advancement of generative AI highlights the importance of text-to-image (T2I) security, particularly with the threat of backdoor poisoning. Timely disclosure and mitigation of security vulnerabilities in T2I models are crucial for ensuring the safe deployment of generative models. We explore a novel training-free backdoor poisoning paradigm through model editing, which is recently employed for knowledge updating in large language models. Nevertheless, we reveal the potential security risks posed by model editing techniques to image generation models. In this work, we establish the principles for backdoor attacks based on model editing, and propose a relationship-driven precise backdoor poisoning method, REDEditing. Drawing on the principles of equivalent-attribute alignment and stealthy poisoning, we develop an equivalent relationship retrieval and joint-attribute transfer approach that ensures consistent backdoor image generation through concept rebinding. A knowledge isolation constraint is proposed to preserve benign generation integrity. Our method achieves an 11\% higher attack success rate compared to state-of-the-art approaches. Remarkably, adding just one line of code enhances output naturalness while improving backdoor stealthiness by 24\%. This work aims to heighten awareness regarding this security vulnerability in editable image generation models.



## **20. Reconstruction of Differentially Private Text Sanitization via Large Language Models**

cs.CR

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2410.12443v2) [paper-pdf](http://arxiv.org/pdf/2410.12443v2)

**Authors**: Shuchao Pang, Zhigang Lu, Haichen Wang, Peng Fu, Yongbin Zhou, Minhui Xue

**Abstract**: Differential privacy (DP) is the de facto privacy standard against privacy leakage attacks, including many recently discovered ones against large language models (LLMs). However, we discovered that LLMs could reconstruct the altered/removed privacy from given DP-sanitized prompts. We propose two attacks (black-box and white-box) based on the accessibility to LLMs and show that LLMs could connect the pair of DP-sanitized text and the corresponding private training data of LLMs by giving sample text pairs as instructions (in the black-box attacks) or fine-tuning data (in the white-box attacks). To illustrate our findings, we conduct comprehensive experiments on modern LLMs (e.g., LLaMA-2, LLaMA-3, ChatGPT-3.5, ChatGPT-4, ChatGPT-4o, Claude-3, Claude-3.5, OPT, GPT-Neo, GPT-J, Gemma-2, and Pythia) using commonly used datasets (such as WikiMIA, Pile-CC, and Pile-Wiki) against both word-level and sentence-level DP. The experimental results show promising recovery rates, e.g., the black-box attacks against the word-level DP over WikiMIA dataset gave 72.18% on LLaMA-2 (70B), 82.39% on LLaMA-3 (70B), 75.35% on Gemma-2, 91.2% on ChatGPT-4o, and 94.01% on Claude-3.5 (Sonnet). More urgently, this study indicates that these well-known LLMs have emerged as a new security risk for existing DP text sanitization approaches in the current environment.



## **21. SHIELD : An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models**

cs.CV

Accepted by Visual Intelligence

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2402.04178v2) [paper-pdf](http://arxiv.org/pdf/2402.04178v2)

**Authors**: Yichen Shi, Yuhao Gao, Yingxin Lai, Hongyang Wang, Jun Feng, Lei He, Jun Wan, Changsheng Chen, Zitong Yu, Xiaochun Cao

**Abstract**: Multimodal large language models (MLLMs) have demonstrated strong capabilities in vision-related tasks, capitalizing on their visual semantic comprehension and reasoning capabilities. However, their ability to detect subtle visual spoofing and forgery clues in face attack detection tasks remains underexplored. In this paper, we introduce a benchmark, SHIELD, to evaluate MLLMs for face spoofing and forgery detection. Specifically, we design true/false and multiple-choice questions to assess MLLM performance on multimodal face data across two tasks. For the face anti-spoofing task, we evaluate three modalities (i.e., RGB, infrared, and depth) under six attack types. For the face forgery detection task, we evaluate GAN-based and diffusion-based data, incorporating visual and acoustic modalities. We conduct zero-shot and few-shot evaluations in standard and chain of thought (COT) settings. Additionally, we propose a novel multi-attribute chain of thought (MA-COT) paradigm for describing and judging various task-specific and task-irrelevant attributes of face images. The findings of this study demonstrate that MLLMs exhibit strong potential for addressing the challenges associated with the security of facial recognition technology applications.



## **22. Jailbreaking as a Reward Misspecification Problem**

cs.LG

Accepted to ICLR 2025. Code:  https://github.com/zhxieml/remiss-jailbreak

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2406.14393v5) [paper-pdf](http://arxiv.org/pdf/2406.14393v5)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.



## **23. Multi-Stage Retrieval for Operational Technology Cybersecurity Compliance Using Large Language Models: A Railway Casestudy**

cs.AI

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.14044v1) [paper-pdf](http://arxiv.org/pdf/2504.14044v1)

**Authors**: Regan Bolton, Mohammadreza Sheikhfathollahi, Simon Parkinson, Dan Basher, Howard Parkinson

**Abstract**: Operational Technology Cybersecurity (OTCS) continues to be a dominant challenge for critical infrastructure such as railways. As these systems become increasingly vulnerable to malicious attacks due to digitalization, effective documentation and compliance processes are essential to protect these safety-critical systems. This paper proposes a novel system that leverages Large Language Models (LLMs) and multi-stage retrieval to enhance the compliance verification process against standards like IEC 62443 and the rail-specific IEC 63452. We first evaluate a Baseline Compliance Architecture (BCA) for answering OTCS compliance queries, then develop an extended approach called Parallel Compliance Architecture (PCA) that incorporates additional context from regulatory standards. Through empirical evaluation comparing OpenAI-gpt-4o and Claude-3.5-haiku models in these architectures, we demonstrate that the PCA significantly improves both correctness and reasoning quality in compliance verification. Our research establishes metrics for response correctness, logical reasoning, and hallucination detection, highlighting the strengths and limitations of using LLMs for compliance verification in railway cybersecurity. The results suggest that retrieval-augmented approaches can significantly improve the efficiency and accuracy of compliance assessments, particularly valuable in an industry facing a shortage of cybersecurity expertise.



## **24. Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?**

cs.SE

The paper has been peer-reviewed and accepted for publication to the  29th International Conference on Evaluation and Assessment in Software  Engineering (EASE 2025)

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13769v1) [paper-pdf](http://arxiv.org/pdf/2504.13769v1)

**Authors**: Motunrayo Ibiyo, Thinakone Louangdy, Phuong T. Nguyen, Claudio Di Sipio, Davide Di Ruscio

**Abstract**: Malicious software packages in open-source ecosystems, such as PyPI, pose growing security risks. Unlike traditional vulnerabilities, these packages are intentionally designed to deceive users, making detection challenging due to evolving attack methods and the lack of structured datasets. In this work, we empirically evaluate the effectiveness of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and few-shot learning for detecting malicious source code. We fine-tune LLMs on curated datasets and integrate YARA rules, GitHub Security Advisories, and malicious code snippets with the aim of enhancing classification accuracy. We came across a counterintuitive outcome: While RAG is expected to boost up the prediction performance, it fails in the performed evaluation, obtaining a mediocre accuracy. In contrast, few-shot learning is more effective as it significantly improves the detection of malicious code, achieving 97% accuracy and 95% balanced accuracy, outperforming traditional RAG approaches. Thus, future work should expand structured knowledge bases, refine retrieval models, and explore hybrid AI-driven cybersecurity solutions.



## **25. DETAM: Defending LLMs Against Jailbreak Attacks via Targeted Attention Modification**

cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13562v1) [paper-pdf](http://arxiv.org/pdf/2504.13562v1)

**Authors**: Yu Li, Han Jiang, Zhihua Wei

**Abstract**: With the widespread adoption of Large Language Models (LLMs), jailbreak attacks have become an increasingly pressing safety concern. While safety-aligned LLMs can effectively defend against normal harmful queries, they remain vulnerable to such attacks. Existing defense methods primarily rely on fine-tuning or input modification, which often suffer from limited generalization and reduced utility. To address this, we introduce DETAM, a finetuning-free defense approach that improves the defensive capabilities against jailbreak attacks of LLMs via targeted attention modification. Specifically, we analyze the differences in attention scores between successful and unsuccessful defenses to identify the attention heads sensitive to jailbreak attacks. During inference, we reallocate attention to emphasize the user's core intention, minimizing interference from attack tokens. Our experimental results demonstrate that DETAM outperforms various baselines in jailbreak defense and exhibits robust generalization across different attacks and models, maintaining its effectiveness even on in-the-wild jailbreak data. Furthermore, in evaluating the model's utility, we incorporated over-defense datasets, which further validate the superior performance of our approach. The code will be released immediately upon acceptance.



## **26. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

cs.CL

WWW'25 research track accepted

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2406.11260v3) [paper-pdf](http://arxiv.org/pdf/2406.11260v3)

**Authors**: Sungwon Park, Sungwon Han, Xing Xie, Jae-Gil Lee, Meeyoung Cha

**Abstract**: The spread of fake news harms individuals and presents a critical social challenge that must be addressed. Although numerous algorithmic and insightful features have been developed to detect fake news, many of these features can be manipulated with style-conversion attacks, especially with the emergence of advanced language models, making it more difficult to differentiate from genuine news. This study proposes adversarial style augmentation, AdStyle, designed to train a fake news detector that remains robust against various style-conversion attacks. The primary mechanism involves the strategic use of LLMs to automatically generate a diverse and coherent array of style-conversion attack prompts, enhancing the generation of particularly challenging prompts for the detector. Experiments indicate that our augmentation strategy significantly improves robustness and detection performance when evaluated on fake news benchmark datasets.



## **27. Large Language Models for Validating Network Protocol Parsers**

cs.SE

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13515v1) [paper-pdf](http://arxiv.org/pdf/2504.13515v1)

**Authors**: Mingwei Zheng, Danning Xie, Xiangyu Zhang

**Abstract**: Network protocol parsers are essential for enabling correct and secure communication between devices. Bugs in these parsers can introduce critical vulnerabilities, including memory corruption, information leakage, and denial-of-service attacks. An intuitive way to assess parser correctness is to compare the implementation with its official protocol standard. However, this comparison is challenging because protocol standards are typically written in natural language, whereas implementations are in source code. Existing methods like model checking, fuzzing, and differential testing have been used to find parsing bugs, but they either require significant manual effort or ignore the protocol standards, limiting their ability to detect semantic violations. To enable more automated validation of parser implementations against protocol standards, we propose PARVAL, a multi-agent framework built on large language models (LLMs). PARVAL leverages the capabilities of LLMs to understand both natural language and code. It transforms both protocol standards and their implementations into a unified intermediate representation, referred to as format specifications, and performs a differential comparison to uncover inconsistencies. We evaluate PARVAL on the Bidirectional Forwarding Detection (BFD) protocol. Our experiments demonstrate that PARVAL successfully identifies inconsistencies between the implementation and its RFC standard, achieving a low false positive rate of 5.6%. PARVAL uncovers seven unique bugs, including five previously unknown issues.



## **28. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.05050v2) [paper-pdf](http://arxiv.org/pdf/2504.05050v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.



## **29. GraphQLer: Enhancing GraphQL Security with Context-Aware API Testing**

cs.CR

Publicly available on: https://github.com/omar2535/GraphQLer

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13358v1) [paper-pdf](http://arxiv.org/pdf/2504.13358v1)

**Authors**: Omar Tsai, Jianing Li, Tsz Tung Cheung, Lejing Huang, Hao Zhu, Jianrui Xiao, Iman Sharafaldin, Mohammad A. Tayebi

**Abstract**: GraphQL is an open-source data query and manipulation language for web applications, offering a flexible alternative to RESTful APIs. However, its dynamic execution model and lack of built-in security mechanisms expose it to vulnerabilities such as unauthorized data access, denial-of-service (DoS) attacks, and injections. Existing testing tools focus on functional correctness, often overlooking security risks stemming from query interdependencies and execution context. This paper presents GraphQLer, the first context-aware security testing framework for GraphQL APIs. GraphQLer constructs a dependency graph to analyze relationships among mutations, queries, and objects, capturing critical interdependencies. It chains related queries and mutations to reveal authentication and authorization flaws, access control bypasses, and resource misuse. Additionally, GraphQLer tracks internal resource usage to uncover data leakage, privilege escalation, and replay attack vectors. We assess GraphQLer on various GraphQL APIs, demonstrating improved testing coverage - averaging a 35% increase, with up to 84% in some cases - compared to top-performing baselines. Remarkably, this is achieved in less time, making GraphQLer suitable for time-sensitive contexts. GraphQLer also successfully detects a known CVE and potential vulnerabilities in large-scale production APIs. These results underline GraphQLer's utility in proactively securing GraphQL APIs through automated, context-aware vulnerability detection.



## **30. GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms**

cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13052v1) [paper-pdf](http://arxiv.org/pdf/2504.13052v1)

**Authors**: Sinan He, An Wang

**Abstract**: Large Language Models (LLMs) have been equipped with safety mechanisms to prevent harmful outputs, but these guardrails can often be bypassed through "jailbreak" prompts. This paper introduces a novel graph-based approach to systematically generate jailbreak prompts through semantic transformations. We represent malicious prompts as nodes in a graph structure with edges denoting different transformations, leveraging Abstract Meaning Representation (AMR) and Resource Description Framework (RDF) to parse user goals into semantic components that can be manipulated to evade safety filters. We demonstrate a particularly effective exploitation vector by instructing LLMs to generate code that realizes the intent described in these semantic graphs, achieving success rates of up to 87% against leading commercial LLMs. Our analysis reveals that contextual framing and abstraction are particularly effective at circumventing safety measures, highlighting critical gaps in current safety alignment techniques that focus primarily on surface-level patterns. These findings provide insights for developing more robust safeguards against structured semantic attacks. Our research contributes both a theoretical framework and practical methodology for systematically stress-testing LLM safety mechanisms.



## **31. From Sands to Mansions: Towards Automated Cyberattack Emulation with Classical Planning and Large Language Models**

cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2407.16928v3) [paper-pdf](http://arxiv.org/pdf/2407.16928v3)

**Authors**: Lingzhi Wang, Zhenyuan Li, Yi Jiang, Zhengkai Wang, Zonghan Guo, Jiahui Wang, Yangyang Wei, Xiangmin Shen, Wei Ruan, Yan Chen

**Abstract**: As attackers continually advance their tools, skills, and techniques during cyberattacks - particularly in modern Advanced Persistence Threats (APT) campaigns - there is a pressing need for a comprehensive and up-to-date cyberattack dataset to support threat-informed defense and enable benchmarking of defense systems in both academia and commercial solutions. However, there is a noticeable scarcity of cyberattack datasets: recent academic studies continue to rely on outdated benchmarks, while cyberattack emulation in industry remains limited due to the significant human effort and expertise required. Creating datasets by emulating advanced cyberattacks presents several challenges, such as limited coverage of attack techniques, the complexity of chaining multiple attack steps, and the difficulty of realistically mimicking actual threat groups. In this paper, we introduce modularized Attack Action and Attack Action Linking Model as a structured way to organizing and chaining individual attack steps into multi-step cyberattacks. Building on this, we propose Aurora, a system that autonomously emulates cyberattacks using third-party attack tools and threat intelligence reports with the help of classical planning and large language models. Aurora can automatically generate detailed attack plans, set up emulation environments, and semi-automatically execute the attacks. We utilize Aurora to create a dataset containing over 1,000 attack chains. To our best knowledge, Aurora is the only system capable of automatically constructing such a large-scale cyberattack dataset with corresponding attack execution scripts and environments. Our evaluation further demonstrates that Aurora outperforms the previous similar work and even the most advanced generative AI models in cyberattack emulation. To support further research, we published the cyberattack dataset and will publish the source code of Aurora.



## **32. ControlNET: A Firewall for RAG-based LLM System**

cs.CR

Project Page: https://ai.zjuicsr.cn/firewall

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.09593v2) [paper-pdf](http://arxiv.org/pdf/2504.09593v2)

**Authors**: Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, Zhan Qin

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.



## **33. PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization**

cs.CR

Accepted at SIGIR 2025

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.07717v2) [paper-pdf](http://arxiv.org/pdf/2504.07717v2)

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications, e.g., medical question-answering, mathematical sciences, and code generation. However, they also exhibit inherent limitations, such as outdated knowledge and susceptibility to hallucinations. Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address these issues, but it also introduces new vulnerabilities. Recent efforts have focused on the security of RAG-based LLMs, yet existing attack methods face three critical challenges: (1) their effectiveness declines sharply when only a limited number of poisoned texts can be injected into the knowledge database, (2) they lack sufficient stealth, as the attacks are often detectable by anomaly detection systems, which compromises their effectiveness, and (3) they rely on heuristic approaches to generate poisoned texts, lacking formal optimization frameworks and theoretic guarantees, which limits their effectiveness and applicability. To address these issues, we propose coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack that introduces a small number of poisoned texts into the knowledge database while embedding a backdoor trigger within the prompt. When activated, the trigger causes the LLM to generate pre-designed responses to targeted queries, while maintaining normal behavior in other contexts. This ensures both high effectiveness and stealth. We formulate the attack generation process as a bilevel optimization problem leveraging a principled optimization framework to develop optimal poisoned texts and triggers. Extensive experiments across diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving a high attack success rate even with a limited number of poisoned texts and significantly improved stealth compared to existing methods.



## **34. Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails**

cs.CR

12 pages, 5 figures, 6 tables

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11168v2) [paper-pdf](http://arxiv.org/pdf/2504.11168v2)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.



## **35. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2402.09063v2) [paper-pdf](http://arxiv.org/pdf/2402.09063v2)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.



## **36. LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks**

cs.CL

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.10185v2) [paper-pdf](http://arxiv.org/pdf/2504.10185v2)

**Authors**: Soumyadeep Pal, Changsheng Wang, James Diffenderfer, Bhavya Kailkhura, Sijia Liu

**Abstract**: Large language model unlearning has become a critical challenge in ensuring safety and controlled model behavior by removing undesired data-model influences from the pretrained model while preserving general utility. Significant recent efforts have been dedicated to developing LLM unlearning benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine Unlearning Six-way Evaluation), facilitating standardized unlearning performance assessment and method comparison. Despite their usefulness, we uncover for the first time a novel coreset effect within these benchmarks. Specifically, we find that LLM unlearning achieved with the original (full) forget set can be effectively maintained using a significantly smaller subset (functioning as a "coreset"), e.g., as little as 5% of the forget set, even when selected at random. This suggests that LLM unlearning in these benchmarks can be performed surprisingly easily, even in an extremely low-data regime. We demonstrate that this coreset effect remains strong, regardless of the LLM unlearning method used, such as NPO (Negative Preference Optimization) and RMU (Representation Misdirection Unlearning), the popular ones in these benchmarks. The surprisingly strong coreset effect is also robust across various data selection methods, ranging from random selection to more sophisticated heuristic approaches. We explain the coreset effect in LLM unlearning through a keyword-based perspective, showing that keywords extracted from the forget set alone contribute significantly to unlearning effectiveness and indicating that current unlearning is driven by a compact set of high-impact tokens rather than the entire dataset. We further justify the faithfulness of coreset-unlearned models along additional dimensions, such as mode connectivity and robustness to jailbreaking attacks. Codes are available at https://github.com/OPTML-Group/MU-Coreset.



## **37. Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation**

cs.CL

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.12108v1) [paper-pdf](http://arxiv.org/pdf/2504.12108v1)

**Authors**: Shizhan Cai, Liang Ding, Dacheng Tao

**Abstract**: The rapid development of Large Language Models (LLMs) has intensified concerns about content traceability and potential misuse. Existing watermarking schemes for sampled text often face trade-offs between maintaining text quality and ensuring robust detection against various attacks. To address these issues, we propose a novel watermarking scheme that improves both detectability and text quality by introducing a cumulative watermark entropy threshold. Our approach is compatible with and generalizes existing sampling functions, enhancing adaptability. Experimental results across multiple LLMs show that our scheme significantly outperforms existing methods, achieving over 80\% improvements on widely-used datasets, e.g., MATH and GSM8K, while maintaining high detection accuracy.



## **38. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2410.02644v3) [paper-pdf](http://arxiv.org/pdf/2410.02644v3)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.



## **39. On the Feasibility of Using MultiModal LLMs to Execute AR Social Engineering Attacks**

cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.13209v1) [paper-pdf](http://arxiv.org/pdf/2504.13209v1)

**Authors**: Ting Bi, Chenghang Ye, Zheyu Yang, Ziyi Zhou, Cui Tang, Jun Zhang, Zui Tao, Kailong Wang, Liting Zhou, Yang Yang, Tianlong Yu

**Abstract**: Augmented Reality (AR) and Multimodal Large Language Models (LLMs) are rapidly evolving, providing unprecedented capabilities for human-computer interaction. However, their integration introduces a new attack surface for social engineering. In this paper, we systematically investigate the feasibility of orchestrating AR-driven Social Engineering attacks using Multimodal LLM for the first time, via our proposed SEAR framework, which operates through three key phases: (1) AR-based social context synthesis, which fuses Multimodal inputs (visual, auditory and environmental cues); (2) role-based Multimodal RAG (Retrieval-Augmented Generation), which dynamically retrieves and integrates contextual data while preserving character differentiation; and (3) ReInteract social engineering agents, which execute adaptive multiphase attack strategies through inference interaction loops. To verify SEAR, we conducted an IRB-approved study with 60 participants in three experimental configurations (unassisted, AR+LLM, and full SEAR pipeline) compiling a new dataset of 180 annotated conversations in simulated social scenarios. Our results show that SEAR is highly effective at eliciting high-risk behaviors (e.g., 93.3% of participants susceptible to email phishing). The framework was particularly effective in building trust, with 85% of targets willing to accept an attacker's call after an interaction. Also, we identified notable limitations such as ``occasionally artificial'' due to perceived authenticity gaps. This work provides proof-of-concept for AR-LLM driven social engineering attacks and insights for developing defensive countermeasures against next-generation augmented reality threats.



## **40. Progent: Programmable Privilege Control for LLM Agents**

cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11703v1) [paper-pdf](http://arxiv.org/pdf/2504.11703v1)

**Authors**: Tianneng Shi, Jingxuan He, Zhun Wang, Linyu Wu, Hongwei Li, Wenbo Guo, Dawn Song

**Abstract**: LLM agents are an emerging form of AI systems where large language models (LLMs) serve as the central component, utilizing a diverse set of tools to complete user-assigned tasks. Despite their great potential, LLM agents pose significant security risks. When interacting with the external world, they may encounter malicious commands from attackers, leading to the execution of dangerous actions. A promising way to address this is by enforcing the principle of least privilege: allowing only essential actions for task completion while blocking unnecessary ones. However, achieving this is challenging, as it requires covering diverse agent scenarios while preserving both security and utility.   We introduce Progent, the first privilege control mechanism for LLM agents. At its core is a domain-specific language for flexibly expressing privilege control policies applied during agent execution. These policies provide fine-grained constraints over tool calls, deciding when tool calls are permissible and specifying fallbacks if they are not. This enables agent developers and users to craft suitable policies for their specific use cases and enforce them deterministically to guarantee security. Thanks to its modular design, integrating Progent does not alter agent internals and requires only minimal changes to agent implementation, enhancing its practicality and potential for widespread adoption. To automate policy writing, we leverage LLMs to generate policies based on user queries, which are then updated dynamically for improved security and utility. Our extensive evaluation shows that it enables strong security while preserving high utility across three distinct scenarios or benchmarks: AgentDojo, ASB, and AgentPoison. Furthermore, we perform an in-depth analysis, showcasing the effectiveness of its core components and the resilience of its automated policy generation against adaptive attacks.



## **41. Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' "Typo" Correction**

cs.CR

Length: 13 pages Figures: 5 figures Tables: 7 tables Keywords:  Acoustic side-channel attacks, machine learning, Visual Transformers, Large  Language Models (LLMs), security Conference: Accepted at the 19th USENIX WOOT  Conference on Offensive Technologies (WOOT '25). Licensing: This paper is  submitted under the CC BY Creative Commons Attribution license. arXiv admin  note: text overlap with arXiv:2502.09782

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11622v1) [paper-pdf](http://arxiv.org/pdf/2504.11622v1)

**Authors**: Seyyed Ali Ayati, Jin Hyun Park, Yichen Cai, Marcus Botacin

**Abstract**: The large integration of microphones into devices increases the opportunities for Acoustic Side-Channel Attacks (ASCAs), as these can be used to capture keystrokes' audio signals that might reveal sensitive information. However, the current State-Of-The-Art (SOTA) models for ASCAs, including Convolutional Neural Networks (CNNs) and hybrid models, such as CoAtNet, still exhibit limited robustness under realistic noisy conditions. Solving this problem requires either: (i) an increased model's capacity to infer contextual information from longer sequences, allowing the model to learn that an initially noisily typed word is the same as a futurely collected non-noisy word, or (ii) an approach to fix misidentified information from the contexts, as one does not type random words, but the ones that best fit the conversation context. In this paper, we demonstrate that both strategies are viable and complementary solutions for making ASCAs practical. We observed that no existing solution leverages advanced transformer architectures' power for these tasks and propose that: (i) Visual Transformers (VTs) are the candidate solutions for capturing long-term contextual information and (ii) transformer-powered Large Language Models (LLMs) are the candidate solutions to fix the ``typos'' (mispredictions) the model might make. Thus, we here present the first-of-its-kind approach that integrates VTs and LLMs for ASCAs.   We first show that VTs achieve SOTA performance in classifying keystrokes when compared to the previous CNN benchmark. Second, we demonstrate that LLMs can mitigate the impact of real-world noise. Evaluations on the natural sentences revealed that: (i) incorporating LLMs (e.g., GPT-4o) in our ASCA pipeline boosts the performance of error-correction tasks; and (ii) the comparable performance can be attained by a lightweight, fine-tuned smaller LLM (67 times smaller than GPT-4o), using...



## **42. Propaganda via AI? A Study on Semantic Backdoors in Large Language Models**

cs.CL

18 pages, 1 figure

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.12344v1) [paper-pdf](http://arxiv.org/pdf/2504.12344v1)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Large language models (LLMs) demonstrate remarkable performance across myriad language tasks, yet they remain vulnerable to backdoor attacks, where adversaries implant hidden triggers that systematically manipulate model outputs. Traditional defenses focus on explicit token-level anomalies and therefore overlook semantic backdoors-covert triggers embedded at the conceptual level (e.g., ideological stances or cultural references) that rely on meaning-based cues rather than lexical oddities. We first show, in a controlled finetuning setting, that such semantic backdoors can be implanted with only a small poisoned corpus, establishing their practical feasibility. We then formalize the notion of semantic backdoors in LLMs and introduce a black-box detection framework, RAVEN (short for "Response Anomaly Vigilance for uncovering semantic backdoors"), which combines semantic entropy with cross-model consistency analysis. The framework probes multiple models with structured topic-perspective prompts, clusters the sampled responses via bidirectional entailment, and flags anomalously uniform outputs; cross-model comparison isolates model-specific anomalies from corpus-wide biases. Empirical evaluations across diverse LLM families (GPT-4o, Llama, DeepSeek, Mistral) uncover previously undetected semantic backdoors, providing the first proof-of-concept evidence of these hidden vulnerabilities and underscoring the urgent need for concept-level auditing of deployed language models. We open-source our code and data at https://github.com/NayMyatMin/RAVEN.



## **43. Lateral Phishing With Large Language Models: A Large Organization Comparative Study**

cs.CR

Accepted for publication in IEEE Access. This version includes  revisions following peer review

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2401.09727v2) [paper-pdf](http://arxiv.org/pdf/2401.09727v2)

**Authors**: Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nicole Beebe, Nishant Vishwamitra, Peyman Najafirad

**Abstract**: The emergence of Large Language Models (LLMs) has heightened the threat of phishing emails by enabling the generation of highly targeted, personalized, and automated attacks. Traditionally, many phishing emails have been characterized by typos, errors, and poor language. These errors can be mitigated by LLMs, potentially lowering the barrier for attackers. Despite this, there is a lack of large-scale studies comparing the effectiveness of LLM-generated lateral phishing emails to those crafted by humans. Current literature does not adequately address the comparative effectiveness of LLM and human-generated lateral phishing emails in a real-world, large-scale organizational setting, especially considering the potential for LLMs to generate more convincing and error-free phishing content. To address this gap, we conducted a pioneering study within a large university, targeting its workforce of approximately 9,000 individuals including faculty, staff, administrators, and student workers. Our results indicate that LLM-generated lateral phishing emails are as effective as those written by communications professionals, emphasizing the critical threat posed by LLMs in leading phishing campaigns. We break down the results of the overall phishing experiment, comparing vulnerability between departments and job roles. Furthermore, to gather qualitative data, we administered a detailed questionnaire, revealing insights into the reasons and motivations behind vulnerable employee's actions. This study contributes to the understanding of cyber security threats in educational institutions and provides a comprehensive comparison of LLM and human-generated phishing emails' effectiveness, considering the potential for LLMs to generate more convincing content. The findings highlight the need for enhanced user education and system defenses to mitigate the growing threat of AI-powered phishing attacks.



## **44. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2410.02240v5) [paper-pdf](http://arxiv.org/pdf/2410.02240v5)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.



## **45. The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections**

cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11281v1) [paper-pdf](http://arxiv.org/pdf/2504.11281v1)

**Authors**: Chaoran Chen, Zhiping Zhang, Bingcan Guo, Shang Ma, Ibrahim Khalilov, Simret A Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li

**Abstract**: A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.



## **46. Exploring Backdoor Attack and Defense for LLM-empowered Recommendations**

cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11182v1) [paper-pdf](http://arxiv.org/pdf/2504.11182v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner.



## **47. QAVA: Query-Agnostic Visual Attack to Large Vision-Language Models**

cs.CV

Accepted by NAACL 2025 main

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11038v1) [paper-pdf](http://arxiv.org/pdf/2504.11038v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Yu Wang

**Abstract**: In typical multimodal tasks, such as Visual Question Answering (VQA), adversarial attacks targeting a specific image and question can lead large vision-language models (LVLMs) to provide incorrect answers. However, it is common for a single image to be associated with multiple questions, and LVLMs may still answer other questions correctly even for an adversarial image attacked by a specific question. To address this, we introduce the query-agnostic visual attack (QAVA), which aims to create robust adversarial examples that generate incorrect responses to unspecified and unknown questions. Compared to traditional adversarial attacks focused on specific images and questions, QAVA significantly enhances the effectiveness and efficiency of attacks on images when the question is unknown, achieving performance comparable to attacks on known target questions. Our research broadens the scope of visual adversarial attacks on LVLMs in practical settings, uncovering previously overlooked vulnerabilities, particularly in the context of visual adversarial threats. The code is available at https://github.com/btzyd/qava.



## **48. Concept Enhancement Engineering: A Lightweight and Efficient Robust Defense Against Jailbreak Attacks in Embodied AI**

cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.13201v1) [paper-pdf](http://arxiv.org/pdf/2504.13201v1)

**Authors**: Jirui Yang, Zheyu Lin, Shuhan Yang, Zhihui Lu, Xin Du

**Abstract**: Embodied Intelligence (EI) systems integrated with large language models (LLMs) face significant security risks, particularly from jailbreak attacks that manipulate models into generating harmful outputs or executing unsafe physical actions. Traditional defense strategies, such as input filtering and output monitoring, often introduce high computational overhead or interfere with task performance in real-time embodied scenarios. To address these challenges, we propose Concept Enhancement Engineering (CEE), a novel defense framework that leverages representation engineering to enhance the safety of embodied LLMs by dynamically steering their internal activations. CEE operates by (1) extracting multilingual safety patterns from model activations, (2) constructing control directions based on safety-aligned concept subspaces, and (3) applying subspace concept rotation to reinforce safe behavior during inference. Our experiments demonstrate that CEE effectively mitigates jailbreak attacks while maintaining task performance, outperforming existing defense methods in both robustness and efficiency. This work contributes a scalable and interpretable safety mechanism for embodied AI, bridging the gap between theoretical representation engineering and practical security applications. Our findings highlight the potential of latent-space interventions as a viable defense paradigm against emerging adversarial threats in physically grounded AI systems.



## **49. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

cs.CR

7 pages; In submission

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2412.21051v2) [paper-pdf](http://arxiv.org/pdf/2412.21051v2)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Yuyu Zhao

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided a large number of benefits in daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks. Recent advancements in generative foundation models (GFMs), particularly in the large language models (LLMs), offer promising solutions for security intelligence. By exploiting the powerful abilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel proactive defense architecture that defeats various threats in a proactive manner. LLM-PD can efficiently make a decision through comprehensive data analysis and sequential reasoning, as well as dynamically creating and deploying actionable defense mechanisms on the target cloud. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. The experimental results demonstrate its remarkable ability in terms of defense effectiveness and efficiency, particularly highlighting an outstanding success rate when compared with other existing methods.



## **50. Adversarial Prompt Distillation for Vision-Language Models**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.15244v2) [paper-pdf](http://arxiv.org/pdf/2411.15244v2)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.



