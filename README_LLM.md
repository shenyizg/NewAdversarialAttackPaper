# Latest Large Language Model Attack Papers
**update at 2025-09-17 09:51:53**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. TrojanRobot: Physical-world Backdoor Attacks Against VLM-based Robotic Manipulation**

cs.RO

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.11683v5) [paper-pdf](http://arxiv.org/pdf/2411.11683v5)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Aishan Liu, Yunpeng Jiang, Leo Yu Zhang, Xiaohua Jia

**Abstract**: Robotic manipulation in the physical world is increasingly empowered by \textit{large language models} (LLMs) and \textit{vision-language models} (VLMs), leveraging their understanding and perception capabilities. Recently, various attacks against such robotic policies have been proposed, with backdoor attacks drawing considerable attention for their high stealth and strong persistence capabilities. However, existing backdoor efforts are limited to simulators and suffer from physical-world realization. To address this, we propose \textit{TrojanRobot}, a highly stealthy and broadly effective robotic backdoor attack in the physical world. Specifically, we introduce a module-poisoning approach by embedding a backdoor module into the modular robotic policy, enabling backdoor control over the policy's visual perception module thereby backdooring the entire robotic policy. Our vanilla implementation leverages a backdoor-finetuned VLM to serve as the backdoor module. To enhance its generalization in physical environments, we propose a prime implementation, leveraging the LVLM-as-a-backdoor paradigm and developing three types of prime attacks, \ie, \textit{permutation}, \textit{stagnation}, and \textit{intentional} attacks, thus achieving finer-grained backdoors. Extensive experiments on the UR3e manipulator with 18 task instructions using robotic policies based on four VLMs demonstrate the broad effectiveness and physical-world stealth of TrojanRobot. Our attack's video demonstrations are available via a github link https://trojanrobot.github.io.



## **2. Context-Aware Membership Inference Attacks against Pre-trained Large Language Models**

cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2409.13745v2) [paper-pdf](http://arxiv.org/pdf/2409.13745v2)

**Authors**: Hongyan Chang, Ali Shahin Shamsabadi, Kleomenis Katevas, Hamed Haddadi, Reza Shokri

**Abstract**: Membership Inference Attacks (MIAs) on pre-trained Large Language Models (LLMs) aim at determining if a data point was part of the model's training set. Prior MIAs that are built for classification models fail at LLMs, due to ignoring the generative nature of LLMs across token sequences. In this paper, we present a novel attack on pre-trained LLMs that adapts MIA statistical tests to the perplexity dynamics of subsequences within a data point. Our method significantly outperforms prior approaches, revealing context-dependent memorization patterns in pre-trained LLMs.



## **3. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

cs.CL

EMNLP 2025 Main (Oral)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2502.13061v4) [paper-pdf](http://arxiv.org/pdf/2502.13061v4)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL



## **4. Adversarial Prompt Distillation for Vision-Language Models**

cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.15244v3) [paper-pdf](http://arxiv.org/pdf/2411.15244v3)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.



## **5. DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval**

cs.IR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12824v1) [paper-pdf](http://arxiv.org/pdf/2509.12824v1)

**Authors**: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang

**Abstract**: Deep hashing models have been widely adopted to tackle the challenges of large-scale image retrieval. However, these approaches face serious security risks due to their vulnerability to adversarial examples. Despite the increasing exploration of targeted attacks on deep hashing models, existing approaches still suffer from a lack of multimodal guidance, reliance on labeling information and dependence on pixel-level operations for attacks. To address these limitations, we proposed DiffHash, a novel diffusion-based targeted attack for deep hashing. Unlike traditional pixel-based attacks that directly modify specific pixels and lack multimodal guidance, our approach focuses on optimizing the latent representations of images, guided by text information generated by a Large Language Model (LLM) for the target image. Furthermore, we designed a multi-space hash alignment network to align the high-dimension image space and text space to the low-dimension binary hash space. During reconstruction, we also incorporated text-guided attention mechanisms to refine adversarial examples, ensuring them aligned with the target semantics while maintaining visual plausibility. Extensive experiments have demonstrated that our method outperforms state-of-the-art (SOTA) targeted attack methods, achieving better black-box transferability and offering more excellent stability across datasets.



## **6. Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection**

cs.CV

Accepted to EMNLP 2025 (Main). 17 pages, 7 figures

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2507.02844v2) [paper-pdf](http://arxiv.org/pdf/2507.02844v2)

**Authors**: Ziqi Miao, Yi Ding, Lijun Li, Jing Shao

**Abstract**: With the emergence of strong vision language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: vision-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct vision-focused strategies, dynamically generating auxiliary images when necessary to construct a vision-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which achieves a toxicity score of 2.48 and an ASR of 22.2%. Code: https://github.com/Dtc7w3PQ/Visco-Attack.



## **7. SoK: How Sensor Attacks Disrupt Autonomous Vehicles: An End-to-end Analysis, Challenges, and Missed Threats**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11120v2) [paper-pdf](http://arxiv.org/pdf/2509.11120v2)

**Authors**: Qingzhao Zhang, Shaocheng Luo, Z. Morley Mao, Miroslav Pajic, Michael K. Reiter

**Abstract**: Autonomous vehicles, including self-driving cars, robotic ground vehicles, and drones, rely on complex sensor pipelines to ensure safe and reliable operation. However, these safety-critical systems remain vulnerable to adversarial sensor attacks that can compromise their performance and mission success. While extensive research has demonstrated various sensor attack techniques, critical gaps remain in understanding their feasibility in real-world, end-to-end systems. This gap largely stems from the lack of a systematic perspective on how sensor errors propagate through interconnected modules in autonomous systems when autonomous vehicles interact with the physical world.   To bridge this gap, we present a comprehensive survey of autonomous vehicle sensor attacks across platforms, sensor modalities, and attack methods. Central to our analysis is the System Error Propagation Graph (SEPG), a structured demonstration tool that illustrates how sensor attacks propagate through system pipelines, exposing the conditions and dependencies that determine attack feasibility. With the aid of SEPG, our study distills seven key findings that highlight the feasibility challenges of sensor attacks and uncovers eleven previously overlooked attack vectors exploiting inter-module interactions, several of which we validate through proof-of-concept experiments. Additionally, we demonstrate how large language models (LLMs) can automate aspects of SEPG construction and cross-validate expert analysis, showcasing the promise of AI-assisted security evaluation.



## **8. Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content**

cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12672v1) [paper-pdf](http://arxiv.org/pdf/2509.12672v1)

**Authors**: Shaz Furniturewala, Arkaitz Zubiaga

**Abstract**: The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.



## **9. A Systematic Evaluation of Parameter-Efficient Fine-Tuning Methods for the Security of Code LLMs**

cs.CR

25 pages

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12649v1) [paper-pdf](http://arxiv.org/pdf/2509.12649v1)

**Authors**: Kiho Lee, Jungkon Kim, Doowon Kim, Hyoungshick Kim

**Abstract**: Code-generating Large Language Models (LLMs) significantly accelerate software development. However, their frequent generation of insecure code presents serious risks. We present a comprehensive evaluation of seven parameter-efficient fine-tuning (PEFT) techniques, demonstrating substantial gains in secure code generation without compromising functionality. Our research identifies prompt-tuning as the most effective PEFT method, achieving an 80.86% Overall-Secure-Rate on CodeGen2 16B, a 13.5-point improvement over the 67.28% baseline. Optimizing decoding strategies through sampling temperature further elevated security to 87.65%. This equates to a reduction of approximately 203,700 vulnerable code snippets per million generated. Moreover, prompt and prefix tuning increase robustness against poisoning attacks in our TrojanPuzzle evaluation, with strong performance against CWE-79 and CWE-502 attack vectors. Our findings generalize across Python and Java, confirming prompt-tuning's consistent effectiveness. This study provides essential insights and practical guidance for building more resilient software systems with LLMs.



## **10. Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs**

cs.CL

Preprint

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11177v2) [paper-pdf](http://arxiv.org/pdf/2509.11177v2)

**Authors**: Hang Guo, Yawei Li, Luca Benini

**Abstract**: Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.



## **11. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.20890v2) [paper-pdf](http://arxiv.org/pdf/2508.20890v2)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.



## **12. Phi: Preference Hijacking in Multi-modal Large Language Models at Inference Time**

cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.12521v1) [paper-pdf](http://arxiv.org/pdf/2509.12521v1)

**Authors**: Yifan Lan, Yuanpu Cao, Weitong Zhang, Lu Lin, Jinghui Chen

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have gained significant attention across various domains. However, their widespread adoption has also raised serious safety concerns. In this paper, we uncover a new safety risk of MLLMs: the output preference of MLLMs can be arbitrarily manipulated by carefully optimized images. Such attacks often generate contextually relevant yet biased responses that are neither overtly harmful nor unethical, making them difficult to detect. Specifically, we introduce a novel method, Preference Hijacking (Phi), for manipulating the MLLM response preferences using a preference hijacked image. Our method works at inference time and requires no model modifications. Additionally, we introduce a universal hijacking perturbation -- a transferable component that can be embedded into different images to hijack MLLM responses toward any attacker-specified preferences. Experimental results across various tasks demonstrate the effectiveness of our approach. The code for Phi is accessible at https://github.com/Yifan-Lan/Phi.



## **13. Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering**

cs.CL

EMNLP 2025 (Main Conference)

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2505.15805v2) [paper-pdf](http://arxiv.org/pdf/2505.15805v2)

**Authors**: Hwan Chang, Yumin Kim, Yonghyun Jun, Hwanhee Lee

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains such as enterprise and government, ensuring that they adhere to user-defined security policies within context is critical-especially with respect to information non-disclosure. While prior LLM studies have focused on general safety and socially sensitive data, large-scale benchmarks for contextual security preservation against attacks remain lacking. To address this, we introduce a novel large-scale benchmark dataset, CoPriva, evaluating LLM adherence to contextual non-disclosure policies in question answering. Derived from realistic contexts, our dataset includes explicit policies and queries designed as direct and challenging indirect attacks seeking prohibited information. We evaluate 10 LLMs on our benchmark and reveal a significant vulnerability: many models violate user-defined policies and leak sensitive information. This failure is particularly severe against indirect attacks, highlighting a critical gap in current LLM safety alignment for sensitive applications. Our analysis reveals that while models can often identify the correct answer to a query, they struggle to incorporate policy constraints during generation. In contrast, they exhibit a partial ability to revise outputs when explicitly prompted. Our findings underscore the urgent need for more robust methods to guarantee contextual security.



## **14. Safety Pretraining: Toward the Next Generation of Safe AI**

cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2504.16980v2) [paper-pdf](http://arxiv.org/pdf/2504.16980v2)

**Authors**: Pratyush Maini, Sachin Goyal, Dylan Sam, Alex Robey, Yash Savani, Yiding Jiang, Andy Zou, Matt Fredrikson, Zacharcy C. Lipton, J. Zico Kolter

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes settings, the risk of generating harmful or toxic content remains a central challenge. Post-hoc alignment methods are brittle: once unsafe patterns are learned during pretraining, they are hard to remove. In this work, we present a data-centric pretraining framework that builds safety into the model from the start. Our framework consists of four key steps: (i) Safety Filtering: building a safety classifier to classify webdata into safe and unsafe categories; (ii) Safety Rephrasing: we recontextualize unsafe webdata into safer narratives; (iii) Native Refusal: we develop RefuseWeb and Moral Education pretraining datasets that actively teach model to refuse on unsafe content and the moral reasoning behind it, and (iv) Harmfulness-Tag annotated pretraining: we flag unsafe content during pretraining using a special token, and use it to steer model away from unsafe generations at inference. Our safety-pretrained models reduce attack success rates from 38.8\% to 8.4\% on standard LLM safety benchmarks with no performance degradation on general tasks.



## **15. Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.05755v2) [paper-pdf](http://arxiv.org/pdf/2509.05755v2)

**Authors**: Yu Liu, Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.



## **16. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11864v1) [paper-pdf](http://arxiv.org/pdf/2509.11864v1)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.



## **17. One Goal, Many Challenges: Robust Preference Optimization Amid Content-Aware and Multi-Source Noise**

cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2503.12301v2) [paper-pdf](http://arxiv.org/pdf/2503.12301v2)

**Authors**: Amirabbas Afzali, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi, Sanjay Lall

**Abstract**: Large Language Models (LLMs) have made significant strides in generating human-like responses, largely due to preference alignment techniques. However, these methods often assume unbiased human feedback, which is rarely the case in real-world scenarios. This paper introduces Content-Aware Noise-Resilient Preference Optimization (CNRPO), a novel framework that addresses multiple sources of content-dependent noise in preference learning. CNRPO employs a multi-objective optimization approach to separate true preferences from content-aware noises, effectively mitigating their impact. We leverage backdoor attack mechanisms to efficiently learn and control various noise sources within a single model. Theoretical analysis and extensive experiments on different synthetic noisy datasets demonstrate that CNRPO significantly improves alignment with primary human preferences while controlling for secondary noises and biases, such as response length and harmfulness.



## **18. Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check**

cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11629v1) [paper-pdf](http://arxiv.org/pdf/2509.11629v1)

**Authors**: Chentao Cao, Xiaojun Xu, Bo Han, Hang Li

**Abstract**: As large language models (LLMs) continue to advance in capabilities, ensuring their safety against jailbreak attacks remains a critical challenge. In this paper, we introduce a novel safety alignment approach called Answer-Then-Check, which enhances LLM robustness against malicious prompts by applying thinking ability to mitigate jailbreaking problems before producing a final answer to the user. Our method enables models to directly answer the question in their thought and then critically evaluate its safety before deciding whether to provide it. To implement this approach, we construct the Reasoned Safety Alignment (ReSA) dataset, comprising 80K examples that teach models to reason through direct responses and then analyze their safety. Experimental results demonstrate that our approach achieves the Pareto frontier with superior safety capability while decreasing over-refusal rates on over-refusal benchmarks. Notably, the model fine-tuned with ReSA maintains general reasoning capabilities on benchmarks like MMLU, MATH500, and HumanEval. Besides, our method equips models with the ability to perform safe completion. Unlike post-hoc methods that can only reject harmful queries, our model can provide helpful and safe alternative responses for sensitive topics (e.g., self-harm). Furthermore, we discover that training on a small subset of just 500 examples can achieve comparable performance to using the full dataset, suggesting that safety alignment may require less data than previously assumed.



## **19. Multilingual Collaborative Defense for Large Language Models**

cs.CL

21 pages, 4figures

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2505.11835v2) [paper-pdf](http://arxiv.org/pdf/2505.11835v2)

**Authors**: Hongliang Li, Jinan Xu, Gengping Cui, Changhao Guan, Fengran Mo, Kaiyu Huang

**Abstract**: The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at https://github.com/HLiang-Lee/MCD.



## **20. Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2508.16347v2) [paper-pdf](http://arxiv.org/pdf/2508.16347v2)

**Authors**: Yu Yan, Sheng Sun, Zhe Wang, Yijun Lin, Zenghao Duan, zhifei zheng, Min Liu, Zhiyi yin, Jianping Zhang

**Abstract**: With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.



## **21. Enhancing Prompt Injection Attacks to LLMs via Poisoning Alignment**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2410.14827v3) [paper-pdf](http://arxiv.org/pdf/2410.14827v3)

**Authors**: Zedian Shao, Hongbin Liu, Jaden Mu, Neil Zhenqiang Gong

**Abstract**: Prompt injection attack, where an attacker injects a prompt into the original one, aiming to make an Large Language Model (LLM) follow the injected prompt to perform an attacker-chosen task, represent a critical security threat. Existing attacks primarily focus on crafting these injections at inference time, treating the LLM itself as a static target. Our experiments show that these attacks achieve some success, but there is still significant room for improvement. In this work, we introduces a more foundational attack vector: poisoning the LLM's alignment process to amplify the success of future prompt injection attacks. Specifically, we propose PoisonedAlign, a method that strategically creates poisoned alignment samples to poison an LLM's alignment dataset. Our experiments across five LLMs and two alignment datasets show that when even a small fraction of the alignment data is poisoned, the resulting model becomes substantially more vulnerable to a wide range of prompt injection attacks. Crucially, this vulnerability is instilled while the LLM's performance on standard capability benchmarks remains largely unchanged, making the manipulation difficult to detect through automated, general-purpose performance evaluations. The code for implementing the attack is available at https://github.com/Sadcardation/PoisonedAlign.



## **22. Securing AI Agents: Implementing Role-Based Access Control for Industrial Applications**

cs.AI

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11431v1) [paper-pdf](http://arxiv.org/pdf/2509.11431v1)

**Authors**: Aadil Gani Ganie

**Abstract**: The emergence of Large Language Models (LLMs) has significantly advanced solutions across various domains, from political science to software development. However, these models are constrained by their training data, which is static and limited to information available up to a specific date. Additionally, their generalized nature often necessitates fine-tuning -- whether for classification or instructional purposes -- to effectively perform specific downstream tasks. AI agents, leveraging LLMs as their core, mitigate some of these limitations by accessing external tools and real-time data, enabling applications such as live weather reporting and data analysis. In industrial settings, AI agents are transforming operations by enhancing decision-making, predictive maintenance, and process optimization. For example, in manufacturing, AI agents enable near-autonomous systems that boost productivity and support real-time decision-making. Despite these advancements, AI agents remain vulnerable to security threats, including prompt injection attacks, which pose significant risks to their integrity and reliability. To address these challenges, this paper proposes a framework for integrating Role-Based Access Control (RBAC) into AI agents, providing a robust security guardrail. This framework aims to support the effective and scalable deployment of AI agents, with a focus on on-premises implementations.



## **23. Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs**

cs.CV

Accepted by ACM Multimedia 2025 BNI track (Oral)

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.01064v3) [paper-pdf](http://arxiv.org/pdf/2506.01064v3)

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive ``fighting fire with fire'' strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code is available at https://github.com/btzyd/F3.



## **24. Beyond the Protocol: Unveiling Attack Vectors in the Model Context Protocol (MCP) Ecosystem**

cs.CR

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.02040v4) [paper-pdf](http://arxiv.org/pdf/2506.02040v4)

**Authors**: Hao Song, Yiming Shen, Wenxuan Luo, Leixin Guo, Ting Chen, Jiashui Wang, Beibei Li, Xiaosong Zhang, Jiachi Chen

**Abstract**: The Model Context Protocol (MCP) is an emerging standard designed to enable seamless interaction between Large Language Model (LLM) applications and external tools or resources. Within a short period, thousands of MCP services have been developed and deployed. However, the client-server integration architecture inherent in MCP may expand the attack surface against LLM Agent systems, introducing new vulnerabilities that allow attackers to exploit by designing malicious MCP servers. In this paper, we present the first end-to-end empirical evaluation of attack vectors targeting the MCP ecosystem. We identify four categories of attacks, i.e., Tool Poisoning Attacks, Puppet Attacks, Rug Pull Attacks, and Exploitation via Malicious External Resources. To evaluate their feasibility, we conduct experiments following the typical steps of launching an attack through malicious MCP servers: upload -> download -> attack. Specifically, we first construct malicious MCP servers and successfully upload them to three widely used MCP aggregation platforms. The results indicate that current audit mechanisms are insufficient to identify and prevent these threats. Next, through a user study and interview with 20 participants, we demonstrate that users struggle to identify malicious MCP servers and often unknowingly install them from aggregator platforms. Finally, we empirically demonstrate that these attacks can trigger harmful actions within the user's local environment, such as accessing private files or controlling devices to transfer digital assets. Additionally, based on interview results, we discuss four key challenges faced by the current MCP security ecosystem. These findings underscore the urgent need for robust security mechanisms to defend against malicious MCP servers and ensure the safe deployment of increasingly autonomous LLM agents.



## **25. Character-Level Perturbations Disrupt LLM Watermarks**

cs.CR

accepted by Network and Distributed System Security (NDSS) Symposium  2026

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.09112v2) [paper-pdf](http://arxiv.org/pdf/2509.09112v2)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.   To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.



## **26. Free-MAD: Consensus-Free Multi-Agent Debate**

cs.AI

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11035v1) [paper-pdf](http://arxiv.org/pdf/2509.11035v1)

**Authors**: Yu Cui, Hang Fu, Haibin Zhang, Licheng Wang, Cong Zuo

**Abstract**: Multi-agent debate (MAD) is an emerging approach to improving the reasoning capabilities of large language models (LLMs). Existing MAD methods rely on multiple rounds of interaction among agents to reach consensus, and the final output is selected by majority voting in the last round. However, this consensus-based design faces several limitations. First, multiple rounds of communication increases token overhead and limits scalability. Second, due to the inherent conformity of LLMs, agents that initially produce correct responses may be influenced by incorrect ones during the debate process, causing error propagation. Third, majority voting introduces randomness and unfairness in the decision-making phase, and can degrade the reasoning performance.   To address these issues, we propose \textsc{Free-MAD}, a novel MAD framework that eliminates the need for consensus among agents. \textsc{Free-MAD} introduces a novel score-based decision mechanism that evaluates the entire debate trajectory rather than relying on the last round only. This mechanism tracks how each agent's reasoning evolves, enabling more accurate and fair outcomes. In addition, \textsc{Free-MAD} reconstructs the debate phase by introducing anti-conformity, a mechanism that enables agents to mitigate excessive influence from the majority. Experiments on eight benchmark datasets demonstrate that \textsc{Free-MAD} significantly improves reasoning performance while requiring only a single-round debate and thus reducing token costs. We also show that compared to existing MAD approaches, \textsc{Free-MAD} exhibits improved robustness in real-world attack scenarios.



## **27. Public Data Assisted Differentially Private In-Context Learning**

cs.AI

EMNLP 2025 Findings

**SubmitDate**: 2025-09-13    [abs](http://arxiv.org/abs/2509.10932v1) [paper-pdf](http://arxiv.org/pdf/2509.10932v1)

**Authors**: Seongho Joo, Hyukhun Koh, Kyomin Jung

**Abstract**: In-context learning (ICL) in Large Language Models (LLMs) has shown remarkable performance across various tasks without requiring fine-tuning. However, recent studies have highlighted the risk of private data leakage through the prompt in ICL, especially when LLMs are exposed to malicious attacks. While differential privacy (DP) provides strong privacy guarantees, it often significantly reduces the utility of in-context learning (ICL). To address this challenge, we incorporate task-related public data into the ICL framework while maintaining the DP guarantee. Based on this approach, we propose a private in-context learning algorithm that effectively balances privacy protection and model utility. Through experiments, we demonstrate that our approach significantly improves the utility of private ICL with the assistance of public data. Additionally, we show that our method is robust against membership inference attacks, demonstrating empirical privacy protection.



## **28. Harmful Prompt Laundering: Jailbreaking LLMs with Abductive Styles and Symbolic Encoding**

cs.AI

EMNLP 2025

**SubmitDate**: 2025-09-13    [abs](http://arxiv.org/abs/2509.10931v1) [paper-pdf](http://arxiv.org/pdf/2509.10931v1)

**Authors**: Seongho Joo, Hyukhun Koh, Kyomin Jung

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their potential misuse for harmful purposes remains a significant concern. To strengthen defenses against such vulnerabilities, it is essential to investigate universal jailbreak attacks that exploit intrinsic weaknesses in the architecture and learning paradigms of LLMs. In response, we propose \textbf{H}armful \textbf{P}rompt \textbf{La}undering (HaPLa), a novel and broadly applicable jailbreaking technique that requires only black-box access to target models. HaPLa incorporates two primary strategies: 1) \textit{abductive framing}, which instructs LLMs to infer plausible intermediate steps toward harmful activities, rather than directly responding to explicit harmful queries; and 2) \textit{symbolic encoding}, a lightweight and flexible approach designed to obfuscate harmful content, given that current LLMs remain sensitive primarily to explicit harmful keywords. Experimental results show that HaPLa achieves over 95% attack success rate on GPT-series models and 70% across all targets. Further analysis with diverse symbolic encoding rules also reveals a fundamental challenge: it remains difficult to safely tune LLMs without significantly diminishing their helpfulness in responding to benign queries.



## **29. LLM in the Middle: A Systematic Review of Threats and Mitigations to Real-World LLM-based Systems**

cs.CR

37 pages, 8 figures, 13 tables

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10682v1) [paper-pdf](http://arxiv.org/pdf/2509.10682v1)

**Authors**: Vitor Hugo Galhardo Moia, Igor Jochem Sanz, Gabriel Antonio Fontes Rebello, Rodrigo Duarte de Meneses, Briland Hitaj, Ulf Lindqvist

**Abstract**: The success and wide adoption of generative AI (GenAI), particularly large language models (LLMs), has attracted the attention of cybercriminals seeking to abuse models, steal sensitive data, or disrupt services. Moreover, providing security to LLM-based systems is a great challenge, as both traditional threats to software applications and threats targeting LLMs and their integration must be mitigated. In this survey, we shed light on security and privacy concerns of such LLM-based systems by performing a systematic review and comprehensive categorization of threats and defensive strategies considering the entire software and LLM life cycles. We analyze real-world scenarios with distinct characteristics of LLM usage, spanning from development to operation. In addition, threats are classified according to their severity level and to which scenarios they pertain, facilitating the identification of the most relevant threats. Recommended defense strategies are systematically categorized and mapped to the corresponding life cycle phase and possible attack strategies they attenuate. This work paves the way for consumers and vendors to understand and efficiently mitigate risks during integration of LLMs in their respective solutions or organizations. It also enables the research community to benefit from the discussion of open challenges and edge cases that may hinder the secure and privacy-preserving adoption of LLM-based systems.



## **30. URL2Graph++: Unified Semantic-Structural-Character Learning for Malicious URL Detection**

cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10287v1) [paper-pdf](http://arxiv.org/pdf/2509.10287v1)

**Authors**: Ye Tian, Yifan Jia, Yanbin Wang, Jianguo Sun, Zhiquan Liu, Xiaowen Ling

**Abstract**: Malicious URL detection remains a major challenge in cybersecurity, primarily due to two factors: (1) the exponential growth of the Internet has led to an immense diversity of URLs, making generalized detection increasingly difficult; and (2) attackers are increasingly employing sophisticated obfuscation techniques to evade detection. We advocate that addressing these challenges fundamentally requires: (1) obtaining semantic understanding to improve generalization across vast and diverse URL sets, and (2) accurately modeling contextual relationships within the structural composition of URLs. In this paper, we propose a novel malicious URL detection method combining multi-granularity graph learning with semantic embedding to jointly capture semantic, character-level, and structural features for robust URL analysis. To model internal dependencies within URLs, we first construct dual-granularity URL graphs at both subword and character levels, where nodes represent URL tokens/characters and edges encode co-occurrence relationships. To obtain fine-grained embeddings, we initialize node representations using a character-level convolutional network. The two graphs are then processed through jointly trained Graph Convolutional Networks to learn consistent graph-level representations, enabling the model to capture complementary structural features that reflect co-occurrence patterns and character-level dependencies. Furthermore, we employ BERT to derive semantic representations of URLs for semantically aware understanding. Finally, we introduce a gated dynamic fusion network to combine the semantically enriched BERT representations with the jointly optimized graph vectors, further enhancing detection performance. We extensively evaluate our method across multiple challenging dimensions. Results show our method exceeds SOTA performance, including against large language models.



## **31. Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs**

cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.05367v2) [paper-pdf](http://arxiv.org/pdf/2509.05367v2)

**Authors**: Shei Pern Chua, Zhen Leng Thai, Teh Kai Jun, Xiao Li, Xiaolin Hu

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.



## **32. Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective**

cs.IR

Accepted at ACM RecSys 2025 (10 pages, 4 figures)

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2508.03703v2) [paper-pdf](http://arxiv.org/pdf/2508.03703v2)

**Authors**: Yubo Wang, Min Tang, Nuo Shen, Shujie Cui, Weiqing Wang

**Abstract**: The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems.



## **33. When Your Reviewer is an LLM: Biases, Divergence, and Prompt Injection Risks in Peer Review**

cs.CY

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.09912v1) [paper-pdf](http://arxiv.org/pdf/2509.09912v1)

**Authors**: Changjia Zhu, Junjie Xiong, Renkai Ma, Zhicong Lu, Yao Liu, Lingyao Li

**Abstract**: Peer review is the cornerstone of academic publishing, yet the process is increasingly strained by rising submission volumes, reviewer overload, and expertise mismatches. Large language models (LLMs) are now being used as "reviewer aids," raising concerns about their fairness, consistency, and robustness against indirect prompt injection attacks. This paper presents a systematic evaluation of LLMs as academic reviewers. Using a curated dataset of 1,441 papers from ICLR 2023 and NeurIPS 2022, we evaluate GPT-5-mini against human reviewers across ratings, strengths, and weaknesses. The evaluation employs structured prompting with reference paper calibration, topic modeling, and similarity analysis to compare review content. We further embed covert instructions into PDF submissions to assess LLMs' susceptibility to prompt injection. Our findings show that LLMs consistently inflate ratings for weaker papers while aligning more closely with human judgments on stronger contributions. Moreover, while overarching malicious prompts induce only minor shifts in topical focus, explicitly field-specific instructions successfully manipulate specific aspects of LLM-generated reviews. This study underscores both the promises and perils of integrating LLMs into peer review and points to the importance of designing safeguards that ensure integrity and trust in future review processes.



## **34. Advancing Security with Digital Twins: A Comprehensive Survey**

cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2505.17310v2) [paper-pdf](http://arxiv.org/pdf/2505.17310v2)

**Authors**: Blessing Airehenbuwa, Touseef Hasan, Souvika Sarkar, Ujjwal Guin

**Abstract**: The proliferation of electronic devices has greatly transformed every aspect of human life, such as communication, healthcare, transportation, and energy. Unfortunately, the global electronics supply chain is vulnerable to various attacks, including piracy of intellectual properties, tampering, counterfeiting, information leakage, side-channel, and fault injection attacks, due to the complex nature of electronic products and vulnerabilities present in them. Although numerous solutions have been proposed to address these threats, significant gaps remain, particularly in providing scalable and comprehensive protection against emerging attacks. Digital twin, a dynamic virtual replica of a physical system, has emerged as a promising solution to address these issues by providing backward traceability, end-to-end visibility, and continuous verification of component integrity and behavior. In this paper, we comprehensively present the latest digital twin-based security implementations, including their role in cyber-physical systems, Internet of Things, cryptographic systems, detection of counterfeit electronics, intrusion detection, fault injection, and side-channel leakage. This work considers these critical security use cases within a single study to offer researchers and practitioners a unified reference for securing hardware with digital twins. The paper also explores the integration of large language models with digital twins for enhanced security and discusses current challenges, solutions, and future research directions.



## **35. Steering MoE LLMs via Expert (De)Activation**

cs.CL

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09660v1) [paper-pdf](http://arxiv.org/pdf/2509.09660v1)

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan Rossi, Trung Bui, Hinrich Schütze, Nanyun Peng

**Abstract**: Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts.



## **36. Can LLMs Hack Enterprise Networks? Autonomous Assumed Breach Penetration-Testing Active Directory Networks**

cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2502.04227v3) [paper-pdf](http://arxiv.org/pdf/2502.04227v3)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Enterprise penetration-testing is often limited by high operational costs and the scarcity of human expertise. This paper investigates the feasibility and effectiveness of using Large Language Model (LLM)-driven autonomous systems to address these challenges in real-world Active Directory (AD) enterprise networks.   We introduce a novel prototype designed to employ LLMs to autonomously perform Assumed Breach penetration-testing against enterprise networks. Our system represents the first demonstration of a fully autonomous, LLM-driven framework capable of compromising accounts within a real-life Microsoft Active Directory testbed, GOAD.   We perform our empirical evaluation using five LLMs, comparing reasoning to non-reasoning models as well as including open-weight models. Through quantitative and qualitative analysis, incorporating insights from cybersecurity experts, we demonstrate that autonomous LLMs can effectively conduct Assumed Breach simulations. Key findings highlight their ability to dynamically adapt attack strategies, perform inter-context attacks (e.g., web-app audits, social engineering, and unstructured data analysis for credentials), and generate scenario-specific attack parameters like realistic password candidates. The prototype exhibits robust self-correction mechanisms, installing missing tools and rectifying invalid command generations.   We find that the associated costs are competitive with, and often significantly lower than, those incurred by professional human pen-testers, suggesting a path toward democratizing access to essential security testing for organizations with budgetary constraints. However, our research also illuminates existing limitations, including instances of LLM ``going down rabbit holes'', challenges in comprehensive information transfer between planning and execution modules, and critical safety concerns that necessitate human oversight.



## **37. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2411.00827v5) [paper-pdf](http://arxiv.org/pdf/2411.00827v5)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLJailbreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.VLJailbreakBench is publicly available at https://roywang021.github.io/VLJailbreakBench.



## **38. SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models**

cs.CL

Accepted to EMNLP 25 main

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2502.02787v2) [paper-pdf](http://arxiv.org/pdf/2502.02787v2)

**Authors**: Amirhossein Dabiriaghdam, Lele Wang

**Abstract**: The widespread adoption of large language models (LLMs) necessitates reliable methods to detect LLM-generated text. We introduce SimMark, a robust sentence-level watermarking algorithm that makes LLMs' outputs traceable without requiring access to model internals, making it compatible with both open and API-based LLMs. By leveraging the similarity of semantic sentence embeddings combined with rejection sampling to embed detectable statistical patterns imperceptible to humans, and employing a soft counting mechanism, SimMark achieves robustness against paraphrasing attacks. Experimental results demonstrate that SimMark sets a new benchmark for robust watermarking of LLM-generated content, surpassing prior sentence-level watermarking techniques in robustness, sampling efficiency, and applicability across diverse domains, all while maintaining the text quality and fluency.



## **39. ACE: A Security Architecture for LLM-Integrated App Systems**

cs.CR

25 pages, 13 figures, 8 tables; accepted by Network and Distributed  System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2504.20984v3) [paper-pdf](http://arxiv.org/pdf/2504.20984v3)

**Authors**: Evan Li, Tushin Mallick, Evan Rose, William Robertson, Alina Oprea, Cristina Nita-Rotaru

**Abstract**: LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution.   In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that ACE is secure against attacks from the InjecAgent and Agent Security Bench benchmarks for indirect prompt injection, and our newly introduced attacks. We also evaluate the utility of ACE in realistic environments, using the Tool Usage suite from the LangChain benchmark. Our architecture represents a significant advancement towards hardening LLM-based systems using system security principles.



## **40. Architecting Resilient LLM Agents: A Guide to Secure Plan-then-Execute Implementations**

cs.CR

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08646v1) [paper-pdf](http://arxiv.org/pdf/2509.08646v1)

**Authors**: Ron F. Del Rosario, Klaudia Krawiecka, Christian Schroeder de Witt

**Abstract**: As Large Language Model (LLM) agents become increasingly capable of automating complex, multi-step tasks, the need for robust, secure, and predictable architectural patterns is paramount. This paper provides a comprehensive guide to the ``Plan-then-Execute'' (P-t-E) pattern, an agentic design that separates strategic planning from tactical execution. We explore the foundational principles of P-t-E, detailing its core components - the Planner and the Executor - and its architectural advantages in predictability, cost-efficiency, and reasoning quality over reactive patterns like ReAct (Reason + Act). A central focus is placed on the security implications of this design, particularly its inherent resilience to indirect prompt injection attacks by establishing control-flow integrity. We argue that while P-t-E provides a strong foundation, a defense-in-depth strategy is necessary, and we detail essential complementary controls such as the Principle of Least Privilege, task-scoped tool access, and sandboxed code execution. To make these principles actionable, this guide provides detailed implementation blueprints and working code references for three leading agentic frameworks: LangChain (via LangGraph), CrewAI, and AutoGen. Each framework's approach to implementing the P-t-E pattern is analyzed, highlighting unique features like LangGraph's stateful graphs for re-planning, CrewAI's declarative tool scoping for security, and AutoGen's built-in Docker sandboxing. Finally, we discuss advanced patterns, including dynamic re-planning loops, parallel execution with Directed Acyclic Graphs (DAGs), and the critical role of Human-in-the-Loop (HITL) verification, to offer a complete strategic blueprint for architects, developers, and security engineers aiming to build production-grade, resilient, and trustworthy LLM agents.



## **41. Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors**

cs.CL

Accepted by EMNLP-2025

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2505.15337v3) [paper-pdf](http://arxiv.org/pdf/2505.15337v3)

**Authors**: Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Abstract**: The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.



## **42. TraceRAG: A LLM-Based Framework for Explainable Android Malware Detection and Behavior Analysis**

cs.SE

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08865v1) [paper-pdf](http://arxiv.org/pdf/2509.08865v1)

**Authors**: Guangyu Zhang, Xixuan Wang, Shiyu Sun, Peiyan Xiao, Kun Sun, Yanhai Xiong

**Abstract**: Sophisticated evasion tactics in malicious Android applications, combined with their intricate behavioral semantics, enable attackers to conceal malicious logic within legitimate functions, underscoring the critical need for robust and in-depth analysis frameworks. However, traditional analysis techniques often fail to recover deeply hidden behaviors or provide human-readable justifications for their decisions. Inspired by advances in large language models (LLMs), we introduce TraceRAG, a retrieval-augmented generation (RAG) framework that bridges natural language queries and Java code to deliver explainable malware detection and analysis. First, TraceRAG generates summaries of method-level code snippets, which are indexed in a vector database. At query time, behavior-focused questions retrieve the most semantically relevant snippets for deeper inspection. Finally, based on the multi-turn analysis results, TraceRAG produces human-readable reports that present the identified malicious behaviors and their corresponding code implementations. Experimental results demonstrate that our method achieves 96\% malware detection accuracy and 83.81\% behavior identification accuracy based on updated VirusTotal (VT) scans and manual verification. Furthermore, expert evaluation confirms the practical utility of the reports generated by TraceRAG.



## **43. ImportSnare: Directed "Code Manual" Hijacking in Retrieval-Augmented Code Generation**

cs.CR

This paper has been accepted by the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07941v1) [paper-pdf](http://arxiv.org/pdf/2509.07941v1)

**Authors**: Kai Ye, Liangcai Su, Chenxiong Qian

**Abstract**: Code generation has emerged as a pivotal capability of Large Language Models(LLMs), revolutionizing development efficiency for programmers of all skill levels. However, the complexity of data structures and algorithmic logic often results in functional deficiencies and security vulnerabilities in generated code, reducing it to a prototype requiring extensive manual debugging. While Retrieval-Augmented Generation (RAG) can enhance correctness and security by leveraging external code manuals, it simultaneously introduces new attack surfaces.   In this paper, we pioneer the exploration of attack surfaces in Retrieval-Augmented Code Generation (RACG), focusing on malicious dependency hijacking. We demonstrate how poisoned documentation containing hidden malicious dependencies (e.g., matplotlib_safe) can subvert RACG, exploiting dual trust chains: LLM reliance on RAG and developers' blind trust in LLM suggestions. To construct poisoned documents, we propose ImportSnare, a novel attack framework employing two synergistic strategies: 1)Position-aware beam search optimizes hidden ranking sequences to elevate poisoned documents in retrieval results, and 2)Multilingual inductive suggestions generate jailbreaking sequences to manipulate LLMs into recommending malicious dependencies. Through extensive experiments across Python, Rust, and JavaScript, ImportSnare achieves significant attack success rates (over 50% for popular libraries such as matplotlib and seaborn) in general, and is also able to succeed even when the poisoning ratio is as low as 0.01%, targeting both custom and real-world malicious packages. Our findings reveal critical supply chain risks in LLM-powered development, highlighting inadequate security alignment for code generation tasks. To support future research, we will release the multilingual benchmark suite and datasets. The project homepage is https://importsnare.github.io.



## **44. Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees**

cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07939v1) [paper-pdf](http://arxiv.org/pdf/2509.07939v1)

**Authors**: Katsuaki Nakano, Reza Feyyazi, Shanchieh Jay Yang, Michael Zuzak

**Abstract**: Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments



## **45. AgentSentinel: An End-to-End and Real-Time Security Defense Framework for Computer-Use Agents**

cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07764v1) [paper-pdf](http://arxiv.org/pdf/2509.07764v1)

**Authors**: Haitao Hu, Peng Chen, Yanpeng Zhao, Yuqi Chen

**Abstract**: Large Language Models (LLMs) have been increasingly integrated into computer-use agents, which can autonomously operate tools on a user's computer to accomplish complex tasks. However, due to the inherently unstable and unpredictable nature of LLM outputs, they may issue unintended tool commands or incorrect inputs, leading to potentially harmful operations. Unlike traditional security risks stemming from insecure user prompts, tool execution results from LLM-driven decisions introduce new and unique security challenges. These vulnerabilities span across all components of a computer-use agent. To mitigate these risks, we propose AgentSentinel, an end-to-end, real-time defense framework designed to mitigate potential security threats on a user's computer. AgentSentinel intercepts all sensitive operations within agent-related services and halts execution until a comprehensive security audit is completed. Our security auditing mechanism introduces a novel inspection process that correlates the current task context with system traces generated during task execution. To thoroughly evaluate AgentSentinel, we present BadComputerUse, a benchmark consisting of 60 diverse attack scenarios across six attack categories. The benchmark demonstrates a 87% average attack success rate on four state-of-the-art LLMs. Our evaluation shows that AgentSentinel achieves an average defense success rate of 79.6%, significantly outperforming all baseline defenses.



## **46. Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling**

cs.AI

Accepted to EMNLP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07617v1) [paper-pdf](http://arxiv.org/pdf/2509.07617v1)

**Authors**: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang

**Abstract**: Direct Prompt Injection (DPI) attacks pose a critical security threat to Large Language Models (LLMs) due to their low barrier of execution and high potential damage. To address the impracticality of existing white-box/gray-box methods and the poor transferability of black-box methods, we propose an activations-guided prompt injection attack framework. We first construct an Energy-based Model (EBM) using activations from a surrogate model to evaluate the quality of adversarial prompts. Guided by the trained EBM, we employ the token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize adversarial prompts, thereby enabling gradient-free black-box attacks. Experimental results demonstrate our superior cross-model transferability, achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6% improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen task scenarios. Interpretability analysis reveals a correlation between activations and attack effectiveness, highlighting the critical role of semantic patterns in transferable vulnerability exploitation.



## **47. A Decade-long Landscape of Advanced Persistent Threats: Longitudinal Analysis and Global Trends**

cs.CR

18 pages, 13 figures (including subfigures), 11 tables. To appear in  the Proceedings of the ACM Conference on Computer and Communications Security  (CCS) 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07457v1) [paper-pdf](http://arxiv.org/pdf/2509.07457v1)

**Authors**: Shakhzod Yuldoshkhujaev, Mijin Jeon, Doowon Kim, Nick Nikiforakis, Hyungjoon Koo

**Abstract**: An advanced persistent threat (APT) refers to a covert, long-term cyberattack, typically conducted by state-sponsored actors, targeting critical sectors and often remaining undetected for long periods. In response, collective intelligence from around the globe collaborates to identify and trace surreptitious activities, generating substantial documentation on APT campaigns publicly available on the web. While prior works predominantly focus on specific aspects of APT cases, such as detection, evaluation, cyber threat intelligence, and dataset creation, limited attention has been devoted to revisiting and investigating these scattered dossiers in a longitudinal manner. The objective of our study is to fill the gap by offering a macro perspective, connecting key insights and global trends in past APT attacks. We systematically analyze six reliable sources-three focused on technical reports and another three on threat actors-examining 1,509 APT dossiers (24,215 pages) spanning 2014-2023, and identifying 603 unique APT groups worldwide. To efficiently unearth relevant information, we employ a hybrid methodology that combines rule-based information retrieval with large-language-model-based search techniques. Our longitudinal analysis reveals shifts in threat actor activities, global attack vectors, changes in targeted sectors, and relationships between cyberattacks and significant events such as elections or wars, which provide insights into historical patterns in APT evolution. Over the past decade, 154 countries have been affected, primarily using malicious documents and spear phishing as dominant initial infiltration vectors, with a noticeable decline in zero-day exploitation since 2016. Furthermore, we present our findings through interactive visualization tools, such as an APT map or flow diagram, to facilitate intuitive understanding of global patterns and trends in APT activities.



## **48. OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models**

cs.CL

To appear at EMNLP 25 main conference

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.04416v2) [paper-pdf](http://arxiv.org/pdf/2505.04416v2)

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose \textbf{OBLIVIATE}, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA) ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: \emph{forget quality} (via a new document-level memorization score), \emph{model utility}, and \emph{fluency}. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios.



## **49. GRADA: Graph-based Reranking against Adversarial Documents Attack**

cs.IR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.07546v2) [paper-pdf](http://arxiv.org/pdf/2505.07546v2)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.



## **50. Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection**

cs.CR

Accepted as a paper at COLM 2025 Workshop on AI Agents: Capabilities  and Safety

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.15552v2) [paper-pdf](http://arxiv.org/pdf/2503.15552v2)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.



