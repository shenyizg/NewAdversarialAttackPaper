# Latest Large Language Model Attack Papers
**update at 2025-03-05 10:17:45**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. LLM-Safety Evaluations Lack Robustness**

cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02574v1) [paper-pdf](http://arxiv.org/pdf/2503.02574v1)

**Authors**: Tim Beyer, Sophie Xhonneux, Simon Geisler, Gauthier Gidel, Leo Schwinn, Stephan Günnemann

**Abstract**: In this paper, we argue that current safety alignment research efforts for large language models are hindered by many intertwined sources of noise, such as small datasets, methodological inconsistencies, and unreliable evaluation setups. This can, at times, make it impossible to evaluate and compare attacks and defenses fairly, thereby slowing progress. We systematically analyze the LLM safety evaluation pipeline, covering dataset curation, optimization strategies for automated red-teaming, response generation, and response evaluation using LLM judges. At each stage, we identify key issues and highlight their practical impact. We also propose a set of guidelines for reducing noise and bias in evaluations of future attack and defense papers. Lastly, we offer an opposing perspective, highlighting practical reasons for existing limitations. We believe that addressing the outlined problems in future research will improve the field's ability to generate easily comparable results and make measurable progress.



## **2. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2407.09164v5) [paper-pdf](http://arxiv.org/pdf/2407.09164v5)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully exploited to simplify and facilitate programming. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. In this paper, we reveal that both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process, which may not be practical; adversarial attacks struggle with fulfilling specific malicious purposes. To alleviate these problems, this paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an attack success rate of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an attack success rate of over 90%) in all threat cases, using only a 12-token non-functional perturbation.



## **3. Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents**

cs.CR

17 pages, 5 figures, 6 tables (NAACL 2025 Findings)

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.00061v2) [paper-pdf](http://arxiv.org/pdf/2503.00061v2)

**Authors**: Qiusi Zhan, Richard Fang, Henil Shalin Panchal, Daniel Kang

**Abstract**: Large Language Model (LLM) agents exhibit remarkable performance across diverse applications by using external tools to interact with environments. However, integrating external tools introduces security risks, such as indirect prompt injection (IPI) attacks. Despite defenses designed for IPI attacks, their robustness remains questionable due to insufficient testing against adaptive attacks. In this paper, we evaluate eight different defenses and bypass all of them using adaptive attacks, consistently achieving an attack success rate of over 50%. This reveals critical vulnerabilities in current defenses. Our research underscores the need for adaptive attack evaluation when designing defenses to ensure robustness and reliability. The code is available at https://github.com/uiuc-kang-lab/AdaptiveAttackAgent.



## **4. Confidential Prompting: Protecting User Prompts from Cloud LLM Providers**

cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2409.19134v3) [paper-pdf](http://arxiv.org/pdf/2409.19134v3)

**Authors**: In Gim, Caihua Li, Lin Zhong

**Abstract**: Our work tackles the challenge of securing user inputs in cloud-hosted large language model (LLM) serving while ensuring model confidentiality, output invariance, and compute efficiency. We introduce Secure Partitioned Decoding (SPD), which uses confidential computing to confine user prompts to a trusted execution environment (TEE), namely a confidential virtual machine (CVM), while allowing service providers to generate tokens efficiently. We also introduce a novel cryptographic method, Prompt Obfuscation (PO), to ensure robustness against reconstruction attacks on SPD. We demonstrate our approach preserves both prompt confidentiality and LLM serving efficiency. Our solution enables privacy-preserving cloud LLM serving that handles sensitive prompts, such as clinical records, financial data, and personal information.



## **5. De-identification is not enough: a comparison between de-identified and synthetic clinical notes**

cs.CL

https://www.nature.com/articles/s41598-024-81170-y

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2402.00179v2) [paper-pdf](http://arxiv.org/pdf/2402.00179v2)

**Authors**: Atiquer Rahman Sarkar, Yao-Shun Chuang, Noman Mohammed, Xiaoqian Jiang

**Abstract**: For sharing privacy-sensitive data, de-identification is commonly regarded as adequate for safeguarding privacy. Synthetic data is also being considered as a privacy-preserving alternative. Recent successes with numerical and tabular data generative models and the breakthroughs in large generative language models raise the question of whether synthetically generated clinical notes could be a viable alternative to real notes for research purposes. In this work, we demonstrated that (i) de-identification of real clinical notes does not protect records against a membership inference attack, (ii) proposed a novel approach to generate synthetic clinical notes using the current state-of-the-art large language models, (iii) evaluated the performance of the synthetically generated notes in a clinical domain task, and (iv) proposed a way to mount a membership inference attack where the target model is trained with synthetic data. We observed that when synthetically generated notes closely match the performance of real data, they also exhibit similar privacy concerns to the real data. Whether other approaches to synthetically generated clinical notes could offer better trade-offs and become a better alternative to sensitive real notes warrants further investigation.



## **6. Jailbreaking Safeguarded Text-to-Image Models via Large Language Models**

cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01839v1) [paper-pdf](http://arxiv.org/pdf/2503.01839v1)

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.



## **7. AutoAdvExBench: Benchmarking autonomous exploitation of adversarial example defenses**

cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01811v1) [paper-pdf](http://arxiv.org/pdf/2503.01811v1)

**Authors**: Nicholas Carlini, Javier Rando, Edoardo Debenedetti, Milad Nasr, Florian Tramèr

**Abstract**: We introduce AutoAdvExBench, a benchmark to evaluate if large language models (LLMs) can autonomously exploit defenses to adversarial examples. Unlike existing security benchmarks that often serve as proxies for real-world tasks, bench directly measures LLMs' success on tasks regularly performed by machine learning security experts. This approach offers a significant advantage: if a LLM could solve the challenges presented in bench, it would immediately present practical utility for adversarial machine learning researchers. We then design a strong agent that is capable of breaking 75% of CTF-like ("homework exercise") adversarial example defenses. However, we show that this agent is only able to succeed on 13% of the real-world defenses in our benchmark, indicating the large gap between difficulty in attacking "real" code, and CTF-like code. In contrast, a stronger LLM that can attack 21% of real defenses only succeeds on 54% of CTF-like defenses. We make this benchmark available at https://github.com/ethz-spylab/AutoAdvExBench.



## **8. Building Safe GenAI Applications: An End-to-End Overview of Red Teaming for Large Language Models**

cs.CL

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01742v1) [paper-pdf](http://arxiv.org/pdf/2503.01742v1)

**Authors**: Alberto Purpura, Sahil Wadhwa, Jesse Zymet, Akshay Gupta, Andy Luo, Melissa Kazemi Rad, Swapnil Shinde, Mohammad Shahed Sorower

**Abstract**: The rapid growth of Large Language Models (LLMs) presents significant privacy, security, and ethical concerns. While much research has proposed methods for defending LLM systems against misuse by malicious actors, researchers have recently complemented these efforts with an offensive approach that involves red teaming, i.e., proactively attacking LLMs with the purpose of identifying their vulnerabilities. This paper provides a concise and practical overview of the LLM red teaming literature, structured so as to describe a multi-component system end-to-end. To motivate red teaming we survey the initial safety needs of some high-profile LLMs, and then dive into the different components of a red teaming system as well as software packages for implementing them. We cover various attack methods, strategies for attack-success evaluation, metrics for assessing experiment outcomes, as well as a host of other considerations. Our survey will be useful for any reader who wants to rapidly obtain a grasp of the major red teaming concepts for their own use in practical applications.



## **9. Attacking Large Language Models with Projected Gradient Descent**

cs.LG

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2402.09154v2) [paper-pdf](http://arxiv.org/pdf/2402.09154v2)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Johannes Gasteiger, Stephan Günnemann

**Abstract**: Current LLM alignment methods are readily broken through specifically crafted adversarial prompts. While crafting adversarial prompts using discrete optimization is highly effective, such attacks typically use more than 100,000 LLM calls. This high computational cost makes them unsuitable for, e.g., quantitative analyses and adversarial training. To remedy this, we revisit Projected Gradient Descent (PGD) on the continuously relaxed input prompt. Although previous attempts with ordinary gradient-based attacks largely failed, we show that carefully controlling the error introduced by the continuous relaxation tremendously boosts their efficacy. Our PGD for LLMs is up to one order of magnitude faster than state-of-the-art discrete optimization to achieve the same devastating attack results.



## **10. PAPILLON: Efficient and Stealthy Fuzz Testing-Powered Jailbreaks for LLMs**

cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.14866v5) [paper-pdf](http://arxiv.org/pdf/2409.14866v5)

**Authors**: Xueluan Gong, Mingzhe Li, Yilin Zhang, Fengyuan Ran, Chen Chen, Yanjiao Chen, Qian Wang, Kwok-Yan Lam

**Abstract**: Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs. In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework called PAPILLON, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates,PAPILLON starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks. We evaluated PAPILLON on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, PAPILLONs achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60\%. Additionally, PAPILLON can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, PAPILLON can achieve over 78% attack success rate even with 100 tokens. Moreover, PAPILLON demonstrates transferability and is robust to state-of-the-art defenses. Code: https://github.com/aaFrostnova/Papillon



## **11. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2403.17710v4) [paper-pdf](http://arxiv.org/pdf/2403.17710v4)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.



## **12. Exploring Adversarial Robustness in Classification tasks using DNA Language Models**

cs.CL

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.19788v2) [paper-pdf](http://arxiv.org/pdf/2409.19788v2)

**Authors**: Hyunwoo Yoo, Haebin Shin, Kaidi Xu, Gail Rosen

**Abstract**: DNA Language Models, such as GROVER, DNABERT2 and the Nucleotide Transformer, operate on DNA sequences that inherently contain sequencing errors, mutations, and laboratory-induced noise, which may significantly impact model performance. Despite the importance of this issue, the robustness of DNA language models remains largely underexplored. In this paper, we comprehensivly investigate their robustness in DNA classification by applying various adversarial attack strategies: the character (nucleotide substitutions), word (codon modifications), and sentence levels (back-translation-based transformations) to systematically analyze model vulnerabilities. Our results demonstrate that DNA language models are highly susceptible to adversarial attacks, leading to significant performance degradation. Furthermore, we explore adversarial training method as a defense mechanism, which enhances both robustness and classification accuracy. This study highlights the limitations of DNA language models and underscores the necessity of robustness in bioinformatics.



## **13. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2502.08079v3) [paper-pdf](http://arxiv.org/pdf/2502.08079v3)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.



## **14. We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs**

cs.SE

To appear in the 2025 USENIX Security Symposium. 22 pages, 14  figures, 8 tables. Edited from original version for submission to a different  conference. No change to original results or findings

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2406.10279v3) [paper-pdf](http://arxiv.org/pdf/2406.10279v3)

**Authors**: Joseph Spracklen, Raveen Wijewickrama, A H M Nazmus Sakib, Anindya Maiti, Bimal Viswanath, Murtuza Jadliwala

**Abstract**: The reliance of popular programming languages such as Python and JavaScript on centralized package repositories and open-source software, combined with the emergence of code-generating Large Language Models (LLMs), has created a new type of threat to the software supply chain: package hallucinations. These hallucinations, which arise from fact-conflicting errors when generating code using LLMs, represent a novel form of package confusion attack that poses a critical threat to the integrity of the software supply chain. This paper conducts a rigorous and comprehensive evaluation of package hallucinations across different programming languages, settings, and parameters, exploring how a diverse set of models and configurations affect the likelihood of generating erroneous package recommendations and identifying the root causes of this phenomenon. Using 16 popular LLMs for code generation and two unique prompt datasets, we generate 576,000 code samples in two programming languages that we analyze for package hallucinations. Our findings reveal that that the average percentage of hallucinated packages is at least 5.2% for commercial models and 21.7% for open-source models, including a staggering 205,474 unique examples of hallucinated package names, further underscoring the severity and pervasiveness of this threat. To overcome this problem, we implement several hallucination mitigation strategies and show that they are able to significantly reduce the number of package hallucinations while maintaining code quality. Our experiments and findings highlight package hallucinations as a persistent and systemic phenomenon while using state-of-the-art LLMs for code generation, and a significant challenge which deserves the research community's urgent attention.



## **15. Boosting Jailbreak Attack with Momentum**

cs.LG

Accepted by ICASSP 2025

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2405.01229v2) [paper-pdf](http://arxiv.org/pdf/2405.01229v2)

**Authors**: Yihao Zhang, Zeming Wei

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, yet they remain vulnerable to adversarial attacks, notably the well-known jailbreak attack. In particular, the Greedy Coordinate Gradient (GCG) attack has demonstrated efficacy in exploiting this vulnerability by optimizing adversarial prompts through a combination of gradient heuristics and greedy search. However, the efficiency of this attack has become a bottleneck in the attacking process. To mitigate this limitation, in this paper we rethink the generation of the adversarial prompts through an optimization lens, aiming to stabilize the optimization process and harness more heuristic insights from previous optimization iterations. Specifically, we propose the \textbf{M}omentum \textbf{A}ccelerated G\textbf{C}G (\textbf{MAC}) attack, which integrates a momentum term into the gradient heuristic to boost and stabilize the random search for tokens in adversarial prompts. Experimental results showcase the notable enhancement achieved by MAC over baselines in terms of attack success rate and optimization efficiency. Moreover, we demonstrate that MAC can still exhibit superior performance for transfer attacks and models under defense mechanisms. Our code is available at https://github.com/weizeming/momentum-attack-llm.



## **16. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2502.18176v2) [paper-pdf](http://arxiv.org/pdf/2502.18176v2)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.



## **17. Output Length Effect on DeepSeek-R1's Safety in Forced Thinking**

cs.CL

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2503.01923v1) [paper-pdf](http://arxiv.org/pdf/2503.01923v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Victor Bian

**Abstract**: Large Language Models (LLMs) have demonstrated strong reasoning capabilities, but their safety under adversarial conditions remains a challenge. This study examines the impact of output length on the robustness of DeepSeek-R1, particularly in Forced Thinking scenarios. We analyze responses across various adversarial prompts and find that while longer outputs can improve safety through self-correction, certain attack types exploit extended generations. Our findings suggest that output length should be dynamically controlled to balance reasoning effectiveness and security. We propose reinforcement learning-based policy adjustments and adaptive token length regulation to enhance LLM safety.



## **18. SeqAR: Jailbreak LLMs with Sequential Auto-Generated Characters**

cs.CR

Accepted by NAACL 2025

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2407.01902v2) [paper-pdf](http://arxiv.org/pdf/2407.01902v2)

**Authors**: Yan Yang, Zeguan Xiao, Xin Lu, Hongru Wang, Xuetao Wei, Hailiang Huang, Guanhua Chen, Yun Chen

**Abstract**: The widespread applications of large language models (LLMs) have brought about concerns regarding their potential misuse. Although aligned with human preference data before release, LLMs remain vulnerable to various malicious attacks. In this paper, we adopt a red-teaming strategy to enhance LLM safety and introduce SeqAR, a simple yet effective framework to design jailbreak prompts automatically. The SeqAR framework generates and optimizes multiple jailbreak characters and then applies sequential jailbreak characters in a single query to bypass the guardrails of the target LLM. Different from previous work which relies on proprietary LLMs or seed jailbreak templates crafted by human expertise, SeqAR can generate and optimize the jailbreak prompt in a cold-start scenario using open-sourced LLMs without any seed jailbreak templates. Experimental results show that SeqAR achieves attack success rates of 88% and 60% in bypassing the safety alignment of GPT-3.5-1106 and GPT-4, respectively. Furthermore, we extensively evaluate the transferability of the generated templates across different LLMs and held-out malicious requests, while also exploring defense strategies against the jailbreak attack designed by SeqAR.



## **19. Unmasking Digital Falsehoods: A Comparative Analysis of LLM-Based Misinformation Detection Strategies**

cs.CL

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2503.00724v1) [paper-pdf](http://arxiv.org/pdf/2503.00724v1)

**Authors**: Tianyi Huang, Jingyuan Yi, Peiyang Yu, Xiaochuan Xu

**Abstract**: The proliferation of misinformation on social media has raised significant societal concerns, necessitating robust detection mechanisms. Large Language Models such as GPT-4 and LLaMA2 have been envisioned as possible tools for detecting misinformation based on their advanced natural language understanding and reasoning capabilities. This paper conducts a comparison of LLM-based approaches to detecting misinformation between text-based, multimodal, and agentic approaches. We evaluate the effectiveness of fine-tuned models, zero-shot learning, and systematic fact-checking mechanisms in detecting misinformation across different topic domains like public health, politics, and finance. We also discuss scalability, generalizability, and explainability of the models and recognize key challenges such as hallucination, adversarial attacks on misinformation, and computational resources. Our findings point towards the importance of hybrid approaches that pair structured verification protocols with adaptive learning techniques to enhance detection accuracy and explainability. The paper closes by suggesting potential avenues of future work, including real-time tracking of misinformation, federated learning, and cross-platform detection models.



## **20. Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore**

cs.CL

COLING 2025

**SubmitDate**: 2025-03-01    [abs](http://arxiv.org/abs/2405.04286v2) [paper-pdf](http://arxiv.org/pdf/2405.04286v2)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xuebo Liu, Lidia S. Chao, Min Zhang

**Abstract**: The efficacy of detectors for texts generated by large language models (LLMs) substantially depends on the availability of large-scale training data. However, white-box zero-shot detectors, which require no such data, are limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose a simple yet effective black-box zero-shot detection approach based on the observation that, from the perspective of LLMs, human-written texts typically contain more grammatical errors than LLM-generated texts. This approach involves calculating the Grammar Error Correction Score (GECScore) for the given text to differentiate between human-written and LLM-generated text. Experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.62% across XSum and Writing Prompts dataset. Additionally, our approach demonstrates strong reliability in the wild, exhibiting robust generalization and resistance to paraphrasing attacks. Data and code are available at: https://github.com/NLP2CT/GECScore.



## **21. Modification and Generated-Text Detection: Achieving Dual Detection Capabilities for the Outputs of LLM by Watermark**

cs.CR

**SubmitDate**: 2025-03-01    [abs](http://arxiv.org/abs/2502.08332v2) [paper-pdf](http://arxiv.org/pdf/2502.08332v2)

**Authors**: Yuhang Cai, Yaofei Wang, Donghui Hu, Chen Gu

**Abstract**: The development of large language models (LLMs) has raised concerns about potential misuse. One practical solution is to embed a watermark in the text, allowing ownership verification through watermark extraction. Existing methods primarily focus on defending against modification attacks, often neglecting other spoofing attacks. For example, attackers can alter the watermarked text to produce harmful content without compromising the presence of the watermark, which could lead to false attribution of this malicious content to the LLM. This situation poses a serious threat to the LLMs service providers and highlights the significance of achieving modification detection and generated-text detection simultaneously. Therefore, we propose a technique to detect modifications in text for unbiased watermark which is sensitive to modification. We introduce a new metric called ``discarded tokens", which measures the number of tokens not included in watermark detection. When a modification occurs, this metric changes and can serve as evidence of the modification. Additionally, we improve the watermark detection process and introduce a novel method for unbiased watermark. Our experiments demonstrate that we can achieve effective dual detection capabilities: modification detection and generated-text detection by watermark.



## **22. Safeguarding AI Agents: Developing and Analyzing Safety Architectures**

cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2409.03793v3) [paper-pdf](http://arxiv.org/pdf/2409.03793v3)

**Authors**: Ishaan Domkundwar, Mukunda N S, Ishaan Bhola, Riddhik Kochhar

**Abstract**: AI agents, specifically powered by large language models, have demonstrated exceptional capabilities in various applications where precision and efficacy are necessary. However, these agents come with inherent risks, including the potential for unsafe or biased actions, vulnerability to adversarial attacks, lack of transparency, and tendency to generate hallucinations. As AI agents become more prevalent in critical sectors of the industry, the implementation of effective safety protocols becomes increasingly important. This paper addresses the critical need for safety measures in AI systems, especially ones that collaborate with human teams. We propose and evaluate three frameworks to enhance safety protocols in AI agent systems: an LLM-powered input-output filter, a safety agent integrated within the system, and a hierarchical delegation-based system with embedded safety checks. Our methodology involves implementing these frameworks and testing them against a set of unsafe agentic use cases, providing a comprehensive evaluation of their effectiveness in mitigating risks associated with AI agent deployment. We conclude that these frameworks can significantly strengthen the safety and security of AI agent systems, minimizing potential harmful actions or outputs. Our work contributes to the ongoing effort to create safe and reliable AI applications, particularly in automated operations, and provides a foundation for developing robust guardrails to ensure the responsible use of AI agents in real-world applications.



## **23. UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning**

cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2503.01908v1) [paper-pdf](http://arxiv.org/pdf/2503.01908v1)

**Authors**: Jiawei Zhang, Shuang Yang, Bo Li

**Abstract**: Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for handling complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements also amplify the risks of adversarial attacks, particularly when LLM agents can access sensitive external functionalities. Moreover, because LLM agents engage in extensive reasoning or planning before executing final actions, manipulating them into performing targeted malicious actions or invoking specific tools remains a significant challenge. Consequently, directly embedding adversarial strings in malicious instructions or injecting malicious prompts into tool interactions has become less effective against modern LLM agents. In this work, we present UDora, a unified red teaming framework designed for LLM Agents that dynamically leverages the agent's own reasoning processes to compel it toward malicious behavior. Specifically, UDora first samples the model's reasoning for the given task, then automatically identifies multiple optimal positions within these reasoning traces to insert targeted perturbations. Subsequently, it uses the modified reasoning as the objective to optimize the adversarial strings. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets.



## **24. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

cs.CL

28 pages, 10 figures, 7 tables

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2503.00187v1) [paper-pdf](http://arxiv.org/pdf/2503.00187v1)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are highly vulnerable to jailbreaking attacks, wherein adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off between safety and helpfulness under different multi-turn jailbreak methods. Our code is available at https://github.com/HanjiangHu/NBF-LLM .



## **25. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

cs.AI

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2407.00075v5) [paper-pdf](http://arxiv.org/pdf/2407.00075v5)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.



## **26. Learning diverse attacks on large language models for robust red-teaming and safety tuning**

cs.CL

ICLR 2025

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2405.18540v2) [paper-pdf](http://arxiv.org/pdf/2405.18540v2)

**Authors**: Seanie Lee, Minsu Kim, Lynn Cherif, David Dobre, Juho Lee, Sung Ju Hwang, Kenji Kawaguchi, Gauthier Gidel, Yoshua Bengio, Nikolay Malkin, Moksh Jain

**Abstract**: Red-teaming, or identifying prompts that elicit harmful responses, is a critical step in ensuring the safe and responsible deployment of large language models (LLMs). Developing effective protection against many modes of attack prompts requires discovering diverse attacks. Automated red-teaming typically uses reinforcement learning to fine-tune an attacker language model to generate prompts that elicit undesirable responses from a target LLM, as measured, for example, by an auxiliary toxicity classifier. We show that even with explicit regularization to favor novelty and diversity, existing approaches suffer from mode collapse or fail to generate effective attacks. As a flexible and probabilistically principled alternative, we propose to use GFlowNet fine-tuning, followed by a secondary smoothing phase, to train the attacker model to generate diverse and effective attack prompts. We find that the attacks generated by our method are effective against a wide range of target LLMs, both with and without safety tuning, and transfer well between target LLMs. Finally, we demonstrate that models safety-tuned using a dataset of red-teaming prompts generated by our method are robust to attacks from other RL-based red-teaming approaches.



## **27. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2406.04755v4) [paper-pdf](http://arxiv.org/pdf/2406.04755v4)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing that our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.



## **28. FC-Attack: Jailbreaking Large Vision-Language Models via Auto-Generated Flowcharts**

cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.21059v1) [paper-pdf](http://arxiv.org/pdf/2502.21059v1)

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Jihui Guo, Xinlei He

**Abstract**: Large Vision-Language Models (LVLMs) have become powerful and widely adopted in some practical applications. However, recent research has revealed their vulnerability to multimodal jailbreak attacks, whereby the model can be induced to generate harmful content, leading to safety risks. Although most LVLMs have undergone safety alignment, recent research shows that the visual modality is still vulnerable to jailbreak attacks. In our work, we discover that by using flowcharts with partially harmful information, LVLMs can be induced to provide additional harmful details. Based on this, we propose a jailbreak attack method based on auto-generated flowcharts, FC-Attack. Specifically, FC-Attack first fine-tunes a pre-trained LLM to create a step-description generator based on benign datasets. The generator is then used to produce step descriptions corresponding to a harmful query, which are transformed into flowcharts in 3 different shapes (vertical, horizontal, and S-shaped) as visual prompts. These flowcharts are then combined with a benign textual prompt to execute a jailbreak attack on LVLMs. Our evaluations using the Advbench dataset show that FC-Attack achieves over 90% attack success rates on Gemini-1.5, Llaval-Next, Qwen2-VL, and InternVL-2.5 models, outperforming existing LVLM jailbreak methods. Additionally, we investigate factors affecting the attack performance, including the number of steps and the font styles in the flowcharts. Our evaluation shows that FC-Attack can improve the jailbreak performance from 4% to 28% in Claude-3.5 by changing the font style. To mitigate the attack, we explore several defenses and find that AdaShield can largely reduce the jailbreak performance but with the cost of utility drop.



## **29. Behind the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models**

cs.CR

12 pages. 6 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.19883v2) [paper-pdf](http://arxiv.org/pdf/2502.19883v2)

**Authors**: Sibo Yi, Tianshuo Cong, Xinlei He, Qi Li, Jiaxing Song

**Abstract**: Small language models (SLMs) have become increasingly prominent in the deployment on edge devices due to their high efficiency and low computational cost. While researchers continue to advance the capabilities of SLMs through innovative training strategies and model compression techniques, the security risks of SLMs have received considerably less attention compared to large language models (LLMs).To fill this gap, we provide a comprehensive empirical study to evaluate the security performance of 13 state-of-the-art SLMs under various jailbreak attacks. Our experiments demonstrate that most SLMs are quite susceptible to existing jailbreak attacks, while some of them are even vulnerable to direct harmful prompts.To address the safety concerns, we evaluate several representative defense methods and demonstrate their effectiveness in enhancing the security of SLMs. We further analyze the potential security degradation caused by different SLM techniques including architecture compression, quantization, knowledge distillation, and so on. We expect that our research can highlight the security challenges of SLMs and provide valuable insights to future work in developing more robust and secure SLMs.



## **30. LoRec: Large Language Model for Robust Sequential Recommendation against Poisoning Attacks**

cs.IR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2401.17723v2) [paper-pdf](http://arxiv.org/pdf/2401.17723v2)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Sequential recommender systems stand out for their ability to capture users' dynamic interests and the patterns of item-to-item transitions. However, the inherent openness of sequential recommender systems renders them vulnerable to poisoning attacks, where fraudulent users are injected into the training data to manipulate learned patterns. Traditional defense strategies predominantly depend on predefined assumptions or rules extracted from specific known attacks, limiting their generalizability to unknown attack types. To solve the above problems, considering the rich open-world knowledge encapsulated in Large Language Models (LLMs), our research initially focuses on the capabilities of LLMs in the detection of unknown fraudulent activities within recommender systems, a strategy we denote as LLM4Dec. Empirical evaluations demonstrate the substantial capability of LLMs in identifying unknown fraudsters, leveraging their expansive, open-world knowledge.   Building upon this, we propose the integration of LLMs into defense strategies to extend their effectiveness beyond the confines of known attacks. We propose LoRec, an advanced framework that employs LLM-Enhanced Calibration to strengthen the robustness of sequential recommender systems against poisoning attacks. LoRec integrates an LLM-enhanced CalibraTor (LCT) that refines the training process of sequential recommender systems with knowledge derived from LLMs, applying a user-wise reweighting to diminish the impact of fraudsters injected by attacks. By incorporating LLMs' open-world knowledge, the LCT effectively converts the limited, specific priors or rules into a more general pattern of fraudsters, offering improved defenses against poisoning attacks. Our comprehensive experiments validate that LoRec, as a general framework, significantly strengthens the robustness of sequential recommender systems.



## **31. Efficient Jailbreaking of Large Models by Freeze Training: Lower Layers Exhibit Greater Sensitivity to Harmful Content**

cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.20952v1) [paper-pdf](http://arxiv.org/pdf/2502.20952v1)

**Authors**: Hongyuan Shen, Min Zheng, Jincheng Wang, Yang Zhao

**Abstract**: With the widespread application of Large Language Models across various domains, their security issues have increasingly garnered significant attention from both academic and industrial communities. This study conducts sampling and normalization of the parameters of the LLM to generate visual representations and heatmaps of parameter distributions, revealing notable discrepancies in parameter distributions among certain layers within the hidden layers. Further analysis involves calculating statistical metrics for each layer, followed by the computation of a Comprehensive Sensitivity Score based on these metrics, which identifies the lower layers as being particularly sensitive to the generation of harmful content. Based on this finding, we employ a Freeze training strategy, selectively performing Supervised Fine-Tuning only on the lower layers. Experimental results demonstrate that this method significantly reduces training duration and GPU memory consumption while maintaining a high jailbreak success rate and a high harm score, outperforming the results achieved by applying the LoRA method for SFT across all layers. Additionally, the method has been successfully extended to other open-source large models, validating its generality and effectiveness across different model architectures. Furthermore, we compare our method with ohter jailbreak method, demonstrating the superior performance of our approach. By innovatively proposing a method to statistically analyze and compare large model parameters layer by layer, this study provides new insights into the interpretability of large models. These discoveries emphasize the necessity of continuous research and the implementation of adaptive security measures in the rapidly evolving field of LLMs to prevent potential jailbreak attack risks, thereby promoting the development of more robust and secure LLMs.



## **32. Beyond Natural Language Perplexity: Detecting Dead Code Poisoning in Code Generation Datasets**

cs.CL

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.20246v2) [paper-pdf](http://arxiv.org/pdf/2502.20246v2)

**Authors**: Chi-Chien Tsai, Chia-Mu Yu, Ying-Dar Lin, Yu-Sung Wu, Wei-Bin Lee

**Abstract**: The increasing adoption of large language models (LLMs) for code-related tasks has raised concerns about the security of their training datasets. One critical threat is dead code poisoning, where syntactically valid but functionally redundant code is injected into training data to manipulate model behavior. Such attacks can degrade the performance of neural code search systems, leading to biased or insecure code suggestions. Existing detection methods, such as token-level perplexity analysis, fail to effectively identify dead code due to the structural and contextual characteristics of programming languages. In this paper, we propose DePA (Dead Code Perplexity Analysis), a novel line-level detection and cleansing method tailored to the structural properties of code. DePA computes line-level perplexity by leveraging the contextual relationships between code lines and identifies anomalous lines by comparing their perplexity to the overall distribution within the file. Our experiments on benchmark datasets demonstrate that DePA significantly outperforms existing methods, achieving 0.14-0.19 improvement in detection F1-score and a 44-65% increase in poisoned segment localization precision. Furthermore, DePA enhances detection speed by 0.62-23x, making it practical for large-scale dataset cleansing. Overall, by addressing the unique challenges of dead code poisoning, DePA provides a robust and efficient solution for safeguarding the integrity of code generation model training datasets.



## **33. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.19820v2) [paper-pdf](http://arxiv.org/pdf/2502.19820v2)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.



## **34. FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering**

cs.CR

20 pages (including bibliography and Appendix), Submitted to ACM CCS  '24

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2310.16152v3) [paper-pdf](http://arxiv.org/pdf/2310.16152v3)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz

**Abstract**: Federated learning (FL) has become a key component in various language modeling applications such as machine translation, next-word prediction, and medical record analysis. These applications are trained on datasets from many FL participants that often include privacy-sensitive data, such as healthcare records, phone/credit card numbers, login credentials, etc. Although FL enables computation without necessitating clients to share their raw data, determining the extent of privacy leakage in federated language models is challenging and not straightforward. Moreover, existing attacks aim to extract data regardless of how sensitive or naive it is. To fill this research gap, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated large language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other users in FL even without any cooperation from the server. Our best-performing method improves the membership inference recall by 29% and achieves up to 71% private data reconstruction, evidently outperforming existing attacks with stronger assumptions of adversary capabilities.



## **35. Backdooring Vision-Language Models with Out-Of-Distribution Data**

cs.CV

ICLR 2025

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2410.01264v2) [paper-pdf](http://arxiv.org/pdf/2410.01264v2)

**Authors**: Weimin Lyu, Jiachen Yao, Saumya Gupta, Lu Pang, Tao Sun, Lingjie Yi, Lijie Hu, Haibin Ling, Chao Chen

**Abstract**: The emergence of Vision-Language Models (VLMs) represents a significant advancement in integrating computer vision with Large Language Models (LLMs) to generate detailed text descriptions from visual inputs. Despite their growing importance, the security of VLMs, particularly against backdoor attacks, is under explored. Moreover, prior works often assume attackers have access to the original training data, which is often unrealistic. In this paper, we address a more practical and challenging scenario where attackers must rely solely on Out-Of-Distribution (OOD) data. We introduce VLOOD (Backdooring Vision-Language Models with Out-of-Distribution Data), a novel approach with two key contributions: (1) demonstrating backdoor attacks on VLMs in complex image-to-text tasks while minimizing degradation of the original semantics under poisoned inputs, and (2) proposing innovative techniques for backdoor injection without requiring any access to the original training data. Our evaluation on image captioning and visual question answering (VQA) tasks confirms the effectiveness of VLOOD, revealing a critical security vulnerability in VLMs and laying the foundation for future research on securing multimodal models against sophisticated threats.



## **36. SQL Injection Jailbreak: A Structural Disaster of Large Language Models**

cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2411.01565v5) [paper-pdf](http://arxiv.org/pdf/2411.01565v5)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality into various domains, generating substantial social and economic benefits. However, jailbreaking, a form of attack that induces LLMs to produce harmful content through carefully crafted prompts, presents a significant challenge to the safe and trustworthy development of LLMs. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak methods and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. For open-source models, SIJ achieves near 100\% attack success rates on five well-known LLMs on the AdvBench and HEx-PHI, while incurring lower time costs compared to previous methods. For closed-source models, SIJ achieves an average attack success rate over 85\% across five models in the GPT and Doubao series. Additionally, SIJ exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.



## **37. LLMs Have Rhythm: Fingerprinting Large Language Models Using Inter-Token Times and Network Traffic Analysis**

cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.20589v1) [paper-pdf](http://arxiv.org/pdf/2502.20589v1)

**Authors**: Saeif Alhazbi, Ahmed Mohamed Hussain, Gabriele Oligeri, Panos Papadimitratos

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into many technological ecosystems across various domains and industries, identifying which model is deployed or being interacted with is critical for the security and trustworthiness of the systems. Current verification methods typically rely on analyzing the generated output to determine the source model. However, these techniques are susceptible to adversarial attacks, operate in a post-hoc manner, and may require access to the model weights to inject a verifiable fingerprint. In this paper, we propose a novel passive and non-invasive fingerprinting technique that operates in real-time and remains effective even under encrypted network traffic conditions. Our method leverages the intrinsic autoregressive generation nature of language models, which generate text one token at a time based on all previously generated tokens, creating a unique temporal pattern like a rhythm or heartbeat that persists even when the output is streamed over a network. We find that measuring the Inter-Token Times (ITTs)-time intervals between consecutive tokens-can identify different language models with high accuracy. We develop a Deep Learning (DL) pipeline to capture these timing patterns using network traffic analysis and evaluate it on 16 Small Language Models (SLMs) and 10 proprietary LLMs across different deployment scenarios, including local host machine (GPU/CPU), Local Area Network (LAN), Remote Network, and Virtual Private Network (VPN). The experimental results confirm that our proposed technique is effective and maintains high accuracy even when tested in different network conditions. This work opens a new avenue for model identification in real-world scenarios and contributes to more secure and trustworthy language model deployment.



## **38. Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks**

cs.CL

15 pages, 6 figures, 7 tables

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2410.18210v2) [paper-pdf](http://arxiv.org/pdf/2410.18210v2)

**Authors**: Samuele Poppi, Zheng-Xin Yong, Yifei He, Bobbie Chern, Han Zhao, Aobo Yang, Jianfeng Chi

**Abstract**: Recent advancements in Large Language Models (LLMs) have sparked widespread concerns about their safety. Recent work demonstrates that safety alignment of LLMs can be easily removed by fine-tuning with a few adversarially chosen instruction-following examples, i.e., fine-tuning attacks. We take a further step to understand fine-tuning attacks in multilingual LLMs. We first discover cross-lingual generalization of fine-tuning attacks: using a few adversarially chosen instruction-following examples in one language, multilingual LLMs can also be easily compromised (e.g., multilingual LLMs fail to refuse harmful prompts in other languages). Motivated by this finding, we hypothesize that safety-related information is language-agnostic and propose a new method termed Safety Information Localization (SIL) to identify the safety-related information in the model parameter space. Through SIL, we validate this hypothesis and find that only changing 20% of weight parameters in fine-tuning attacks can break safety alignment across all languages. Furthermore, we provide evidence to the alternative pathways hypothesis for why freezing safety-related parameters does not prevent fine-tuning attacks, and we demonstrate that our attack vector can still jailbreak LLMs adapted to new languages.



## **39. Trustworthy AI-Generative Content for Intelligent Network Service: Robustness, Security, and Fairness**

cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2405.05930v2) [paper-pdf](http://arxiv.org/pdf/2405.05930v2)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Xiang Chen, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have revolutionized content creation. High-speed next-generation communication technology is an ideal platform for providing powerful AIGC network services. At the same time, advanced AIGC techniques can also make future network services more intelligent, especially various online content generation services. However, the significant untrustworthiness concerns of current AIGC models, such as robustness, security, and fairness, greatly affect the credibility of intelligent network services, especially in ensuring secure AIGC services. This paper proposes TrustGAIN, a trustworthy AIGC framework that incorporates robust, secure, and fair network services. We first discuss the robustness to adversarial attacks faced by AIGC models in network systems and the corresponding protection issues. Subsequently, we emphasize the importance of avoiding unsafe and illegal services and ensuring the fairness of the AIGC network services. Then as a case study, we propose a novel sentiment analysis-based detection method to guide the robust detection of unsafe content in network services. We conduct our experiments on fake news, malicious code, and unsafe review datasets to represent LLM application scenarios. Our results indicate that TrustGAIN is an exploration of future networks that can support trustworthy AIGC network services.



## **40. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

cs.CY

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12659v3) [paper-pdf](http://arxiv.org/pdf/2502.12659v3)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.



## **41. Tokens for Learning, Tokens for Unlearning: Mitigating Membership Inference Attacks in Large Language Models via Dual-Purpose Training**

cs.LG

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19726v1) [paper-pdf](http://arxiv.org/pdf/2502.19726v1)

**Authors**: Toan Tran, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) have become the backbone of modern natural language processing but pose privacy concerns about leaking sensitive training data. Membership inference attacks (MIAs), which aim to infer whether a sample is included in a model's training dataset, can serve as a foundation for broader privacy threats. Existing defenses designed for traditional classification models do not account for the sequential nature of text data. As a result, they either require significant computational resources or fail to effectively mitigate privacy risks in LLMs. In this work, we propose a lightweight yet effective empirical privacy defense for protecting training data of language modeling by leveraging the token-specific characteristics. By analyzing token dynamics during training, we propose a token selection strategy that categorizes tokens into hard tokens for learning and memorized tokens for unlearning. Subsequently, our training-phase defense optimizes a novel dual-purpose token-level loss to achieve a Pareto-optimal balance between utility and privacy. Extensive experiments demonstrate that our approach not only provides strong protection against MIAs but also improves language modeling performance by around 10\% across various LLM architectures and datasets compared to the baselines.



## **42. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.18290v2) [paper-pdf](http://arxiv.org/pdf/2502.18290v2)

**Authors**: Zhaoyi Liu, Huan Zhang

**Abstract**: Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.



## **43. M-LLM Based Video Frame Selection for Efficient Video Understanding**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19680v1) [paper-pdf](http://arxiv.org/pdf/2502.19680v1)

**Authors**: Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, Trishul Chilimbi

**Abstract**: Recent advances in Multi-Modal Large Language Models (M-LLMs) show promising results in video reasoning. Popular Multi-Modal Large Language Model (M-LLM) frameworks usually apply naive uniform sampling to reduce the number of video frames that are fed into an M-LLM, particularly for long context videos. However, it could lose crucial context in certain periods of a video, so that the downstream M-LLM may not have sufficient visual information to answer a question. To attack this pain point, we propose a light-weight M-LLM -based frame selection method that adaptively select frames that are more relevant to users' queries. In order to train the proposed frame selector, we introduce two supervision signals (i) Spatial signal, where single frame importance score by prompting a M-LLM; (ii) Temporal signal, in which multiple frames selection by prompting Large Language Model (LLM) using the captions of all frame candidates. The selected frames are then digested by a frozen downstream video M-LLM for visual reasoning and question answering. Empirical results show that the proposed M-LLM video frame selector improves the performances various downstream video Large Language Model (video-LLM) across medium (ActivityNet, NExT-QA) and long (EgoSchema, LongVideoBench) context video question answering benchmarks.



## **44. Improving Adversarial Transferability in MLLMs via Dynamic Vision-Language Alignment Attack**

cs.CV

arXiv admin note: text overlap with arXiv:2403.09766

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19672v1) [paper-pdf](http://arxiv.org/pdf/2502.19672v1)

**Authors**: Chenhe Gu, Jindong Gu, Andong Hua, Yao Qin

**Abstract**: Multimodal Large Language Models (MLLMs), built upon LLMs, have recently gained attention for their capabilities in image recognition and understanding. However, while MLLMs are vulnerable to adversarial attacks, the transferability of these attacks across different models remains limited, especially under targeted attack setting. Existing methods primarily focus on vision-specific perturbations but struggle with the complex nature of vision-language modality alignment. In this work, we introduce the Dynamic Vision-Language Alignment (DynVLA) Attack, a novel approach that injects dynamic perturbations into the vision-language connector to enhance generalization across diverse vision-language alignment of different models. Our experimental results show that DynVLA significantly improves the transferability of adversarial examples across various MLLMs, including BLIP2, InstructBLIP, MiniGPT4, LLaVA, and closed-source models such as Gemini.



## **45. H-CoT: Hijacking the Chain-of-Thought Safety Reasoning Mechanism to Jailbreak Large Reasoning Models, Including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking**

cs.CL

Website: https://maliciouseducator.org/

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12893v2) [paper-pdf](http://arxiv.org/pdf/2502.12893v2)

**Authors**: Martin Kuo, Jianyi Zhang, Aolin Ding, Qinsi Wang, Louis DiValentin, Yujia Bao, Wei Wei, Hai Li, Yiran Chen

**Abstract**: Large Reasoning Models (LRMs) have recently extended their powerful reasoning capabilities to safety checks-using chain-of-thought reasoning to decide whether a request should be answered. While this new approach offers a promising route for balancing model utility and safety, its robustness remains underexplored. To address this gap, we introduce Malicious-Educator, a benchmark that disguises extremely dangerous or malicious requests beneath seemingly legitimate educational prompts. Our experiments reveal severe security flaws in popular commercial-grade LRMs, including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking. For instance, although OpenAI's o1 model initially maintains a high refusal rate of about 98%, subsequent model updates significantly compromise its safety; and attackers can easily extract criminal strategies from DeepSeek-R1 and Gemini 2.0 Flash Thinking without any additional tricks. To further highlight these vulnerabilities, we propose Hijacking Chain-of-Thought (H-CoT), a universal and transferable attack method that leverages the model's own displayed intermediate reasoning to jailbreak its safety reasoning mechanism. Under H-CoT, refusal rates sharply decline-dropping from 98% to below 2%-and, in some instances, even transform initially cautious tones into ones that are willing to provide harmful content. We hope these findings underscore the urgent need for more robust safety mechanisms to preserve the benefits of advanced reasoning capabilities without compromising ethical standards.



## **46. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**

cs.CR

Accepted at USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2404.01833v3) [paper-pdf](http://arxiv.org/pdf/2404.01833v3)

**Authors**: Mark Russinovich, Ahmed Salem, Ronen Eldan

**Abstract**: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.



## **47. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

cs.CR

15 pages, 12 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19041v1) [paper-pdf](http://arxiv.org/pdf/2502.19041v1)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.



## **48. Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models**

cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18943v1) [paper-pdf](http://arxiv.org/pdf/2502.18943v1)

**Authors**: Yu He, Boheng Li, Liu Liu, Zhongjie Ba, Wei Dong, Yiming Li, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Membership Inference Attacks (MIAs) aim to predict whether a data sample belongs to the model's training set or not. Although prior research has extensively explored MIAs in Large Language Models (LLMs), they typically require accessing to complete output logits (\ie, \textit{logits-based attacks}), which are usually not available in practice. In this paper, we study the vulnerability of pre-trained LLMs to MIAs in the \textit{label-only setting}, where the adversary can only access generated tokens (text). We first reveal that existing label-only MIAs have minor effects in attacking pre-trained LLMs, although they are highly effective in inferring fine-tuning datasets used for personalized LLMs. We find that their failure stems from two main reasons, including better generalization and overly coarse perturbation. Specifically, due to the extensive pre-training corpora and exposing each sample only a few times, LLMs exhibit minimal robustness differences between members and non-members. This makes token-level perturbations too coarse to capture such differences.   To alleviate these problems, we propose \textbf{PETAL}: a label-only membership inference attack based on \textbf{PE}r-\textbf{T}oken sem\textbf{A}ntic simi\textbf{L}arity. Specifically, PETAL leverages token-level semantic similarity to approximate output probabilities and subsequently calculate the perplexity. It finally exposes membership based on the common assumption that members are `better' memorized and have smaller perplexity. We conduct extensive experiments on the WikiMIA benchmark and the more challenging MIMIR benchmark. Empirically, our PETAL performs better than the extensions of existing label-only attacks against personalized LLMs and even on par with other advanced logit-based attacks across all metrics on five prevalent open-source LLMs.



## **49. JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models**

cs.CL

12 pages, 5 figures, accepted at PAKDD 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18935v1) [paper-pdf](http://arxiv.org/pdf/2502.18935v1)

**Authors**: Shuyi Liu, Simiao Cui, Haoran Bu, Yuming Shang, Xi Zhang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various applications, highlighting the urgent need for comprehensive safety evaluations. In particular, the enhanced Chinese language proficiency of LLMs, combined with the unique characteristics and complexity of Chinese expressions, has driven the emergence of Chinese-specific benchmarks for safety assessment. However, these benchmarks generally fall short in effectively exposing LLM safety vulnerabilities. To address the gap, we introduce JailBench, the first comprehensive Chinese benchmark for evaluating deep-seated vulnerabilities in LLMs, featuring a refined hierarchical safety taxonomy tailored to the Chinese context. To improve generation efficiency, we employ a novel Automatic Jailbreak Prompt Engineer (AJPE) framework for JailBench construction, which incorporates jailbreak techniques to enhance assessing effectiveness and leverages LLMs to automatically scale up the dataset through context-learning. The proposed JailBench is extensively evaluated over 13 mainstream LLMs and achieves the highest attack success rate against ChatGPT compared to existing Chinese benchmarks, underscoring its efficacy in identifying latent vulnerabilities in LLMs, as well as illustrating the substantial room for improvement in the security and trustworthiness of LLMs within the Chinese context. Our benchmark is publicly available at https://github.com/STAIR-BUPT/JailBench.



## **50. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

cs.CR

9 pages

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2411.00459v3) [paper-pdf](http://arxiv.org/pdf/2411.00459v3)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.



