# Latest Large Language Model Attack Papers
**update at 2025-09-05 19:09:50**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. An Automated, Scalable Machine Learning Model Inversion Assessment Pipeline**

cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04214v1) [paper-pdf](http://arxiv.org/pdf/2509.04214v1)

**Authors**: Tyler Shumaker, Jessica Carpenter, David Saranchak, Nathaniel D. Bastian

**Abstract**: Machine learning (ML) models have the potential to transform military battlefields, presenting a large external pressure to rapidly incorporate them into operational settings. However, it is well-established that these ML models are vulnerable to a number of adversarial attacks throughout the model deployment pipeline that threaten to negate battlefield advantage. One broad category is privacy attacks (such as model inversion) where an adversary can reverse engineer information from the model, such as the sensitive data used in its training. The ability to quantify the risk of model inversion attacks (MIAs) is not well studied, and there is a lack of automated developmental test and evaluation (DT&E) tools and metrics to quantify the effectiveness of privacy loss of the MIA. The current DT&E process is difficult because ML model inversions can be hard for a human to interpret, subjective when they are interpretable, and difficult to quantify in terms of inversion quality. Additionally, scaling the DT&E process is challenging due to many ML model architectures and data modalities that need to be assessed. In this work, we present a novel DT&E tool that quantifies the risk of data privacy loss from MIAs and introduces four adversarial risk dimensions to quantify privacy loss. Our DT&E pipeline combines inversion with vision language models (VLMs) to improve effectiveness while enabling scalable analysis. We demonstrate effectiveness using multiple MIA techniques and VLMs configured for zero-shot classification and image captioning. We benchmark the pipeline using several state-of-the-art MIAs in the computer vision domain with an image classification task that is typical in military applications. In general, our innovative pipeline extends the current model inversion DT&E capabilities by improving the effectiveness and scalability of the privacy loss analysis in an automated fashion.



## **2. KubeGuard: LLM-Assisted Kubernetes Hardening via Configuration Files and Runtime Logs Analysis**

cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04191v1) [paper-pdf](http://arxiv.org/pdf/2509.04191v1)

**Authors**: Omri Sgan Cohen, Ehud Malul, Yair Meidan, Dudu Mimran, Yuval Elovici, Asaf Shabtai

**Abstract**: The widespread adoption of Kubernetes (K8s) for orchestrating cloud-native applications has introduced significant security challenges, such as misconfigured resources and overly permissive configurations. Failing to address these issues can result in unauthorized access, privilege escalation, and lateral movement within clusters. Most existing K8s security solutions focus on detecting misconfigurations, typically through static analysis or anomaly detection. In contrast, this paper presents KubeGuard, a novel runtime log-driven recommender framework aimed at mitigating risks by addressing overly permissive configurations. KubeGuard is designed to harden K8s environments through two complementary tasks: Resource Creation and Resource Refinement. It leverages large language models (LLMs) to analyze manifests and runtime logs reflecting actual system behavior, using modular prompt-chaining workflows. This approach enables KubeGuard to create least-privilege configurations for new resources and refine existing manifests to reduce the attack surface. KubeGuard's output manifests are presented as recommendations that users (e.g., developers and operators) can review and adopt to enhance cluster security. Our evaluation demonstrates that KubeGuard effectively generates and refines K8s manifests for Roles, NetworkPolicies, and Deployments, leveraging both proprietary and open-source LLMs. The high precision, recall, and F1-scores affirm KubeGuard's practicality as a framework that translates runtime observability into actionable, least-privilege configuration guidance.



## **3. Privacy Risks in Time Series Forecasting: User- and Record-Level Membership Inference**

cs.LG

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04169v1) [paper-pdf](http://arxiv.org/pdf/2509.04169v1)

**Authors**: Nicolas Johansson, Tobias Olsson, Daniel Nilsson, Johan Östman, Fazeleh Hoseini

**Abstract**: Membership inference attacks (MIAs) aim to determine whether specific data were used to train a model. While extensively studied on classification models, their impact on time series forecasting remains largely unexplored. We address this gap by introducing two new attacks: (i) an adaptation of multivariate LiRA, a state-of-the-art MIA originally developed for classification models, to the time-series forecasting setting, and (ii) a novel end-to-end learning approach called Deep Time Series (DTS) attack. We benchmark these methods against adapted versions of other leading attacks from the classification setting.   We evaluate all attacks in realistic settings on the TUH-EEG and ELD datasets, targeting two strong forecasting architectures, LSTM and the state-of-the-art N-HiTS, under both record- and user-level threat models. Our results show that forecasting models are vulnerable, with user-level attacks often achieving perfect detection. The proposed methods achieve the strongest performance in several settings, establishing new baselines for privacy risk assessment in time series forecasting. Furthermore, vulnerability increases with longer prediction horizons and smaller training populations, echoing trends observed in large language models.



## **4. Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks**

cs.CL

EMNLP 2025 findings

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2508.20038v3) [paper-pdf](http://arxiv.org/pdf/2508.20038v3)

**Authors**: Sheng Liu, Qiang Sheng, Danding Wang, Yang Li, Guang Yang, Juan Cao

**Abstract**: Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.



## **5. NeuroBreak: Unveil Internal Jailbreak Mechanisms in Large Language Models**

cs.CR

12 pages, 9 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.03985v1) [paper-pdf](http://arxiv.org/pdf/2509.03985v1)

**Authors**: Chuhan Zhang, Ye Zhang, Bowen Shi, Yuyou Gan, Tianyu Du, Shouling Ji, Dazhan Deng, Yingcai Wu

**Abstract**: In deployment and application, large language models (LLMs) typically undergo safety alignment to prevent illegal and unethical outputs. However, the continuous advancement of jailbreak attack techniques, designed to bypass safety mechanisms with adversarial prompts, has placed increasing pressure on the security defenses of LLMs. Strengthening resistance to jailbreak attacks requires an in-depth understanding of the security mechanisms and vulnerabilities of LLMs. However, the vast number of parameters and complex structure of LLMs make analyzing security weaknesses from an internal perspective a challenging task. This paper presents NeuroBreak, a top-down jailbreak analysis system designed to analyze neuron-level safety mechanisms and mitigate vulnerabilities. We carefully design system requirements through collaboration with three experts in the field of AI security. The system provides a comprehensive analysis of various jailbreak attack methods. By incorporating layer-wise representation probing analysis, NeuroBreak offers a novel perspective on the model's decision-making process throughout its generation steps. Furthermore, the system supports the analysis of critical neurons from both semantic and functional perspectives, facilitating a deeper exploration of security mechanisms. We conduct quantitative evaluations and case studies to verify the effectiveness of our system, offering mechanistic insights for developing next-generation defense strategies against evolving jailbreak attacks.



## **6. Defending LVLMs Against Vision Attacks through Partial-Perception Supervision**

cs.CV

Accepted to ICML 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.12722v2) [paper-pdf](http://arxiv.org/pdf/2412.12722v2)

**Authors**: Qi Zhou, Tianlin Li, Qing Guo, Dongxia Wang, Yun Lin, Yang Liu, Jin Song Dong

**Abstract**: Recent studies have raised significant concerns regarding the vulnerability of Large Vision Language Models (LVLMs) to maliciously injected or perturbed input images, which can mislead their responses. Existing defense methods show that such vision attacks are sensitive to image modifications especially cropping, using majority voting across responses of modified images as corrected responses. However, these modifications often result in partial images and distort the semantics, which reduces response quality on clean images after voting. Instead of directly using responses from partial images for voting, we investigate using them to supervise the LVLM's responses to the original images. We propose a black-box, training-free method called DPS (Defense through Partial-Perception Supervision). In this approach, the model is prompted using the responses generated by a model that perceives only a partial image. With DPS, the model can adjust its response based on partial image understanding when under attack, while confidently maintaining its original response for clean input. Our findings show that the weak model can supervise the strong model: when faced with an attacked input, the strong model becomes less confident and adjusts its response based on the weak model's partial understanding, effectively defending against the attack. With clean input, it confidently maintains its original response. Empirical experiments show our method outperforms the baseline, cutting the average attack success rate by 76.3% across six datasets on three popular models.



## **7. VulRTex: A Reasoning-Guided Approach to Identify Vulnerabilities from Rich-Text Issue Report**

cs.SE

25 pages, 7 figures, submitting to TOSEM journal

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.03875v1) [paper-pdf](http://arxiv.org/pdf/2509.03875v1)

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Lin Shi, Qing Wang

**Abstract**: Software vulnerabilities exist in open-source software (OSS), and the developers who discover these vulnerabilities may submit issue reports (IRs) to describe their details. Security practitioners need to spend a lot of time manually identifying vulnerability-related IRs from the community, and the time gap may be exploited by attackers to harm the system. Previously, researchers have proposed automatic approaches to facilitate identifying these vulnerability-related IRs, but these works focus on textual descriptions but lack the comprehensive analysis of IR's rich-text information. In this paper, we propose VulRTex, a reasoning-guided approach to identify vulnerability-related IRs with their rich-text information. In particular, VulRTex first utilizes the reasoning ability of the Large Language Model (LLM) to prepare the Vulnerability Reasoning Database with historical IRs. Then, it retrieves the relevant cases from the prepared reasoning database to generate reasoning guidance, which guides LLM to identify vulnerabilities by reasoning analysis on target IRs' rich-text information. To evaluate the performance of VulRTex, we conduct experiments on 973,572 IRs, and the results show that VulRTex achieves the highest performance in identifying the vulnerability-related IRs and predicting CWE-IDs when the dataset is imbalanced, outperforming the best baseline with +11.0% F1, +20.2% AUPRC, and +10.5% Macro-F1, and 2x lower time cost than baseline reasoning approaches. Furthermore, VulRTex has been applied to identify 30 emerging vulnerabilities across 10 representative OSS projects in 2024's GitHub IRs, and 11 of them are successfully assigned CVE-IDs, which illustrates VulRTex's practicality.



## **8. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models**

cs.CL

Presented at EMNLP 2025

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2410.20940v2) [paper-pdf](http://arxiv.org/pdf/2410.20940v2)

**Authors**: Piotr Przybyła, Euan McGill, Horacio Saggion

**Abstract**: Large language models have many beneficial applications, but can they also be used to attack content-filtering algorithms in social media platforms? We investigate the challenge of generating adversarial examples to test the robustness of text classification algorithms detecting low-credibility content, including propaganda, false claims, rumours and hyperpartisan news. We focus on simulation of content moderation by setting realistic limits on the number of queries an attacker is allowed to attempt. Within our solution (TREPAT), initial rephrasings are generated by large language models with prompts inspired by meaning-preserving NLP tasks, such as text simplification and style transfer. Subsequently, these modifications are decomposed into small changes, applied through beam search procedure, until the victim classifier changes its decision. We perform (1) quantitative evaluation using various prompts, models and query limits, (2) targeted manual assessment of the generated text and (3) qualitative linguistic analysis. The results confirm the superiority of our approach in the constrained scenario, especially in case of long input text (news articles), where exhaustive search is not feasible.



## **9. BadPromptFL: A Novel Backdoor Threat to Prompt-based Federated Learning in Multimodal Models**

cs.LG

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2508.08040v2) [paper-pdf](http://arxiv.org/pdf/2508.08040v2)

**Authors**: Maozhen Zhang, Mengnan Zhao, Bo Wang

**Abstract**: Prompt-based tuning has emerged as a lightweight alternative to full fine-tuning in large vision-language models, enabling efficient adaptation via learned contextual prompts. This paradigm has recently been extended to federated learning settings (e.g., PromptFL), where clients collaboratively train prompts under data privacy constraints. However, the security implications of prompt-based aggregation in federated multimodal learning remain largely unexplored, leaving a critical attack surface unaddressed. In this paper, we introduce \textbf{BadPromptFL}, the first backdoor attack targeting prompt-based federated learning in multimodal contrastive models. In BadPromptFL, compromised clients jointly optimize local backdoor triggers and prompt embeddings, injecting poisoned prompts into the global aggregation process. These prompts are then propagated to benign clients, enabling universal backdoor activation at inference without modifying model parameters. Leveraging the contextual learning behavior of CLIP-style architectures, BadPromptFL achieves high attack success rates (e.g., \(>90\%\)) with minimal visibility and limited client participation. Extensive experiments across multiple datasets and aggregation protocols validate the effectiveness, stealth, and generalizability of our attack, raising critical concerns about the robustness of prompt-based federated learning in real-world deployments.



## **10. PromptCOS: Towards System Prompt Copyright Auditing for LLMs via Content-level Output Similarity**

cs.CR

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03117v1) [paper-pdf](http://arxiv.org/pdf/2509.03117v1)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Enhao Huang, Shuo Shao, Bingrun Yang, Zhibo Wang, Dacheng Tao, Zhan Qin

**Abstract**: The rapid progress of large language models (LLMs) has greatly enhanced reasoning tasks and facilitated the development of LLM-based applications. A critical factor in improving LLM-based applications is the design of effective system prompts, which significantly impact the behavior and output quality of LLMs. However, system prompts are susceptible to theft and misuse, which could undermine the interests of prompt owners. Existing methods protect prompt copyrights through watermark injection and verification but face challenges due to their reliance on intermediate LLM outputs (e.g., logits), which limits their practical feasibility.   In this paper, we propose PromptCOS, a method for auditing prompt copyright based on content-level output similarity. It embeds watermarks by optimizing the prompt while simultaneously co-optimizing a special verification query and content-level signal marks. This is achieved by leveraging cyclic output signals and injecting auxiliary tokens to ensure reliable auditing in content-only scenarios. Additionally, it incorporates cover tokens to protect the watermark from malicious deletion. For copyright verification, PromptCOS identifies unauthorized usage by comparing the similarity between the suspicious output and the signal mark. Experimental results demonstrate that our method achieves high effectiveness (99.3% average watermark similarity), strong distinctiveness (60.8% greater than the best baseline), high fidelity (accuracy degradation of no more than 0.58%), robustness (resilience against three types of potential attacks), and computational efficiency (up to 98.1% reduction in computational cost). Our code is available at GitHub https://github.com/LianPing-cyber/PromptCOS.



## **11. EverTracer: Hunting Stolen Large Language Models via Stealthy and Robust Probabilistic Fingerprint**

cs.CR

Accepted by EMNLP2025 Main

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03058v1) [paper-pdf](http://arxiv.org/pdf/2509.03058v1)

**Authors**: Zhenhua Xu, Meng Han, Wenpeng Xing

**Abstract**: The proliferation of large language models (LLMs) has intensified concerns over model theft and license violations, necessitating robust and stealthy ownership verification. Existing fingerprinting methods either require impractical white-box access or introduce detectable statistical anomalies. We propose EverTracer, a novel gray-box fingerprinting framework that ensures stealthy and robust model provenance tracing. EverTracer is the first to repurpose Membership Inference Attacks (MIAs) for defensive use, embedding ownership signals via memorization instead of artificial trigger-output overfitting. It consists of Fingerprint Injection, which fine-tunes the model on any natural language data without detectable artifacts, and Verification, which leverages calibrated probability variation signal to distinguish fingerprinted models. This approach remains robust against adaptive adversaries, including input level modification, and model-level modifications. Extensive experiments across architectures demonstrate EverTracer's state-of-the-art effectiveness, stealthness, and resilience, establishing it as a practical solution for securing LLM intellectual property. Our code and data are publicly available at https://github.com/Xuzhenhua55/EverTracer.



## **12. See No Evil: Adversarial Attacks Against Linguistic-Visual Association in Referring Multi-Object Tracking Systems**

cs.CV

12 pages, 1 figure, 3 tables

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.02028v2) [paper-pdf](http://arxiv.org/pdf/2509.02028v2)

**Authors**: Halima Bouzidi, Haoyu Liu, Mohammad Abdullah Al Faruque

**Abstract**: Language-vision understanding has driven the development of advanced perception systems, most notably the emerging paradigm of Referring Multi-Object Tracking (RMOT). By leveraging natural-language queries, RMOT systems can selectively track objects that satisfy a given semantic description, guided through Transformer-based spatial-temporal reasoning modules. End-to-End (E2E) RMOT models further unify feature extraction, temporal memory, and spatial reasoning within a Transformer backbone, enabling long-range spatial-temporal modeling over fused textual-visual representations. Despite these advances, the reliability and robustness of RMOT remain underexplored. In this paper, we examine the security implications of RMOT systems from a design-logic perspective, identifying adversarial vulnerabilities that compromise both the linguistic-visual referring and track-object matching components. Additionally, we uncover a novel vulnerability in advanced RMOT models employing FIFO-based memory, whereby targeted and consistent attacks on their spatial-temporal reasoning introduce errors that persist within the history buffer over multiple subsequent frames. We present VEIL, a novel adversarial framework designed to disrupt the unified referring-matching mechanisms of RMOT models. We show that carefully crafted digital and physical perturbations can corrupt the tracking logic reliability, inducing track ID switches and terminations. We conduct comprehensive evaluations using the Refer-KITTI dataset to validate the effectiveness of VEIL and demonstrate the urgent need for security-aware RMOT designs for critical large-scale applications.



## **13. A Survey: Towards Privacy and Security in Mobile Large Language Models**

cs.CR

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02411v1) [paper-pdf](http://arxiv.org/pdf/2509.02411v1)

**Authors**: Honghui Xu, Kaiyang Li, Wei Chen, Danyang Zheng, Zhiyuan Li, Zhipeng Cai

**Abstract**: Mobile Large Language Models (LLMs) are revolutionizing diverse fields such as healthcare, finance, and education with their ability to perform advanced natural language processing tasks on-the-go. However, the deployment of these models in mobile and edge environments introduces significant challenges related to privacy and security due to their resource-intensive nature and the sensitivity of the data they process. This survey provides a comprehensive overview of privacy and security issues associated with mobile LLMs, systematically categorizing existing solutions such as differential privacy, federated learning, and prompt encryption. Furthermore, we analyze vulnerabilities unique to mobile LLMs, including adversarial attacks, membership inference, and side-channel attacks, offering an in-depth comparison of their effectiveness and limitations. Despite recent advancements, mobile LLMs face unique hurdles in achieving robust security while maintaining efficiency in resource-constrained environments. To bridge this gap, we propose potential applications, discuss open challenges, and suggest future research directions, paving the way for the development of trustworthy, privacy-compliant, and scalable mobile LLM systems.



## **14. Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety**

cs.RO

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02163v1) [paper-pdf](http://arxiv.org/pdf/2509.02163v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Bräunl, Jin B. Hong

**Abstract**: Integrating large language models (LLMs) into robotic systems has revolutionised embodied artificial intelligence, enabling advanced decision-making and adaptability. However, ensuring reliability, encompassing both security against adversarial attacks and safety in complex environments, remains a critical challenge. To address this, we propose a unified framework that mitigates prompt injection attacks while enforcing operational safety through robust validation mechanisms. Our approach combines prompt assembling, state management, and safety validation, evaluated using both performance and security metrics. Experiments show a 30.8% improvement under injection attacks and up to a 325% improvement in complex environment settings under adversarial conditions compared to baseline scenarios. This work bridges the gap between safety and security in LLM-based robotic systems, offering actionable insights for deploying reliable LLM-integrated mobile robots in real-world settings. The framework is open-sourced with simulation and physical deployment demos at https://llmeyesim.vercel.app/



## **15. Unraveling LLM Jailbreaks Through Safety Knowledge Neurons**

cs.AI

10 pages, 6 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01631v1) [paper-pdf](http://arxiv.org/pdf/2509.01631v1)

**Authors**: Chongwen Zhao, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation, a technique known as "Jailbreak." While some studies have achieved defenses against jailbreak attacks by modifying output distributions or detecting harmful content, the exact rationale still remains elusive. In this work, we present a novel neuron-level interpretability method that focuses on the role of safety-related knowledge neurons. Unlike existing approaches, our method projects the model's internal representation into a more consistent and interpretable vocabulary space. We then show that adjusting the activation of safety-related neurons can effectively control the model's behavior with a mean ASR higher than 97%. Building on this insight, we propose SafeTuning, a fine-tuning strategy that reinforces safety-critical neurons to improve model robustness against jailbreaks. SafeTuning consistently reduces attack success rates across multiple LLMs and outperforms all four baseline defenses. These findings offer a new perspective on understanding and defending against jailbreak attacks.



## **16. LLMHoney: A Real-Time SSH Honeypot with Large Language Model-Driven Dynamic Response Generation**

cs.CR

7 Pages

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01463v1) [paper-pdf](http://arxiv.org/pdf/2509.01463v1)

**Authors**: Pranjay Malhotra

**Abstract**: Cybersecurity honeypots are deception tools for engaging attackers and gather intelligence, but traditional low or medium-interaction honeypots often rely on static, pre-scripted interactions that can be easily identified by skilled adversaries. This Report presents LLMHoney, an SSH honeypot that leverages Large Language Models (LLMs) to generate realistic, dynamic command outputs in real time. LLMHoney integrates a dictionary-based virtual file system to handle common commands with low latency while using LLMs for novel inputs, achieving a balance between authenticity and performance. We implemented LLMHoney using open-source LLMs and evaluated it on a testbed with 138 representative Linux commands. We report comprehensive metrics including accuracy (exact-match, Cosine Similarity, Jaro-Winkler Similarity, Levenshtein Similarity and BLEU score), response latency and memory overhead. We evaluate LLMHoney using multiple LLM backends ranging from 0.36B to 3.8B parameters, including both open-source models and a proprietary model(Gemini). Our experiments compare 13 different LLM variants; results show that Gemini-2.0 and moderately-sized models Qwen2.5:1.5B and Phi3:3.8B provide the most reliable and accurate responses, with mean latencies around 3 seconds, whereas smaller models often produce incorrect or out-of-character outputs. We also discuss how LLM integration improves honeypot realism and adaptability compared to traditional honeypots, as well as challenges such as occasional hallucinated outputs and increased resource usage. Our findings demonstrate that LLM-driven honeypots are a promising approach to enhance attacker engagement and collect richer threat intelligence.



## **17. MEGen: Generative Backdoor into Large Language Models via Model Editing**

cs.CL

ACL 2025 Findings

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2408.10722v2) [paper-pdf](http://arxiv.org/pdf/2408.10722v2)

**Authors**: Jiyang Qiu, Xinbei Ma, Zhuosheng Zhang, Hai Zhao, Yun Li, Qianren Wang

**Abstract**: Large language models (LLMs) have exhibited remarkable versatility and adaptability, while their widespread adoption across various applications also raises critical safety concerns. This paper focuses on the impact of backdoored LLMs. Traditional backdoor injection methods are primarily limited to yes-or-no discriminative tasks, leading users to underestimate the potential risks of backdoored LLMs. Given the inherently generative nature of LLMs, this paper reveals that a generative backdoor injected into LLMs can expose the true safety risks in their applications. We propose an editing-based generative backdoor, named MEGen, aiming to expand the backdoor to generative tasks in a unified format of any text-to any text, leading to natural generations with a specific intention. Experiments show that MEGen achieves a high attack success rate by adjusting only a small set of local parameters with few-shot samples. Notably, we show that the backdoored model, when triggered, can freely output pre-set dangerous information while completing downstream tasks. Our work highlights that MEGen enables backdoors in LLMs to exhibit generative capabilities, causing potential safety risks by altering the generative style. The code is available at https://github.com/MonoQ-hub/MEGen.



## **18. Strata-Sword: A Hierarchical Safety Evaluation towards LLMs based on Reasoning Complexity of Jailbreak Instructions**

cs.CY

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01444v1) [paper-pdf](http://arxiv.org/pdf/2509.01444v1)

**Authors**: Shiji Zhao, Ranjie Duan, Jiexi Liu, Xiaojun Jia, Fengxiang Wang, Cheng Wei, Ruoxi Cheng, Yong Xie, Chang Liu, Qing Guo, Jialing Tao, Hui Xue, Xingxing Wei

**Abstract**: Large language models (LLMs) have gained widespread recognition for their superior comprehension and have been deployed across numerous domains. Building on Chain-of-Thought (CoT) ideology, Large Reasoning models (LRMs) further exhibit strong reasoning skills, enabling them to infer user intent more accurately and respond appropriately. However, both LLMs and LRMs face the potential safety risks under jailbreak attacks, which raise concerns about their safety capabilities. Current safety evaluation methods often focus on the content dimensions, or simply aggregate different attack methods, lacking consideration of the complexity. In fact, instructions of different complexity can reflect the different safety capabilities of the model: simple instructions can reflect the basic values of the model, while complex instructions can reflect the model's ability to deal with deeper safety risks. Therefore, a comprehensive benchmark needs to be established to evaluate the safety performance of the model in the face of instructions of varying complexity, which can provide a better understanding of the safety boundaries of the LLMs. Thus, this paper first quantifies "Reasoning Complexity" as an evaluable safety dimension and categorizes 15 jailbreak attack methods into three different levels according to the reasoning complexity, establishing a hierarchical Chinese-English jailbreak safety benchmark for systematically evaluating the safety performance of LLMs. Meanwhile, to fully utilize unique language characteristics, we first propose some Chinese jailbreak attack methods, including the Chinese Character Disassembly attack, Lantern Riddle attack, and Acrostic Poem attack. A series of experiments indicate that current LLMs and LRMs show different safety boundaries under different reasoning complexity, which provides a new perspective to develop safer LLMs and LRMs.



## **19. An Automated Attack Investigation Approach Leveraging Threat-Knowledge-Augmented Large Language Models**

cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01271v1) [paper-pdf](http://arxiv.org/pdf/2509.01271v1)

**Authors**: Rujie Dai, Peizhuo Lv, Yujiang Gui, Qiujian Lv, Yuanyuan Qiao, Yan Wang, Degang Sun, Weiqing Huang, Yingjiu Li, XiaoFeng Wang

**Abstract**: Advanced Persistent Threats (APTs) are prolonged, stealthy intrusions by skilled adversaries that compromise high-value systems to steal data or disrupt operations. Reconstructing complete attack chains from massive, heterogeneous logs is essential for effective attack investigation, yet existing methods suffer from poor platform generality, limited generalization to evolving tactics, and an inability to produce analyst-ready reports. Large Language Models (LLMs) offer strong semantic understanding and summarization capabilities, but in this domain they struggle to capture the long-range, cross-log dependencies critical for accurate reconstruction.   To solve these problems, we present an LLM-empowered attack investigation framework augmented with a dynamically adaptable Kill-Chain-aligned threat knowledge base. We organizes attack-relevant behaviors into stage-aware knowledge units enriched with semantic annotations, enabling the LLM to iteratively retrieve relevant intelligence, perform causal reasoning, and progressively expand the investigation context. This process reconstructs multi-phase attack scenarios and generates coherent, human-readable investigation reports. Evaluated on 15 attack scenarios spanning single-host and multi-host environments across Windows and Linux (over 4.3M log events, 7.2 GB of data), the system achieves an average True Positive Rate (TPR) of 97.1% and an average False Positive Rate (FPR) of 0.2%, significantly outperforming the SOTA method ATLAS, which achieves an average TPR of 79.2% and an average FPR of 29.1%.



## **20. One Shot Dominance: Knowledge Poisoning Attack on Retrieval-Augmented Generation Systems**

cs.CR

15pages, 4 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2505.11548v3) [paper-pdf](http://arxiv.org/pdf/2505.11548v3)

**Authors**: Zhiyuan Chang, Mingyang Li, Xiaojun Jia, Junjie Wang, Yuekai Huang, Ziyou Jiang, Yang Liu, Qing Wang

**Abstract**: Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG) have shown improved performance in generating accurate responses. However, the dependence on external knowledge bases introduces potential security vulnerabilities, particularly when these knowledge bases are publicly accessible and modifiable. While previous studies have exposed knowledge poisoning risks in RAG systems, existing attack methods suffer from critical limitations: they either require injecting multiple poisoned documents (resulting in poor stealthiness) or can only function effectively on simplistic queries (limiting real-world applicability). This paper reveals a more realistic knowledge poisoning attack against RAG systems that achieves successful attacks by poisoning only a single document while remaining effective for complex multi-hop questions involving complex relationships between multiple elements. Our proposed AuthChain address three challenges to ensure the poisoned documents are reliably retrieved and trusted by the LLM, even against large knowledge bases and LLM's own knowledge. Extensive experiments across six popular LLMs demonstrate that AuthChain achieves significantly higher attack success rates while maintaining superior stealthiness against RAG defense mechanisms compared to state-of-the-art baselines.



## **21. Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation**

cs.CR

8 pages. Accepted for publication in the proceedings of 7th IEEE  International Conference on Trust, Privacy and Security in Intelligent  Systems, and Applications (IEEE TPS 2025)

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00973v1) [paper-pdf](http://arxiv.org/pdf/2509.00973v1)

**Authors**: Kanchon Gharami, Hansaka Aluvihare, Shafika Showkat Moni, Berker Peköz

**Abstract**: Large Language Models (LLMs) are increasingly deployed in mission-critical systems, facilitating tasks such as satellite operations, command-and-control, military decision support, and cyber defense. Many of these systems are accessed through application programming interfaces (APIs). When such APIs lack robust access controls, they can expose full or top-k logits, creating a significant and often overlooked attack surface. Prior art has mainly focused on reconstructing the output projection layer or distilling surface-level behaviors. However, regenerating a black-box model under tight query constraints remains underexplored. We address that gap by introducing a constrained replication pipeline that transforms partial logit leakage into a functional deployable substitute model clone. Our two-stage approach (i) reconstructs the output projection matrix by collecting top-k logits from under 10k black-box queries via singular value decomposition (SVD) over the logits, then (ii) distills the remaining architecture into compact student models with varying transformer depths, trained on an open source dataset. A 6-layer student recreates 97.6% of the 6-layer teacher model's hidden-state geometry, with only a 7.31% perplexity increase, and a 7.58 Negative Log-Likelihood (NLL). A 4-layer variant achieves 17.1% faster inference and 18.1% parameter reduction with comparable performance. The entire attack completes in under 24 graphics processing unit (GPU) hours and avoids triggering API rate-limit defenses. These results demonstrate how quickly a cost-limited adversary can clone an LLM, underscoring the urgent need for hardened inference APIs and secure on-premise defense deployments.



## **22. Membership Inference Attacks on Large-Scale Models: A Survey**

cs.LG

Preprint. Submitted for peer review. The final version may differ

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2503.19338v3) [paper-pdf](http://arxiv.org/pdf/2503.19338v3)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: As large-scale models such as Large Language Models (LLMs) and Large Multimodal Models (LMMs) see increasing deployment, their privacy risks remain underexplored. Membership Inference Attacks (MIAs), which reveal whether a data point was used in training the target model, are an important technique for exposing or assessing privacy risks and have been shown to be effective across diverse machine learning algorithms. However, despite extensive studies on MIAs in classic models, there remains a lack of systematic surveys addressing their effectiveness and limitations in large-scale models. To address this gap, we provide the first comprehensive review of MIAs targeting LLMs and LMMs, analyzing attacks by model type, adversarial knowledge, and strategy. Unlike prior surveys, we further examine MIAs across multiple stages of the model pipeline, including pre-training, fine-tuning, alignment, and Retrieval-Augmented Generation (RAG). Finally, we identify open challenges and propose future research directions for strengthening privacy resilience in large-scale models.



## **23. A Review of Hydrogen-Enabled Resilience Enhancement for Multi-Energy Systems**

eess.SY

28 pages, 14 figures

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2412.19374v2) [paper-pdf](http://arxiv.org/pdf/2412.19374v2)

**Authors**: Liang Yu, Haoyu Fang, Goran Strbac, Dawei Qiu, Dong Yue, Xiaohong Guan, Gerhard P. Hancke

**Abstract**: Ensuring resilience in multi-energy systems (MESs) becomes both more urgent and more challenging due to the rising occurrence and severity of extreme events (e.g., natural disasters, extreme weather, and cyber-physical attacks). Among many measures of strengthening MES resilience, the integration of hydrogen shows exceptional potential in cross-temporal flexibility, cross-spatial flexibility, cross-sector flexibility, and black start capability. Although many hydrogen-enabled MES resilience enhancement measures have been developed, the current literature lacks a systematic overview of hydrogen-enabled resilience enhancement in MESs. To fill the research gap, this paper provides a comprehensive overview of hydrogen-enabled MES resilience enhancement. First, advantages and challenges of adopting hydrogen in MES resilience enhancement are summarized. Then, we propose a resilience enhancement framework for hydrogen-enabled MESs. Under the proposed framework, existing resilience metrics and event-oriented contingency models are summarized and discussed. Furthermore, we classify hydrogen-enabled planning measures by the types of hydrogen-related facilities and provide some insights for planning problem formulation frameworks. Moreover, we categorize the hydrogen-enabled operation enhancement measures into three operation response stages: preventive, emergency, and restoration. Finally, we identify some research gaps and point out possible future directions in aspects of comprehensive resilience metric design, temporally-correlated event-targeted scenario generation, multi-type temporal-spatial cyber-physical contingency modeling under compound extreme events, multi-network multi-timescale coordinated planning and operation, low-carbon resilient planning and operation, and large language model-assisted whole-process resilience enhancement.



## **24. Truthful Text Sanitization Guided by Inference Attacks**

cs.CL

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2412.12928v2) [paper-pdf](http://arxiv.org/pdf/2412.12928v2)

**Authors**: Ildikó Pilán, Benet Manzanares-Salor, David Sánchez, Pierre Lison

**Abstract**: Text sanitization aims to rewrite parts of a document to prevent disclosure of personal information. The central challenge of text sanitization is to strike a balance between privacy protection (avoiding the leakage of personal information) and utility preservation (retaining as much as possible of the document's original content). To this end, we introduce a novel text sanitization method based on generalizations, that is, broader but still informative terms that subsume the semantic content of the original text spans. The approach relies on the use of instruction-tuned large language models (LLMs) and is divided into two stages. Given a document including text spans expressing personally identifiable information (PII), the LLM is first applied to obtain truth-preserving replacement candidates for each text span and rank those according to their abstraction level. Those candidates are then evaluated for their ability to protect privacy by conducting inference attacks with the LLM. Finally, the system selects the most informative replacement candidate shown to be resistant to those attacks. This two-stage process produces replacements that effectively balance privacy and utility.   We also present novel metrics to evaluate these two aspects without needing to manually annotate documents. Results on the Text Anonymization Benchmark show that the proposed approach, implemented with Mistral 7B Instruct, leads to enhanced utility, with only a marginal (< 1 p.p.) increase in re-identification risk compared to fully suppressing the original spans. Furthermore, our approach is shown to be more truth-preserving than existing methods such as Microsoft Presidio's synthetic replacements.



## **25. PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization**

cs.CR

Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for  Toxicity Maximization

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2412.05892v4) [paper-pdf](http://arxiv.org/pdf/2412.05892v4)

**Authors**: Ruoxi Cheng, Yizhong Ding, Shuirong Cao, Ranjie Duan, Xiaoshuang Jia, Shaowei Yuan, Simeng Qin, Zhiqiang Wang, Xiaojun Jia

**Abstract**: Understanding the vulnerabilities of Large Vision Language Models (LVLMs) to jailbreak attacks is essential for their responsible real-world deployment. Most previous work requires access to model gradients, or is based on human knowledge (prompt engineering) to complete jailbreak, and they hardly consider the interaction of images and text, resulting in inability to jailbreak in black box scenarios or poor performance. To overcome these limitations, we propose a Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for toxicity maximization, referred to as PBI-Attack. Our method begins by extracting malicious features from a harmful corpus using an alternative LVLM and embedding these features into a benign image as prior information. Subsequently, we enhance these features through bidirectional cross-modal interaction optimization, which iteratively optimizes the bimodal perturbations in an alternating manner through greedy search, aiming to maximize the toxicity of the generated response. The toxicity level is quantified using a well-trained evaluation model. Experiments demonstrate that PBI-Attack outperforms previous state-of-the-art jailbreak methods, achieving an average attack success rate of 92.5% across three open-source LVLMs and around 67.3% on three closed-source LVLMs. Disclaimer: This paper contains potentially disturbing and offensive content.



## **26. The Resurgence of GCG Adversarial Attacks on Large Language Models**

cs.CL

12 pages, 5 figures

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00391v1) [paper-pdf](http://arxiv.org/pdf/2509.00391v1)

**Authors**: Yuting Tan, Xuying Li, Zhuo Li, Huizhen Shu, Peikang Hu

**Abstract**: Gradient-based adversarial prompting, such as the Greedy Coordinate Gradient (GCG) algorithm, has emerged as a powerful method for jailbreaking large language models (LLMs). In this paper, we present a systematic appraisal of GCG and its annealing-augmented variant, T-GCG, across open-source LLMs of varying scales. Using Qwen2.5-0.5B, LLaMA-3.2-1B, and GPT-OSS-20B, we evaluate attack effectiveness on both safety-oriented prompts (AdvBench) and reasoning-intensive coding prompts. Our study reveals three key findings: (1) attack success rates (ASR) decrease with model size, reflecting the increasing complexity and non-convexity of larger models' loss landscapes; (2) prefix-based heuristics substantially overestimate attack effectiveness compared to GPT-4o semantic judgments, which provide a stricter and more realistic evaluation; and (3) coding-related prompts are significantly more vulnerable than adversarial safety prompts, suggesting that reasoning itself can be exploited as an attack vector. In addition, preliminary results with T-GCG show that simulated annealing can diversify adversarial search and achieve competitive ASR under prefix evaluation, though its benefits under semantic judgment remain limited. Together, these findings highlight the scalability limits of GCG, expose overlooked vulnerabilities in reasoning tasks, and motivate further development of annealing-inspired strategies for more robust adversarial evaluation.



## **27. Progent: Programmable Privilege Control for LLM Agents**

cs.CR

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2504.11703v2) [paper-pdf](http://arxiv.org/pdf/2504.11703v2)

**Authors**: Tianneng Shi, Jingxuan He, Zhun Wang, Hongwei Li, Linyu Wu, Wenbo Guo, Dawn Song

**Abstract**: LLM agents utilize Large Language Models as central components with diverse tools to complete various user tasks, but face significant security risks when interacting with external environments. Attackers can exploit these agents through various vectors, including indirect prompt injection, memory/knowledge base poisoning, and malicious tools, tricking agents into performing dangerous actions such as unauthorized financial transactions or data leakage. The core problem that enables attacks to succeed lies in over-privileged tool access. We introduce Progent, the first privilege control framework to secure LLM agents. Progent enforces security at the tool level by restricting agents to performing tool calls necessary for user tasks while blocking potentially malicious ones. Progent features a domain-specific language that allows for expressing fine-grained policies for controlling tool privileges, flexible fallback actions when calls are blocked, and dynamic policy updates to adapt to changing agent states. The framework operates deterministically at runtime, providing provable security guarantees. Thanks to our modular design, integrating Progent does not alter agent internals and only requires minimal changes to the existing agent implementation, enhancing its practicality and potential for widespread adoption. Our extensive evaluation across various agent use cases, using benchmarks like AgentDojo, ASB, and AgentPoison, demonstrates that Progent reduces attack success rates to 0%, while preserving agent utility and speed. Additionally, we show that LLMs can automatically generate effective policies, highlighting their potential for automating the process of writing Progent's security policies.



## **28. QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety**

cs.CR

Accept to ACLW 2025 (WOAH)

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2506.12299v2) [paper-pdf](http://arxiv.org/pdf/2506.12299v2)

**Authors**: Taegyeong Lee, Jeonghwa Yoo, Hyoungseo Cho, Soo Yong Kim, Yunho Maeng

**Abstract**: The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts.



## **29. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

cs.CV

we update the paper title

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2506.05982v5) [paper-pdf](http://arxiv.org/pdf/2506.05982v5)

**Authors**: Zonglin Wu, Yule Xue, Yaoyao Feng, Xiaolong Wang, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.



## **30. WebInject: Prompt Injection Attack to Web Agents**

cs.LG

EMNLP 2025 main

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2505.11717v3) [paper-pdf](http://arxiv.org/pdf/2505.11717v3)

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.



## **31. Detecting Stealthy Data Poisoning Attacks in AI Code Generators**

cs.CR

Accepted to the 3rd IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE, 2025), co-located with ISSRE  2025

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21636v1) [paper-pdf](http://arxiv.org/pdf/2508.21636v1)

**Authors**: Cristina Improta

**Abstract**: Deep learning (DL) models for natural language-to-code generation have become integral to modern software development pipelines. However, their heavy reliance on large amounts of data, often collected from unsanitized online sources, exposes them to data poisoning attacks, where adversaries inject malicious samples to subtly bias model behavior. Recent targeted attacks silently replace secure code with semantically equivalent but vulnerable implementations without relying on explicit triggers to launch the attack, making it especially hard for detection methods to distinguish clean from poisoned samples. We present a systematic study on the effectiveness of existing poisoning detection methods under this stealthy threat model. Specifically, we perform targeted poisoning on three DL models (CodeBERT, CodeT5+, AST-T5), and evaluate spectral signatures analysis, activation clustering, and static analysis as defenses. Our results show that all methods struggle to detect triggerless poisoning, with representation-based approaches failing to isolate poisoned samples and static analysis suffering false positives and false negatives, highlighting the need for more robust, trigger-independent defenses for AI-assisted code generation.



## **32. SoK: Large Language Model-Generated Textual Phishing Campaigns End-to-End Analysis of Generation, Characteristics, and Detection**

cs.CR

13 pages, 3 tables, 4 figures

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21457v1) [paper-pdf](http://arxiv.org/pdf/2508.21457v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing is a pervasive form of social engineering in which attackers impersonate trusted entities to steal information or induce harmful actions. Text-based phishing dominates for its low cost, scalability, and concealability, advantages recently amplified by large language models (LLMs) that enable ``Phishing-as-a-Service'' attacks at scale within minutes. Despite the growing research into LLM-facilitated phishing attacks, consolidated systematic research on the phishing attack life cycle remains scarce. In this work, we present the first systematization of knowledge (SoK) on LLM-generated phishing, offering an end-to-end analysis that spans generation techniques, attack features, and mitigation strategies. We introduce Generation-Characterization-Defense (GenCharDef), which systematizes the ways in which LLM-generated phishing differs from traditional phishing across methodologies, security perspectives, data dependencies, and evaluation practices. This framework highlights unique challenges of LLM-driven phishing, providing a coherent foundation for understanding the evolving threat landscape and guiding the design of more resilient defenses.



## **33. Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review**

cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.20863v2) [paper-pdf](http://arxiv.org/pdf/2508.20863v2)

**Authors**: Matteo Gioele Collu, Umberto Salviati, Roberto Confalonieri, Mauro Conti, Giovanni Apruzzese

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.



## **34. A Whole New World: Creating a Parallel-Poisoned Web Only AI-Agents Can See**

cs.CR

10 pages, 1 figure

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2509.00124v1) [paper-pdf](http://arxiv.org/pdf/2509.00124v1)

**Authors**: Shaked Zychlinski

**Abstract**: This paper introduces a novel attack vector that leverages website cloaking techniques to compromise autonomous web-browsing agents powered by Large Language Models (LLMs). As these agents become more prevalent, their unique and often homogenous digital fingerprints - comprising browser attributes, automation framework signatures, and network characteristics - create a new, distinguishable class of web traffic. The attack exploits this fingerprintability. A malicious website can identify an incoming request as originating from an AI agent and dynamically serve a different, "cloaked" version of its content. While human users see a benign webpage, the agent is presented with a visually identical page embedded with hidden, malicious instructions, such as indirect prompt injections. This mechanism allows adversaries to hijack agent behavior, leading to data exfiltration, malware execution, or misinformation propagation, all while remaining completely invisible to human users and conventional security crawlers. This work formalizes the threat model, details the mechanics of agent fingerprinting and cloaking, and discusses the profound security implications for the future of agentic AI, highlighting the urgent need for robust defenses against this stealthy and scalable attack.



## **35. Lethe: Purifying Backdoored Large Language Models with Knowledge Dilution**

cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21004v1) [paper-pdf](http://arxiv.org/pdf/2508.21004v1)

**Authors**: Chen Chen, Yuchen Sun, Jiaxin Gao, Xueluan Gong, Qian Wang, Ziyao Wang, Yongsen Zheng, Kwok-Yan Lam

**Abstract**: Large language models (LLMs) have seen significant advancements, achieving superior performance in various Natural Language Processing (NLP) tasks. However, they remain vulnerable to backdoor attacks, where models behave normally for standard queries but generate harmful responses or unintended output when specific triggers are activated. Existing backdoor defenses either lack comprehensiveness, focusing on narrow trigger settings, detection-only mechanisms, and limited domains, or fail to withstand advanced scenarios like model-editing-based, multi-trigger, and triggerless attacks. In this paper, we present LETHE, a novel method to eliminate backdoor behaviors from LLMs through knowledge dilution using both internal and external mechanisms. Internally, LETHE leverages a lightweight dataset to train a clean model, which is then merged with the backdoored model to neutralize malicious behaviors by diluting the backdoor impact within the model's parametric memory. Externally, LETHE incorporates benign and semantically relevant evidence into the prompt to distract LLM's attention from backdoor features. Experimental results on classification and generation domains across 5 widely used LLMs demonstrate that LETHE outperforms 8 state-of-the-art defense baselines against 8 backdoor attacks. LETHE reduces the attack success rate of advanced backdoor attacks by up to 98% while maintaining model utility. Furthermore, LETHE has proven to be cost-efficient and robust against adaptive backdoor attacks.



## **36. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20890v1) [paper-pdf](http://arxiv.org/pdf/2508.20890v1)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.



## **37. Multi-Agent Penetration Testing AI for the Web**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20816v1) [paper-pdf](http://arxiv.org/pdf/2508.20816v1)

**Authors**: Isaac David, Arthur Gervais

**Abstract**: AI-powered development platforms are making software creation accessible to a broader audience, but this democratization has triggered a scalability crisis in security auditing. With studies showing that up to 40% of AI-generated code contains vulnerabilities, the pace of development now vastly outstrips the capacity for thorough security assessment.   We present MAPTA, a multi-agent system for autonomous web application security assessment that combines large language model orchestration with tool-grounded execution and end-to-end exploit validation. On the 104-challenge XBOW benchmark, MAPTA achieves 76.9% overall success with perfect performance on SSRF and misconfiguration vulnerabilities, 83% success on broken authorization, and strong results on injection attacks including server-side template injection (85%) and SQL injection (83%). Cross-site scripting (57%) and blind SQL injection (0%) remain challenging. Our comprehensive cost analysis across all challenges totals $21.38 with a median cost of $0.073 for successful attempts versus $0.357 for failures. Success correlates strongly with resource efficiency, enabling practical early-stopping thresholds at approximately 40 tool calls or $0.30 per challenge.   MAPTA's real-world findings are impactful given both the popularity of the respective scanned GitHub repositories (8K-70K stars) and MAPTA's low average operating cost of $3.67 per open-source assessment: MAPTA discovered critical vulnerabilities including RCEs, command injections, secret exposure, and arbitrary file write vulnerabilities. Findings are responsibly disclosed, 10 findings are under CVE review.



## **38. Addressing Tokenization Inconsistency in Steganography and Watermarking Based on Large Language Models**

cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20718v1) [paper-pdf](http://arxiv.org/pdf/2508.20718v1)

**Authors**: Ruiyi Yan, Yugo Murawaki

**Abstract**: Large language models have significantly enhanced the capacities and efficiency of text generation. On the one hand, they have improved the quality of text-based steganography. On the other hand, they have also underscored the importance of watermarking as a safeguard against malicious misuse. In this study, we focus on tokenization inconsistency (TI) between Alice and Bob in steganography and watermarking, where TI can undermine robustness. Our investigation reveals that the problematic tokens responsible for TI exhibit two key characteristics: infrequency and temporariness. Based on these findings, we propose two tailored solutions for TI elimination: a stepwise verification method for steganography and a post-hoc rollback method for watermarking. Experiments show that (1) compared to traditional disambiguation methods in steganography, directly addressing TI leads to improvements in fluency, imperceptibility, and anti-steganalysis capacity; (2) for watermarking, addressing TI enhances detectability and robustness against attacks.



## **39. Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning**

cs.LG

Project Hompage: https://tokenbuncher.github.io/

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20697v1) [paper-pdf](http://arxiv.org/pdf/2508.20697v1)

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.



## **40. CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics**

cs.CR

Code:  https://github.com/SmartData-Polito/LLM_Agent_Cybersecurity_Forensic

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20643v1) [paper-pdf](http://arxiv.org/pdf/2508.20643v1)

**Authors**: Stefano Fumero, Kai Huang, Matteo Boffa, Danilo Giordano, Marco Mellia, Zied Ben Houidi, Dario Rossi

**Abstract**: Large Language Model (LLM) agents are powerful tools for automating complex tasks. In cybersecurity, researchers have primarily explored their use in red-team operations such as vulnerability discovery and penetration tests. Defensive uses for incident response and forensics have received comparatively less attention and remain at an early stage. This work presents a systematic study of LLM-agent design for the forensic investigation of realistic web application attacks. We propose CyberSleuth, an autonomous agent that processes packet-level traces and application logs to identify the targeted service, the exploited vulnerability (CVE), and attack success. We evaluate the consequences of core design decisions - spanning tool integration and agent architecture - and provide interpretable guidance for practitioners. We benchmark four agent architectures and six LLM backends on 20 incident scenarios of increasing complexity, identifying CyberSleuth as the best-performing design. In a separate set of 10 incidents from 2025, CyberSleuth correctly identifies the exact CVE in 80% of cases. At last, we conduct a human study with 22 experts, which rated the reports of CyberSleuth as complete, useful, and coherent. They also expressed a slight preference for DeepSeek R1, a good news for open source LLM. To foster progress in defensive LLM research, we release both our benchmark and the CyberSleuth platform as a foundation for fair, reproducible evaluation of forensic agents.



## **41. NetGPT: Generative Pretrained Transformer for Network Traffic**

cs.NI

Code is available at https://github.com/ict-net/NetGPT

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2304.09513v3) [paper-pdf](http://arxiv.org/pdf/2304.09513v3)

**Authors**: Xuying Meng, Chungang Lin, Yequan Wang, Yujun Zhang

**Abstract**: All data on the Internet are transferred by network traffic, thus accurately modeling network traffic can help improve network services quality and protect data privacy. Pretrained models for network traffic can utilize large-scale raw data to learn the essential characteristics of network traffic, and generate distinguishable results for input traffic without considering specific downstream tasks. Effective pretrained models can significantly optimize the training efficiency and effectiveness of downstream tasks, such as application classification, attack detection and traffic generation. Despite the great success of pretraining in natural language processing, there is no work in the network field. Considering the diverse demands and characteristics of network traffic and network tasks, it is non-trivial to build a pretrained model for network traffic and we face various challenges, especially the heterogeneous headers and payloads in the multi-pattern network traffic and the different dependencies for contexts of diverse downstream network tasks.   To tackle these challenges, in this paper, we make the first attempt to provide a generative pretrained model NetGPT for both traffic understanding and generation tasks. We propose the multi-pattern network traffic modeling to construct unified text inputs and support both traffic understanding and generation tasks. We further optimize the adaptation effect of the pretrained model to diversified tasks by shuffling header fields, segmenting packets in flows, and incorporating diverse task labels with prompts. With diverse traffic datasets from encrypted software, DNS, private industrial protocols and cryptocurrency mining, expensive experiments demonstrate the effectiveness of our NetGPT in a range of traffic understanding and generation tasks on traffic datasets, and outperform state-of-the-art baselines by a wide margin.



## **42. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2503.06989v4) [paper-pdf](http://arxiv.org/pdf/2503.06989v4)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.



## **43. Ransomware 3.0: Self-Composing and LLM-Orchestrated**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20444v1) [paper-pdf](http://arxiv.org/pdf/2508.20444v1)

**Authors**: Md Raz, Meet Udeshi, P. V. Sai Charan, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri

**Abstract**: Using automated reasoning, code synthesis, and contextual decision-making, we introduce a new threat that exploits large language models (LLMs) to autonomously plan, adapt, and execute the ransomware attack lifecycle. Ransomware 3.0 represents the first threat model and research prototype of LLM-orchestrated ransomware. Unlike conventional malware, the prototype only requires natural language prompts embedded in the binary; malicious code is synthesized dynamically by the LLM at runtime, yielding polymorphic variants that adapt to the execution environment. The system performs reconnaissance, payload generation, and personalized extortion, in a closed-loop attack campaign without human involvement. We evaluate this threat across personal, enterprise, and embedded environments using a phase-centric methodology that measures quantitative fidelity and qualitative coherence in each attack phase. We show that open source LLMs can generate functional ransomware components and sustain closed-loop execution across diverse environments. Finally, we present behavioral signals and multi-level telemetry of Ransomware 3.0 through a case study to motivate future development of better defenses and policy enforcements to address novel AI-enabled ransomware attacks.



## **44. Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs**

cs.LG

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20333v1) [paper-pdf](http://arxiv.org/pdf/2508.20333v1)

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.



## **45. CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors**

cs.CL

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.02997v3) [paper-pdf](http://arxiv.org/pdf/2508.02997v3)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models.



## **46. Disabling Self-Correction in Retrieval-Augmented Generation via Stealthy Retriever Poisoning**

cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20083v1) [paper-pdf](http://arxiv.org/pdf/2508.20083v1)

**Authors**: Yanbo Dai, Zhenlan Ji, Zongjie Li, Kuan Li, Shuai Wang

**Abstract**: Retrieval-Augmented Generation (RAG) has become a standard approach for improving the reliability of large language models (LLMs). Prior work demonstrates the vulnerability of RAG systems by misleading them into generating attacker-chosen outputs through poisoning the knowledge base. However, this paper uncovers that such attacks could be mitigated by the strong \textit{self-correction ability (SCA)} of modern LLMs, which can reject false context once properly configured. This SCA poses a significant challenge for attackers aiming to manipulate RAG systems.   In contrast to previous poisoning methods, which primarily target the knowledge base, we introduce \textsc{DisarmRAG}, a new poisoning paradigm that compromises the retriever itself to suppress the SCA and enforce attacker-chosen outputs. This compromisation enables the attacker to straightforwardly embed anti-SCA instructions into the context provided to the generator, thereby bypassing the SCA. To this end, we present a contrastive-learning-based model editing technique that performs localized and stealthy edits, ensuring the retriever returns a malicious instruction only for specific victim queries while preserving benign retrieval behavior. To further strengthen the attack, we design an iterative co-optimization framework that automatically discovers robust instructions capable of bypassing prompt-based defenses. We extensively evaluate DisarmRAG across six LLMs and three QA benchmarks. Our results show near-perfect retrieval of malicious instructions, which successfully suppress SCA and achieve attack success rates exceeding 90\% under diverse defensive prompts. Also, the edited retriever remains stealthy under several detection methods, highlighting the urgent need for retriever-centric defenses.



## **47. Scaling Decentralized Learning with FLock**

cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.15349v2) [paper-pdf](http://arxiv.org/pdf/2507.15349v2)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.



## **48. IntentionReasoner: Facilitating Adaptive LLM Safeguards through Intent Reasoning and Selective Query Refinement**

cs.AI

17 pages, 9 figures

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20151v1) [paper-pdf](http://arxiv.org/pdf/2508.20151v1)

**Authors**: Yuanzhe Shen, Zisu Huang, Zhengkang Guo, Yide Liu, Guanxu Chen, Ruicheng Yin, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The rapid advancement of large language models (LLMs) has driven their adoption across diverse domains, yet their ability to generate harmful content poses significant safety challenges. While extensive research has focused on mitigating harmful outputs, such efforts often come at the cost of excessively rejecting harmless prompts. Striking a balance among safety, over-refusal, and utility remains a critical challenge. In this work, we introduce IntentionReasoner, a novel safeguard mechanism that leverages a dedicated guard model to perform intent reasoning, multi-level safety classification, and query rewriting to neutralize potentially harmful intent in edge-case queries. Specifically, we first construct a comprehensive dataset comprising approximately 163,000 queries, each annotated with intent reasoning, safety labels, and rewritten versions. Supervised fine-tuning is then applied to equip the guard model with foundational capabilities in format adherence, intent analysis, and safe rewriting. Finally, we apply a tailored multi-reward optimization strategy that integrates rule-based heuristics and reward model signals within a reinforcement learning framework to further enhance performance. Extensive experiments show that IntentionReasoner excels in multiple safeguard benchmarks, generation quality evaluations, and jailbreak attack scenarios, significantly enhancing safety while effectively reducing over-refusal rates and improving the quality of responses.



## **49. Secure Multi-LLM Agentic AI and Agentification for Edge General Intelligence by Zero-Trust: A Survey**

cs.NI

35 pages

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19870v1) [paper-pdf](http://arxiv.org/pdf/2508.19870v1)

**Authors**: Yinqiu Liu, Ruichen Zhang, Haoxiang Luo, Yijing Lin, Geng Sun, Dusit Niyato, Hongyang Du, Zehui Xiong, Yonggang Wen, Abbas Jamalipour, Dong In Kim, Ping Zhang

**Abstract**: Agentification serves as a critical enabler of Edge General Intelligence (EGI), transforming massive edge devices into cognitive agents through integrating Large Language Models (LLMs) and perception, reasoning, and acting modules. These agents collaborate across heterogeneous edge infrastructures, forming multi-LLM agentic AI systems that leverage collective intelligence and specialized capabilities to tackle complex, multi-step tasks. However, the collaborative nature of multi-LLM systems introduces critical security vulnerabilities, including insecure inter-LLM communications, expanded attack surfaces, and cross-domain data leakage that traditional perimeter-based security cannot adequately address. To this end, this survey introduces zero-trust security of multi-LLM in EGI, a paradigmatic shift following the ``never trust, always verify'' principle. We begin by systematically analyzing the security risks in multi-LLM systems within EGI contexts. Subsequently, we present the vision of a zero-trust multi-LLM framework in EGI. We then survey key technical progress to facilitate zero-trust multi-LLM systems in EGI. Particularly, we categorize zero-trust security mechanisms into model- and system-level approaches. The former and latter include strong identification, context-aware access control, etc., and proactive maintenance, blockchain-based management, etc., respectively. Finally, we identify critical research directions. This survey serves as the first systematic treatment of zero-trust applied to multi-LLM systems, providing both theoretical foundations and practical strategies.



## **50. AEGIS : Automated Co-Evolutionary Framework for Guarding Prompt Injections Schema**

cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2509.00088v1) [paper-pdf](http://arxiv.org/pdf/2509.00088v1)

**Authors**: Ting-Chun Liu, Ching-Yu Hsu, Kuan-Yi Lee, Chi-An Fu, Hung-yi Lee

**Abstract**: Prompt injection attacks pose a significant challenge to the safe deployment of Large Language Models (LLMs) in real-world applications. While prompt-based detection offers a lightweight and interpretable defense strategy, its effectiveness has been hindered by the need for manual prompt engineering. To address this issue, we propose AEGIS , an Automated co-Evolutionary framework for Guarding prompt Injections Schema. Both attack and defense prompts are iteratively optimized against each other using a gradient-like natural language prompt optimization technique. This framework enables both attackers and defenders to autonomously evolve via a Textual Gradient Optimization (TGO) module, leveraging feedback from an LLM-guided evaluation loop. We evaluate our system on a real-world assignment grading dataset of prompt injection attacks and demonstrate that our method consistently outperforms existing baselines, achieving superior robustness in both attack success and detection. Specifically, the attack success rate (ASR) reaches 1.0, representing an improvement of 0.26 over the baseline. For detection, the true positive rate (TPR) improves by 0.23 compared to the previous best work, reaching 0.84, and the true negative rate (TNR) remains comparable at 0.89. Ablation studies confirm the importance of co-evolution, gradient buffering, and multi-objective optimization. We also confirm that this framework is effective in different LLMs. Our results highlight the promise of adversarial training as a scalable and effective approach for guarding prompt injections.



