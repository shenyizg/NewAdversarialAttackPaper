# Latest Adversarial Attack Papers
**update at 2025-09-01 10:13:26**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Detecting Stealthy Data Poisoning Attacks in AI Code Generators**

cs.CR

Accepted to the 3rd IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE, 2025), co-located with ISSRE  2025

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21636v1) [paper-pdf](http://arxiv.org/pdf/2508.21636v1)

**Authors**: Cristina Improta

**Abstract**: Deep learning (DL) models for natural language-to-code generation have become integral to modern software development pipelines. However, their heavy reliance on large amounts of data, often collected from unsanitized online sources, exposes them to data poisoning attacks, where adversaries inject malicious samples to subtly bias model behavior. Recent targeted attacks silently replace secure code with semantically equivalent but vulnerable implementations without relying on explicit triggers to launch the attack, making it especially hard for detection methods to distinguish clean from poisoned samples. We present a systematic study on the effectiveness of existing poisoning detection methods under this stealthy threat model. Specifically, we perform targeted poisoning on three DL models (CodeBERT, CodeT5+, AST-T5), and evaluate spectral signatures analysis, activation clustering, and static analysis as defenses. Our results show that all methods struggle to detect triggerless poisoning, with representation-based approaches failing to isolate poisoned samples and static analysis suffering false positives and false negatives, highlighting the need for more robust, trigger-independent defenses for AI-assisted code generation.



## **2. Adversarial Patch Attack for Ship Detection via Localized Augmentation**

cs.CV

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21472v1) [paper-pdf](http://arxiv.org/pdf/2508.21472v1)

**Authors**: Chun Liu, Panpan Ding, Zheng Zheng, Hailong Wang, Bingqian Zhu, Tao Xu, Zhigang Han, Jiayao Wang

**Abstract**: Current ship detection techniques based on remote sensing imagery primarily rely on the object detection capabilities of deep neural networks (DNNs). However, DNNs are vulnerable to adversarial patch attacks, which can lead to misclassification by the detection model or complete evasion of the targets. Numerous studies have demonstrated that data transformation-based methods can improve the transferability of adversarial examples. However, excessive augmentation of image backgrounds or irrelevant regions may introduce unnecessary interference, resulting in false detections of the object detection model. These errors are not caused by the adversarial patches themselves but rather by the over-augmentation of background and non-target areas. This paper proposes a localized augmentation method that applies augmentation only to the target regions, avoiding any influence on non-target areas. By reducing background interference, this approach enables the loss function to focus more directly on the impact of the adversarial patch on the detection model, thereby improving the attack success rate. Experiments conducted on the HRSC2016 dataset demonstrate that the proposed method effectively increases the success rate of adversarial patch attacks and enhances their transferability.



## **3. Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review**

cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.20863v2) [paper-pdf](http://arxiv.org/pdf/2508.20863v2)

**Authors**: Matteo Gioele Collu, Umberto Salviati, Roberto Confalonieri, Mauro Conti, Giovanni Apruzzese

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.



## **4. Time Tells All: Deanonymization of Blockchain RPC Users with Zero Transaction Fee (Extended Version)**

cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21440v1) [paper-pdf](http://arxiv.org/pdf/2508.21440v1)

**Authors**: Shan Wang, Ming Yang, Yu Liu, Yue Zhang, Shuaiqing Zhang, Zhen Ling, Jiannong Cao, Xinwen Fu

**Abstract**: Remote Procedure Call (RPC) services have become a primary gateway for users to access public blockchains. While they offer significant convenience, RPC services also introduce critical privacy challenges that remain insufficiently examined. Existing deanonymization attacks either do not apply to blockchain RPC users or incur costs like transaction fees assuming an active network eavesdropper. In this paper, we propose a novel deanonymization attack that can link an IP address of a RPC user to this user's blockchain pseudonym. Our analysis reveals a temporal correlation between the timestamps of transaction confirmations recorded on the public ledger and those of TCP packets sent by the victim when querying transaction status. We assume a strong passive adversary with access to network infrastructure, capable of monitoring traffic at network border routers or Internet exchange points. By monitoring network traffic and analyzing public ledgers, the attacker can link the IP address of the TCP packet to the pseudonym of the transaction initiator by exploiting the temporal correlation. This deanonymization attack incurs zero transaction fee. We mathematically model and analyze the attack method, perform large-scale measurements of blockchain ledgers, and conduct real-world attacks to validate the attack. Our attack achieves a high success rate of over 95% against normal RPC users on various blockchain networks, including Ethereum, Bitcoin and Solana.



## **5. On the Adversarial Robustness of Spiking Neural Networks Trained by Local Learning**

cs.LG

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2504.08897v2) [paper-pdf](http://arxiv.org/pdf/2504.08897v2)

**Authors**: Jiaqi Lin, Abhronil Sengupta

**Abstract**: Recent research has shown the vulnerability of Spiking Neural Networks (SNNs) under adversarial examples that are nearly indistinguishable from clean data in the context of frame-based and event-based information. The majority of these studies are constrained in generating adversarial examples using Backpropagation Through Time (BPTT), a gradient-based method which lacks biological plausibility. In contrast, local learning methods, which relax many of BPTT's constraints, remain under-explored in the context of adversarial attacks. To address this problem, we examine adversarial robustness in SNNs through the framework of four types of training algorithms. We provide an in-depth analysis of the ineffectiveness of gradient-based adversarial attacks to generate adversarial instances in this scenario. To overcome these limitations, we introduce a hybrid adversarial attack paradigm that leverages the transferability of adversarial instances. The proposed hybrid approach demonstrates superior performance, outperforming existing adversarial attack methods. Furthermore, the generalizability of the method is assessed under multi-step adversarial attacks, adversarial attacks in black-box FGSM scenarios, and within the non-spiking domain.



## **6. The WASM Cloak: Evaluating Browser Fingerprinting Defenses Under WebAssembly based Obfuscation**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21219v1) [paper-pdf](http://arxiv.org/pdf/2508.21219v1)

**Authors**: A H M Nazmus Sakib, Mahsin Bin Akram, Joseph Spracklen, Sahan Kalutarage, Raveen Wijewickrama, Igor Bilogrevic, Murtuza Jadliwala

**Abstract**: Browser fingerprinting defenses have historically focused on detecting JavaScript(JS)-based tracking techniques. However, the widespread adoption of WebAssembly (WASM) introduces a potential blind spot, as adversaries can convert JS to WASM's low-level binary format to obfuscate malicious logic. This paper presents the first systematic evaluation of how such WASM-based obfuscation impacts the robustness of modern fingerprinting defenses. We develop an automated pipeline that translates real-world JS fingerprinting scripts into functional WASM-obfuscated variants and test them against two classes of defenses: state-of-the-art detectors in research literature and commercial, in-browser tools. Our findings reveal a notable divergence: detectors proposed in the research literature that rely on feature-based analysis of source code show moderate vulnerability, stemming from outdated datasets or a lack of WASM compatibility. In contrast, defenses such as browser extensions and native browser features remained completely effective, as their API-level interception is agnostic to the script's underlying implementation. These results highlight a gap between academic and practical defense strategies and offer insights into strengthening detection approaches against WASM-based obfuscation, while also revealing opportunities for more evasive techniques in future attacks.



## **7. First-Place Solution to NeurIPS 2024 Invisible Watermark Removal Challenge**

cs.CV

Winning solution to the NeurIPS 2024 Erasing the Invisible challenge

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21072v1) [paper-pdf](http://arxiv.org/pdf/2508.21072v1)

**Authors**: Fahad Shamshad, Tameem Bakr, Yahia Shaaban, Noor Hussein, Karthik Nandakumar, Nils Lukas

**Abstract**: Content watermarking is an important tool for the authentication and copyright protection of digital media. However, it is unclear whether existing watermarks are robust against adversarial attacks. We present the winning solution to the NeurIPS 2024 Erasing the Invisible challenge, which stress-tests watermark robustness under varying degrees of adversary knowledge. The challenge consisted of two tracks: a black-box and beige-box track, depending on whether the adversary knows which watermarking method was used by the provider. For the beige-box track, we leverage an adaptive VAE-based evasion attack, with a test-time optimization and color-contrast restoration in CIELAB space to preserve the image's quality. For the black-box track, we first cluster images based on their artifacts in the spatial or frequency-domain. Then, we apply image-to-image diffusion models with controlled noise injection and semantic priors from ChatGPT-generated captions to each cluster with optimized parameter settings. Empirical evaluations demonstrate that our method successfully achieves near-perfect watermark removal (95.7%) with negligible impact on the residual image's quality. We hope that our attacks inspire the development of more robust image watermarking methods.



## **8. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20890v1) [paper-pdf](http://arxiv.org/pdf/2508.20890v1)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.



## **9. FusionCounting: Robust visible-infrared image fusion guided by crowd counting via multi-task learning**

cs.CV

11 pages, 9 figures

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20817v1) [paper-pdf](http://arxiv.org/pdf/2508.20817v1)

**Authors**: He Li, Xinyu Liu, Weihang Kong, Xingchen Zhang

**Abstract**: Most visible and infrared image fusion (VIF) methods focus primarily on optimizing fused image quality. Recent studies have begun incorporating downstream tasks, such as semantic segmentation and object detection, to provide semantic guidance for VIF. However, semantic segmentation requires extensive annotations, while object detection, despite reducing annotation efforts compared with segmentation, faces challenges in highly crowded scenes due to overlapping bounding boxes and occlusion. Moreover, although RGB-T crowd counting has gained increasing attention in recent years, no studies have integrated VIF and crowd counting into a unified framework. To address these challenges, we propose FusionCounting, a novel multi-task learning framework that integrates crowd counting into the VIF process. Crowd counting provides a direct quantitative measure of population density with minimal annotation, making it particularly suitable for dense scenes. Our framework leverages both input images and population density information in a mutually beneficial multi-task design. To accelerate convergence and balance tasks contributions, we introduce a dynamic loss function weighting strategy. Furthermore, we incorporate adversarial training to enhance the robustness of both VIF and crowd counting, improving the model's stability and resilience to adversarial attacks. Experimental results on public datasets demonstrate that FusionCounting not only enhances image fusion quality but also achieves superior crowd counting performance.



## **10. Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning**

cs.LG

Project Hompage: https://tokenbuncher.github.io/

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20697v1) [paper-pdf](http://arxiv.org/pdf/2508.20697v1)

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.



## **11. Disruptive Attacks on Face Swapping via Low-Frequency Perceptual Perturbations**

cs.CV

Accepted to IEEE IJCNN 2025

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20595v1) [paper-pdf](http://arxiv.org/pdf/2508.20595v1)

**Authors**: Mengxiao Huang, Minglei Shu, Shuwang Zhou, Zhaoyang Liu

**Abstract**: Deepfake technology, driven by Generative Adversarial Networks (GANs), poses significant risks to privacy and societal security. Existing detection methods are predominantly passive, focusing on post-event analysis without preventing attacks. To address this, we propose an active defense method based on low-frequency perceptual perturbations to disrupt face swapping manipulation, reducing the performance and naturalness of generated content. Unlike prior approaches that used low-frequency perturbations to impact classification accuracy,our method directly targets the generative process of deepfake techniques. We combine frequency and spatial domain features to strengthen defenses. By introducing artifacts through low-frequency perturbations while preserving high-frequency details, we ensure the output remains visually plausible. Additionally, we design a complete architecture featuring an encoder, a perturbation generator, and a decoder, leveraging discrete wavelet transform (DWT) to extract low-frequency components and generate perturbations that disrupt facial manipulation models. Experiments on CelebA-HQ and LFW demonstrate significant reductions in face-swapping effectiveness, improved defense success rates, and preservation of visual quality.



## **12. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2503.06989v4) [paper-pdf](http://arxiv.org/pdf/2503.06989v4)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.



## **13. Formal Verification of Physical Layer Security Protocols for Next-Generation Communication Networks (extended version)**

cs.CR

Extended version (with appendices) of the camera-ready for ICFEM2025;  24 pages, 3 tables, and 6 figures

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.19430v2) [paper-pdf](http://arxiv.org/pdf/2508.19430v2)

**Authors**: Kangfeng Ye, Roberto Metere, Jim Woodcock, Poonam Yadav

**Abstract**: Formal verification is crucial for ensuring the robustness of security protocols against adversarial attacks. The Needham-Schroeder protocol, a foundational authentication mechanism, has been extensively studied, including its integration with Physical Layer Security (PLS) techniques such as watermarking and jamming. Recent research has used ProVerif to verify these mechanisms in terms of secrecy. However, the ProVerif-based approach limits the ability to improve understanding of security beyond verification results. To overcome these limitations, we re-model the same protocol using an Isabelle formalism that generates sound animation, enabling interactive and automated formal verification of security protocols. Our modelling and verification framework is generic and highly configurable, supporting both cryptography and PLS. For the same protocol, we have conducted a comprehensive analysis (secrecy and authenticity in four different eavesdropper locations under both passive and active attacks) using our new web interface. Our findings not only successfully reproduce and reinforce previous results on secrecy but also reveal an uncommon but expected outcome: authenticity is preserved across all examined scenarios, even in cases where secrecy is compromised. We have proposed a PLS-based Diffie-Hellman protocol that integrates watermarking and jamming, and our analysis shows that it is secure for deriving a session key with required authentication. These highlight the advantages of our novel approach, demonstrating its robustness in formally verifying security properties beyond conventional methods.



## **14. Enhancing Resilience for IoE: A Perspective of Networking-Level Safeguard**

cs.CR

To be published in IEEE Network Magazine, 2026

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20504v1) [paper-pdf](http://arxiv.org/pdf/2508.20504v1)

**Authors**: Guan-Yan Yang, Jui-Ning Chen, Farn Wang, Kuo-Hui Yeh

**Abstract**: The Internet of Energy (IoE) integrates IoT-driven digital communication with power grids to enable efficient and sustainable energy systems. Still, its interconnectivity exposes critical infrastructure to sophisticated cyber threats, including adversarial attacks designed to bypass traditional safeguards. Unlike general IoT risks, IoE threats have heightened public safety consequences, demanding resilient solutions. From the networking-level safeguard perspective, we propose a Graph Structure Learning (GSL)-based safeguards framework that jointly optimizes graph topology and node representations to resist adversarial network model manipulation inherently. Through a conceptual overview, architectural discussion, and case study on a security dataset, we demonstrate GSL's superior robustness over representative methods, offering practitioners a viable path to secure IoE networks against evolving attacks. This work highlights the potential of GSL to enhance the resilience and reliability of future IoE networks for practitioners managing critical infrastructure. Lastly, we identify key open challenges and propose future research directions in this novel research area.



## **15. When Memory Becomes a Vulnerability: Towards Multi-turn Jailbreak Attacks against Text-to-Image Generation Systems**

cs.CV

This work proposes a multi-turn jailbreak attack against real-world  chat-based T2I generation systems that intergrate memory mechanism. It also  constructed a simulation system, with considering three industrial-grade  memory mechanisms, 7 kinds of safety filters (both input and output)

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2504.20376v2) [paper-pdf](http://arxiv.org/pdf/2504.20376v2)

**Authors**: Shiqian Zhao, Jiayang Liu, Yiming Li, Runyi Hu, Xiaojun Jia, Wenshu Fan, Xinfeng Li, Jie Zhang, Wei Dong, Tianwei Zhang, Luu Anh Tuan

**Abstract**: Modern text-to-image (T2I) generation systems (e.g., DALL$\cdot$E 3) exploit the memory mechanism, which captures key information in multi-turn interactions for faithful generation. Despite its practicality, the security analyses of this mechanism have fallen far behind. In this paper, we reveal that it can exacerbate the risk of jailbreak attacks. Previous attacks fuse the unsafe target prompt into one ultimate adversarial prompt, which can be easily detected or lead to the generation of non-unsafe images due to under- or over-detoxification. In contrast, we propose embedding the malice at the inception of the chat session in memory, addressing the above limitations.   Specifically, we propose Inception, the first multi-turn jailbreak attack against real-world text-to-image generation systems that explicitly exploits their memory mechanisms. Inception is composed of two key modules: segmentation and recursion. We introduce Segmentation, a semantic-preserving method that generates multi-round prompts. By leveraging NLP analysis techniques, we design policies to decompose a prompt, together with its malicious intent, according to sentence structure, thereby evading safety filters. Recursion further addresses the challenge posed by unsafe sub-prompts that cannot be separated through simple segmentation. It firstly expands the sub-prompt, then invokes segmentation recursively. To facilitate multi-turn adversarial prompts crafting, we build VisionFlow, an emulation T2I system that integrates two-stage safety filters and industrial-grade memory mechanisms. The experiment results show that Inception successfully allures unsafe image generation, surpassing the SOTA by a 20.0\% margin in attack success rate. We also conduct experiments on the real-world commercial T2I generation platforms, further validating the threats of Inception in practice.



## **16. Safe-Control: A Safety Patch for Mitigating Unsafe Content in Text-to-Image Generation Models**

cs.CV

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21099v1) [paper-pdf](http://arxiv.org/pdf/2508.21099v1)

**Authors**: Xiangtao Meng, Yingkai Dong, Ning Yu, Li Wang, Zheng Li, Shanqing Guo

**Abstract**: Despite the advancements in Text-to-Image (T2I) generation models, their potential for misuse or even abuse raises serious safety concerns. Model developers have made tremendous efforts to introduce safety mechanisms that can address these concerns in T2I models. However, the existing safety mechanisms, whether external or internal, either remain susceptible to evasion under distribution shifts or require extensive model-specific adjustments. To address these limitations, we introduce Safe-Control, an innovative plug-and-play safety patch designed to mitigate unsafe content generation in T2I models. Using data-driven strategies and safety-aware conditions, Safe-Control injects safety control signals into the locked T2I model, acting as an update in a patch-like manner. Model developers can also construct various safety patches to meet the evolving safety requirements, which can be flexibly merged into a single, unified patch. Its plug-and-play design further ensures adaptability, making it compatible with other T2I models of similar denoising architecture. We conduct extensive evaluations on six diverse and public T2I models. Empirical results highlight that Safe-Control is effective in reducing unsafe content generation across six diverse T2I models with similar generative architectures, yet it successfully maintains the quality and text alignment of benign images. Compared to seven state-of-the-art safety mechanisms, including both external and internal defenses, Safe-Control significantly outperforms all baselines in reducing unsafe content generation. For example, it reduces the probability of unsafe content generation to 7%, compared to approximately 20% for most baseline methods, under both unsafe prompts and the latest adversarial attacks.



## **17. Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs**

cs.LG

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20333v1) [paper-pdf](http://arxiv.org/pdf/2508.20333v1)

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.



## **18. Differentially Private Federated Quantum Learning via Quantum Noise**

quant-ph

This paper has been accepted at 2025 IEEE International Conference on  Quantum Computing and Engineering (QCE)

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20310v1) [paper-pdf](http://arxiv.org/pdf/2508.20310v1)

**Authors**: Atit Pokharel, Ratun Rahman, Shaba Shaon, Thomas Morris, Dinh C. Nguyen

**Abstract**: Quantum federated learning (QFL) enables collaborative training of quantum machine learning (QML) models across distributed quantum devices without raw data exchange. However, QFL remains vulnerable to adversarial attacks, where shared QML model updates can be exploited to undermine information privacy. In the context of noisy intermediate-scale quantum (NISQ) devices, a key question arises: How can inherent quantum noise be leveraged to enforce differential privacy (DP) and protect model information during training and communication? This paper explores a novel DP mechanism that harnesses quantum noise to safeguard quantum models throughout the QFL process. By tuning noise variance through measurement shots and depolarizing channel strength, our approach achieves desired DP levels tailored to NISQ constraints. Simulations demonstrate the framework's effectiveness by examining the relationship between differential privacy budget and noise parameters, as well as the trade-off between security and training accuracy. Additionally, we demonstrate the framework's robustness against an adversarial attack designed to compromise model performance using adversarial examples, with evaluations based on critical metrics such as accuracy on adversarial examples, confidence scores for correct predictions, and attack success rates. The results reveal a tunable trade-off between privacy and robustness, providing an efficient solution for secure QFL on NISQ devices with significant potential for reliable quantum computing applications.



## **19. CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors**

cs.CL

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.02997v3) [paper-pdf](http://arxiv.org/pdf/2508.02997v3)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models.



## **20. Network-Level Prompt and Trait Leakage in Local Research Agents**

cs.CR

under review

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20282v1) [paper-pdf](http://arxiv.org/pdf/2508.20282v1)

**Authors**: Hyejun Jeong, Mohammadreze Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Badasarian

**Abstract**: We show that Web and Research Agents (WRAs) -- language model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network adversaries such as ISPs. These agents could be deployed \emph{locally} by organizations and individuals for privacy, legal, or financial purposes. Unlike sporadic web browsing by humans, WRAs visit $70{-}140$ domains with distinguishable timing correlations, enabling unique fingerprinting attacks.   Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73\% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29\%.



## **21. Adversarial Manipulation of Reasoning Models using Internal Representations**

cs.CL

Accepted to the ICML 2025 Workshop on Reliable and Responsible  Foundation Models (R2FM). 20 pages, 12 figures

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.03167v2) [paper-pdf](http://arxiv.org/pdf/2507.03167v2)

**Authors**: Kureha Yamaguchi, Benjamin Etheridge, Andy Arditi

**Abstract**: Reasoning models generate chain-of-thought (CoT) tokens before their final output, but how this affects their vulnerability to jailbreak attacks remains unclear. While traditional language models make refusal decisions at the prompt-response boundary, we find evidence that DeepSeek-R1-Distill-Llama-8B makes these decisions within its CoT generation. We identify a linear direction in activation space during CoT token generation that predicts whether the model will refuse or comply -- termed the "caution" direction because it corresponds to cautious reasoning patterns in the generated text. Ablating this direction from model activations increases harmful compliance, effectively jailbreaking the model. We additionally show that intervening only on CoT token activations suffices to control final outputs, and that incorporating this direction into prompt-based attacks improves success rates. Our findings suggest that the chain-of-thought itself is a promising new target for adversarial manipulation in reasoning models. Code available at https://github.com/ky295/reasoning-manipulation.



## **22. Scaling Decentralized Learning with FLock**

cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.15349v2) [paper-pdf](http://arxiv.org/pdf/2507.15349v2)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.



## **23. Cell-Free Massive MIMO-Based Physical-Layer Authentication**

eess.SP

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19931v1) [paper-pdf](http://arxiv.org/pdf/2508.19931v1)

**Authors**: Isabella W. G. da Silva, Zahra Mobini, Hien Quoc Ngo, Michail Matthaiou

**Abstract**: In this paper, we exploit the cell-free massive multiple-input multiple-output (CF-mMIMO) architecture to design a physical-layer authentication (PLA) framework that can simultaneously authenticate multiple distributed users across the coverage area. Our proposed scheme remains effective even in the presence of active adversaries attempting impersonation attacks to disrupt the authentication process. Specifically, we introduce a tag-based PLA CFmMIMO system, wherein the access points (APs) first estimate their channels with the legitimate users during an uplink training phase. Subsequently, a unique secret key is generated and securely shared between each user and the APs. We then formulate a hypothesis testing problem and derive a closed-form expression for the probability of detection for each user in the network. Numerical results validate the effectiveness of the proposed approach, demonstrating that it maintains a high detection probability even as the number of users in the system increases.



## **24. When AIOps Become "AI Oops": Subverting LLM-driven IT Operations via Telemetry Manipulation**

cs.CR

v0.2

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.06394v2) [paper-pdf](http://arxiv.org/pdf/2508.06394v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese, Omer Akgul, Athanasios Theocharis, Petros Efstathopoulos

**Abstract**: AI for IT Operations (AIOps) is transforming how organizations manage complex software systems by automating anomaly detection, incident diagnosis, and remediation. Modern AIOps solutions increasingly rely on autonomous LLM-based agents to interpret telemetry data and take corrective actions with minimal human intervention, promising faster response times and operational cost savings.   In this work, we perform the first security analysis of AIOps solutions, showing that, once again, AI-driven automation comes with a profound security cost. We demonstrate that adversaries can manipulate system telemetry to mislead AIOps agents into taking actions that compromise the integrity of the infrastructure they manage. We introduce techniques to reliably inject telemetry data using error-inducing requests that influence agent behavior through a form of adversarial reward-hacking; plausible but incorrect system error interpretations that steer the agent's decision-making. Our attack methodology, AIOpsDoom, is fully automated--combining reconnaissance, fuzzing, and LLM-driven adversarial input generation--and operates without any prior knowledge of the target system.   To counter this threat, we propose AIOpsShield, a defense mechanism that sanitizes telemetry data by exploiting its structured nature and the minimal role of user-generated content. Our experiments show that AIOpsShield reliably blocks telemetry-based attacks without affecting normal agent performance.   Ultimately, this work exposes AIOps as an emerging attack vector for system compromise and underscores the urgent need for security-aware AIOps design.



## **25. DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective**

cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.05622v2) [paper-pdf](http://arxiv.org/pdf/2507.05622v2)

**Authors**: Shuo Shao, Yiming Li, Mengren Zheng, Zhiyang Hu, Yukun Chen, Boheng Li, Yu He, Junfeng Guo, Dacheng Tao, Zhan Qin

**Abstract**: The widespread application of Deep Learning across diverse domains hinges critically on the quality and composition of training datasets. However, the common lack of disclosure regarding their usage raises significant privacy and copyright concerns. Dataset auditing techniques, which aim to determine if a specific dataset was used to train a given suspicious model, provide promising solutions to addressing these transparency gaps. While prior work has developed various auditing methods, their resilience against dedicated adversarial attacks remains largely unexplored. To bridge the gap, this paper initiates a comprehensive study evaluating dataset auditing from an adversarial perspective. We start with introducing a novel taxonomy, classifying existing methods based on their reliance on internal features (IF) (inherent to the data) versus external features (EF) (artificially introduced for auditing). Subsequently, we formulate two primary attack types: evasion attacks, designed to conceal the use of a dataset, and forgery attacks, intending to falsely implicate an unused dataset. Building on the understanding of existing methods and attack objectives, we further propose systematic attack strategies: decoupling, removal, and detection for evasion; adversarial example-based methods for forgery. These formulations and strategies lead to our new benchmark, DATABench, comprising 17 evasion attacks, 5 forgery attacks, and 9 representative auditing methods. Extensive evaluations using DATABench reveal that none of the evaluated auditing methods are sufficiently robust or distinctive under adversarial settings. These findings underscore the urgent need for developing a more secure and reliable dataset auditing method capable of withstanding sophisticated adversarial manipulation. Code is available at https://github.com/shaoshuo-ss/DATABench.



## **26. Secure Set-based State Estimation for Safety-Critical Applications under Adversarial Attacks on Sensors**

eess.SY

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2309.05075v3) [paper-pdf](http://arxiv.org/pdf/2309.05075v3)

**Authors**: M. Umar B. Niazi, Michelle S. Chong, Amr Alanwar, Karl H. Johansson

**Abstract**: Set-based state estimation provides guaranteed state inclusion certificates that are crucial for the safety verification of dynamical systems. However, when system sensors are subject to cyberattacks, maintaining both safety and security guarantees becomes a fundamental challenge that existing point-based secure state estimation methods cannot adequately address due to their inherent inability to provide state inclusion certificates. This paper introduces a novel approach that simultaneously ensures safety guarantees through guaranteed state inclusion and security guarantees against sensor attacks, without imposing conservative restrictions on system operation. We propose a Secure Set-based State Estimation (S3E) algorithm that maintains the true system state within the estimated set under sensor attacks, provided the initialization set contains the initial state and the system remains observable from the uncompromised sensor subset. The algorithm gives the estimated set as a collection of constrained zonotopes (agreement sets), which can be employed as robust certificates for verifying whether the system adheres to safety constraints. Furthermore, we demonstrate that the estimated set remains unaffected by attack signals of sufficiently large magnitude and also establish sufficient conditions for attack detection, identification, and filtering. This compels the attacker to only inject signals of small magnitudes to evade detection, thus preserving the accuracy of the estimated set. To address the computational complexity of the algorithm, we offer several strategies for complexity-performance trade-offs. The efficacy of the proposed algorithm is illustrated through several examples, including its application to a three-story building model.



## **27. Safety Alignment Should Be Made More Than Just A Few Attention Heads**

cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19697v1) [paper-pdf](http://arxiv.org/pdf/2508.19697v1)

**Authors**: Chao Huang, Zefeng Zhang, Juewei Yue, Quangang Li, Chuang Zhang, Tingwen Liu

**Abstract**: Current safety alignment for large language models(LLMs) continues to present vulnerabilities, given that adversarial prompting can effectively bypass their safety measures.Our investigation shows that these safety mechanisms predominantly depend on a limited subset of attention heads: removing or ablating these heads can severely compromise model safety. To identify and evaluate these safety-critical components, we introduce RDSHA, a targeted ablation method that leverages the model's refusal direction to pinpoint attention heads mostly responsible for safety behaviors. Further analysis shows that existing jailbreak attacks exploit this concentration by selectively bypassing or manipulating these critical attention heads. To address this issue, we propose AHD, a novel training strategy designed to promote the distributed encoding of safety-related behaviors across numerous attention heads. Experimental results demonstrate that AHD successfully distributes safety-related capabilities across more attention heads. Moreover, evaluations under several mainstream jailbreak attacks show that models trained with AHD exhibit considerably stronger safety robustness, while maintaining overall functional utility.



## **28. ProARD: progressive adversarial robustness distillation: provide wide range of robust students**

cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2506.07666v3) [paper-pdf](http://arxiv.org/pdf/2506.07666v3)

**Authors**: Seyedhamidreza Mousavi, Seyedali Mousavi, Masoud Daneshtalab

**Abstract**: Adversarial Robustness Distillation (ARD) has emerged as an effective method to enhance the robustness of lightweight deep neural networks against adversarial attacks. Current ARD approaches have leveraged a large robust teacher network to train one robust lightweight student. However, due to the diverse range of edge devices and resource constraints, current approaches require training a new student network from scratch to meet specific constraints, leading to substantial computational costs and increased CO2 emissions. This paper proposes Progressive Adversarial Robustness Distillation (ProARD), enabling the efficient one-time training of a dynamic network that supports a diverse range of accurate and robust student networks without requiring retraining. We first make a dynamic deep neural network based on dynamic layers by encompassing variations in width, depth, and expansion in each design stage to support a wide range of architectures. Then, we consider the student network with the largest size as the dynamic teacher network. ProARD trains this dynamic network using a weight-sharing mechanism to jointly optimize the dynamic teacher network and its internal student networks. However, due to the high computational cost of calculating exact gradients for all the students within the dynamic network, a sampling mechanism is required to select a subset of students. We show that random student sampling in each iteration fails to produce accurate and robust students.



## **29. R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning**

cs.LG

CVPR 2025 (Corrected the results on the Aircraft dataset)

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2504.11195v2) [paper-pdf](http://arxiv.org/pdf/2504.11195v2)

**Authors**: Lijun Sheng, Jian Liang, Zilei Wang, Ran He

**Abstract**: Vision-language models (VLMs), such as CLIP, have gained significant popularity as foundation models, with numerous fine-tuning methods developed to enhance performance on downstream tasks. However, due to their inherent vulnerability and the common practice of selecting from a limited set of open-source models, VLMs suffer from a higher risk of adversarial attacks than traditional vision models. Existing defense techniques typically rely on adversarial fine-tuning during training, which requires labeled data and lacks of flexibility for downstream tasks. To address these limitations, we propose robust test-time prompt tuning (R-TPT), which mitigates the impact of adversarial attacks during the inference stage. We first reformulate the classic marginal entropy objective by eliminating the term that introduces conflicts under adversarial conditions, retaining only the pointwise entropy minimization. Furthermore, we introduce a plug-and-play reliability-based weighted ensembling strategy, which aggregates useful information from reliable augmented views to strengthen the defense. R-TPT enhances defense against adversarial attacks without requiring labeled training data while offering high flexibility for inference tasks. Extensive experiments on widely used benchmarks with various attacks demonstrate the effectiveness of R-TPT. The code is available in https://github.com/TomSheng21/R-TPT.



## **30. PromptKeeper: Safeguarding System Prompts for LLMs**

cs.CR

Accepted to the Findings of EMNLP 2025. 17 pages, 6 figures, 3 tables

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2412.13426v3) [paper-pdf](http://arxiv.org/pdf/2412.13426v3)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: System prompts are widely used to guide the outputs of large language models (LLMs). These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a defense mechanism designed to safeguard system prompts by tackling two core challenges: reliably detecting leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon leakage detected, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.



## **31. A Systematic Survey of Model Extraction Attacks and Defenses: State-of-the-Art and Perspectives**

cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.15031v2) [paper-pdf](http://arxiv.org/pdf/2508.15031v2)

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong

**Abstract**: Machine learning (ML) models have significantly grown in complexity and utility, driving advances across multiple domains. However, substantial computational resources and specialized expertise have historically restricted their wide adoption. Machine-Learning-as-a-Service (MLaaS) platforms have addressed these barriers by providing scalable, convenient, and affordable access to sophisticated ML models through user-friendly APIs. While this accessibility promotes widespread use of advanced ML capabilities, it also introduces vulnerabilities exploited through Model Extraction Attacks (MEAs). Recent studies have demonstrated that adversaries can systematically replicate a target model's functionality by interacting with publicly exposed interfaces, posing threats to intellectual property, privacy, and system security. In this paper, we offer a comprehensive survey of MEAs and corresponding defense strategies. We propose a novel taxonomy that classifies MEAs according to attack mechanisms, defense approaches, and computing environments. Our analysis covers various attack techniques, evaluates their effectiveness, and highlights challenges faced by existing defenses, particularly the critical trade-off between preserving model utility and ensuring security. We further assess MEAs within different computing paradigms and discuss their technical, ethical, legal, and societal implications, along with promising directions for future research. This systematic survey aims to serve as a valuable reference for researchers, practitioners, and policymakers engaged in AI security and privacy. Additionally, we maintain an online repository continuously updated with related literature at https://github.com/kzhao5/ModelExtractionPapers.



## **32. Servant, Stalker, Predator: How An Honest, Helpful, And Harmless (3H) Agent Unlocks Adversarial Skills**

cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19500v1) [paper-pdf](http://arxiv.org/pdf/2508.19500v1)

**Authors**: David Noever

**Abstract**: This paper identifies and analyzes a novel vulnerability class in Model Context Protocol (MCP) based agent systems. The attack chain describes and demonstrates how benign, individually authorized tasks can be orchestrated to produce harmful emergent behaviors. Through systematic analysis using the MITRE ATLAS framework, we demonstrate how 95 agents tested with access to multiple services-including browser automation, financial analysis, location tracking, and code deployment-can chain legitimate operations into sophisticated attack sequences that extend beyond the security boundaries of any individual service. These red team exercises survey whether current MCP architectures lack cross-domain security measures necessary to detect or prevent a large category of compositional attacks. We present empirical evidence of specific attack chains that achieve targeted harm through service orchestration, including data exfiltration, financial manipulation, and infrastructure compromise. These findings reveal that the fundamental security assumption of service isolation fails when agents can coordinate actions across multiple domains, creating an exponential attack surface that grows with each additional capability. This research provides a barebones experimental framework that evaluate not whether agents can complete MCP benchmark tasks, but what happens when they complete them too well and optimize across multiple services in ways that violate human expectations and safety constraints. We propose three concrete experimental directions using the existing MCP benchmark suite.



## **33. PoolFlip: A Multi-Agent Reinforcement Learning Security Environment for Cyber Defense**

cs.LG

Accepted at GameSec 2025

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19488v1) [paper-pdf](http://arxiv.org/pdf/2508.19488v1)

**Authors**: Xavier Cadet, Simona Boboila, Sie Hendrata Dharmawan, Alina Oprea, Peter Chin

**Abstract**: Cyber defense requires automating defensive decision-making under stealthy, deceptive, and continuously evolving adversarial strategies. The FlipIt game provides a foundational framework for modeling interactions between a defender and an advanced adversary that compromises a system without being immediately detected. In FlipIt, the attacker and defender compete to control a shared resource by performing a Flip action and paying a cost. However, the existing FlipIt frameworks rely on a small number of heuristics or specialized learning techniques, which can lead to brittleness and the inability to adapt to new attacks. To address these limitations, we introduce PoolFlip, a multi-agent gym environment that extends the FlipIt game to allow efficient learning for attackers and defenders. Furthermore, we propose Flip-PSRO, a multi-agent reinforcement learning (MARL) approach that leverages population-based training to train defender agents equipped to generalize against a range of unknown, potentially adaptive opponents. Our empirical results suggest that Flip-PSRO defenders are $2\times$ more effective than baselines to generalize to a heuristic attack not exposed in training. In addition, our newly designed ownership-based utility functions ensure that Flip-PSRO defenders maintain a high level of control while optimizing performance.



## **34. ReLATE+: Unified Framework for Adversarial Attack Detection, Classification, and Resilient Model Selection in Time-Series Classification**

cs.CR

Under review at IEEE TSMC Journal. arXiv admin note: text overlap  with arXiv:2503.07882

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19456v1) [paper-pdf](http://arxiv.org/pdf/2508.19456v1)

**Authors**: Cagla Ipek Kocal, Onat Gungor, Tajana Rosing, Baris Aksanli

**Abstract**: Minimizing computational overhead in time-series classification, particularly in deep learning models, presents a significant challenge due to the high complexity of model architectures and the large volume of sequential data that must be processed in real time. This challenge is further compounded by adversarial attacks, emphasizing the need for resilient methods that ensure robust performance and efficient model selection. To address this challenge, we propose ReLATE+, a comprehensive framework that detects and classifies adversarial attacks, adaptively selects deep learning models based on dataset-level similarity, and thus substantially reduces retraining costs relative to conventional methods that do not leverage prior knowledge, while maintaining strong performance. ReLATE+ first checks whether the incoming data is adversarial and, if so, classifies the attack type, using this insight to identify a similar dataset from a repository and enable the reuse of the best-performing associated model. This approach ensures strong performance while reducing the need for retraining, and it generalizes well across different domains with varying data distributions and feature spaces. Experiments show that ReLATE+ reduces computational overhead by an average of 77.68%, enhancing adversarial resilience and streamlining robust model selection, all without sacrificing performance, within 2.02% of Oracle.



## **35. On Surjectivity of Neural Networks: Can you elicit any behavior from your model?**

cs.LG

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19445v1) [paper-pdf](http://arxiv.org/pdf/2508.19445v1)

**Authors**: Haozhe Jiang, Nika Haghtalab

**Abstract**: Given a trained neural network, can any specified output be generated by some input? Equivalently, does the network correspond to a function that is surjective? In generative models, surjectivity implies that any output, including harmful or undesirable content, can in principle be generated by the networks, raising concerns about model safety and jailbreak vulnerabilities. In this paper, we prove that many fundamental building blocks of modern neural architectures, such as networks with pre-layer normalization and linear-attention modules, are almost always surjective. As corollaries, widely used generative frameworks, including GPT-style transformers and diffusion models with deterministic ODE solvers, admit inverse mappings for arbitrary outputs. By studying surjectivity of these modern and commonly used neural architectures, we contribute a formalism that sheds light on their unavoidable vulnerability to a broad class of adversarial attacks.



## **36. Attackers Strike Back? Not Anymore -- An Ensemble of RL Defenders Awakens for APT Detection**

cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19072v1) [paper-pdf](http://arxiv.org/pdf/2508.19072v1)

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) represent a growing menace to modern digital infrastructure. Unlike traditional cyberattacks, APTs are stealthy, adaptive, and long-lasting, often bypassing signature-based detection systems. This paper introduces a novel framework for APT detection that unites deep learning, reinforcement learning (RL), and active learning into a cohesive, adaptive defense system. Our system combines auto-encoders for latent behavioral encoding with a multi-agent ensemble of RL-based defenders, each trained to distinguish between benign and malicious process behaviors. We identify a critical challenge in existing detection systems: their static nature and inability to adapt to evolving attack strategies. To this end, our architecture includes multiple RL agents (Q-Learning, PPO, DQN, adversarial defenders), each analyzing latent vectors generated by an auto-encoder. When any agent is uncertain about its decision, the system triggers an active learning loop to simulate expert feedback, thus refining decision boundaries. An ensemble voting mechanism, weighted by each agent's performance, ensures robust final predictions.



## **37. Beyond-Diagonal RIS: Adversarial Channels and Optimality of Low-Complexity Architectures**

eess.SP

\copyright 2025 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19000v1) [paper-pdf](http://arxiv.org/pdf/2508.19000v1)

**Authors**: Atso Iivanainen, Robin Rajamäki, Visa Koivunen

**Abstract**: Beyond-diagonal reconfigurable intelligent surfaces (BD-RISs) have recently gained attention as an enhancement to conventional RISs. BD-RISs allow optimizing not only the phase, but also the amplitude responses of their discrete surface elements by introducing adjustable inter-element couplings. Various BD-RIS architectures have been proposed to optimally trade off between average performance and complexity of the architecture. However, little attention has been paid to worst-case performance. This paper characterizes novel sets of adversarial channels for which certain low-complexity BD-RIS architectures have suboptimal performance in terms of received signal power at an intended communications user. Specifically, we consider two recent BD-RIS models: the so-called group-connected and tree-connected architecture. The derived adversarial channel sets reveal new surprising connections between the two architectures. We validate our analytical results numerically, demonstrating that adversarial channels can cause a significant performance loss. Our results pave the way towards efficient BD-RIS designs that are robust to adversarial propagation conditions and malicious attacks.



## **38. The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization**

cs.CR

15 pages, 4 figures, 8 tables. Accepted to WPES @ CCS 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18976v1) [paper-pdf](http://arxiv.org/pdf/2508.18976v1)

**Authors**: Stephen Meisenbacher, Alexandra Klymenko, Andreea-Elena Bodea, Florian Matthes

**Abstract**: Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially.



## **39. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

cs.CR

This paper will be submitted to the Computer Science Review

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.16843v2) [paper-pdf](http://arxiv.org/pdf/2508.16843v2)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.



## **40. EnerSwap: Large-Scale, Privacy-First Automated Market Maker for V2G Energy Trading**

cs.CR

11 pages, 7 figures, 1 table, 1 algorithm, Paper accepted in 27th  MSWiM Conference

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18942v1) [paper-pdf](http://arxiv.org/pdf/2508.18942v1)

**Authors**: Ahmed Mounsf Rafik Bendada, Yacine Ghamri-Doudane

**Abstract**: With the rapid growth of Electric Vehicle (EV) technology, EVs are destined to shape the future of transportation. The large number of EVs facilitates the development of the emerging vehicle-to-grid (V2G) technology, which realizes bidirectional energy exchanges between EVs and the power grid. This has led to the setting up of electricity markets that are usually confined to a small geographical location, often with a small number of participants. Usually, these markets are manipulated by intermediaries responsible for collecting bids from prosumers, determining the market-clearing price, incorporating grid constraints, and accounting for network losses. While centralized models can be highly efficient, they grant excessive power to the intermediary by allowing them to gain exclusive access to prosumers \textquotesingle price preferences. This opens the door to potential market manipulation and raises significant privacy concerns for users, such as the location of energy providers. This lack of protection exposes users to potential risks, as untrustworthy servers and malicious adversaries can exploit this information to infer trading activities and real identities. This work proposes a secure, decentralized exchange market built on blockchain technology, utilizing a privacy-preserving Automated Market Maker (AMM) model to offer open and fair, and equal access to traders, and mitigates the most common trading-manipulation attacks. Additionally, it incorporates a scalable architecture based on geographical dynamic sharding, allowing for efficient resource allocation and improved performance as the market grows.



## **41. Hidden Tail: Adversarial Image Causing Stealthy Resource Consumption in Vision-Language Models**

cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18805v1) [paper-pdf](http://arxiv.org/pdf/2508.18805v1)

**Authors**: Rui Zhang, Zihan Wang, Tianli Yang, Hongwei Li, Wenbo Jiang, Qingchuan Zhao, Yang Liu, Guowen Xu

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in real-world applications, but their high inference cost makes them vulnerable to resource consumption attacks. Prior attacks attempt to extend VLM output sequences by optimizing adversarial images, thereby increasing inference costs. However, these extended outputs often introduce irrelevant abnormal content, compromising attack stealthiness. This trade-off between effectiveness and stealthiness poses a major limitation for existing attacks. To address this challenge, we propose \textit{Hidden Tail}, a stealthy resource consumption attack that crafts prompt-agnostic adversarial images, inducing VLMs to generate maximum-length outputs by appending special tokens invisible to users. Our method employs a composite loss function that balances semantic preservation, repetitive special token induction, and suppression of the end-of-sequence (EOS) token, optimized via a dynamic weighting strategy. Extensive experiments show that \textit{Hidden Tail} outperforms existing attacks, increasing output length by up to 19.2$\times$ and reaching the maximum token limit, while preserving attack stealthiness. These results highlight the urgent need to improve the robustness of VLMs against efficiency-oriented adversarial threats. Our code is available at https://github.com/zhangrui4041/Hidden_Tail.



## **42. FLAegis: A Two-Layer Defense Framework for Federated Learning Against Poisoning Attacks**

cs.LG

15 pages, 5 tables, and 5 figures

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18737v1) [paper-pdf](http://arxiv.org/pdf/2508.18737v1)

**Authors**: Enrique Mármol Campos, Aurora González Vidal, José Luis Hernández Ramos, Antonio Skarmeta

**Abstract**: Federated Learning (FL) has become a powerful technique for training Machine Learning (ML) models in a decentralized manner, preserving the privacy of the training datasets involved. However, the decentralized nature of FL limits the visibility of the training process, relying heavily on the honesty of participating clients. This assumption opens the door to malicious third parties, known as Byzantine clients, which can poison the training process by submitting false model updates. Such malicious clients may engage in poisoning attacks, manipulating either the dataset or the model parameters to induce misclassification. In response, this study introduces FLAegis, a two-stage defensive framework designed to identify Byzantine clients and improve the robustness of FL systems. Our approach leverages symbolic time series transformation (SAX) to amplify the differences between benign and malicious models, and spectral clustering, which enables accurate detection of adversarial behavior. Furthermore, we incorporate a robust FFT-based aggregation function as a final layer to mitigate the impact of those Byzantine clients that manage to evade prior defenses. We rigorously evaluate our method against five poisoning attacks, ranging from simple label flipping to adaptive optimization-based strategies. Notably, our approach outperforms state-of-the-art defenses in both detection precision and final model accuracy, maintaining consistently high performance even under strong adversarial conditions.



## **43. UniC-RAG: Universal Knowledge Corruption Attacks to Retrieval-Augmented Generation**

cs.CR

21 pages, 4 figures

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18652v1) [paper-pdf](http://arxiv.org/pdf/2508.18652v1)

**Authors**: Runpeng Geng, Yanting Wang, Ying Chen, Jinyuan Jia

**Abstract**: Retrieval-augmented generation (RAG) systems are widely deployed in real-world applications in diverse domains such as finance, healthcare, and cybersecurity. However, many studies showed that they are vulnerable to knowledge corruption attacks, where an attacker can inject adversarial texts into the knowledge database of a RAG system to induce the LLM to generate attacker-desired outputs. Existing studies mainly focus on attacking specific queries or queries with similar topics (or keywords). In this work, we propose UniC-RAG, a universal knowledge corruption attack against RAG systems. Unlike prior work, UniC-RAG jointly optimizes a small number of adversarial texts that can simultaneously attack a large number of user queries with diverse topics and domains, enabling an attacker to achieve various malicious objectives, such as directing users to malicious websites, triggering harmful command execution, or launching denial-of-service attacks. We formulate UniC-RAG as an optimization problem and further design an effective solution to solve it, including a balanced similarity-based clustering method to enhance the attack's effectiveness. Our extensive evaluations demonstrate that UniC-RAG is highly effective and significantly outperforms baselines. For instance, UniC-RAG could achieve over 90% attack success rate by injecting 100 adversarial texts into a knowledge database with millions of texts to simultaneously attack a large set of user queries (e.g., 2,000). Additionally, we evaluate existing defenses and show that they are insufficient to defend against UniC-RAG, highlighting the need for new defense mechanisms in RAG systems.



## **44. PRISM: Robust VLM Alignment with Principled Reasoning for Integrated Safety in Multimodality**

cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18649v1) [paper-pdf](http://arxiv.org/pdf/2508.18649v1)

**Authors**: Nanxi Li, Zhengyue Zhao, Chaowei Xiao

**Abstract**: Safeguarding vision-language models (VLMs) is a critical challenge, as existing methods often suffer from over-defense, which harms utility, or rely on shallow alignment, failing to detect complex threats that require deep reasoning. To this end, we introduce PRISM (Principled Reasoning for Integrated Safety in Multimodality), a system2-like framework that aligns VLMs by embedding a structured, safety-aware reasoning process. Our framework consists of two key components: PRISM-CoT, a dataset that teaches safety-aware chain-of-thought reasoning, and PRISM-DPO, generated via Monte Carlo Tree Search (MCTS) to further refine this reasoning through Direct Preference Optimization to help obtain a delicate safety boundary. Comprehensive evaluations demonstrate PRISM's effectiveness, achieving remarkably low attack success rates including 0.15% on JailbreakV-28K for Qwen2-VL and 90% improvement over the previous best method on VLBreak for LLaVA-1.5. PRISM also exhibits strong robustness against adaptive attacks, significantly increasing computational costs for adversaries, and generalizes effectively to out-of-distribution challenges, reducing attack success rates to just 8.70% on the challenging multi-image MIS benchmark. Remarkably, this robust defense is achieved while preserving, and in some cases enhancing, model utility. To promote reproducibility, we have made our code, data, and model weights available at https://github.com/SaFoLab-WISC/PRISM.



## **45. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

cs.CL

23 pages, 10 figures, 11 tables

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2503.00187v2) [paper-pdf](http://arxiv.org/pdf/2503.00187v2)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are shown to be vulnerable to jailbreaking attacks where adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment, prompt-based steering and lightweight LLM guardrails baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off among safety, helpfulness and over-refusal. Check out the website here https://sites.google.com/view/llm-nbf/home . Our code is available on https://github.com/HanjiangHu/NBF-LLM .



## **46. Transferring Styles for Reduced Texture Bias and Improved Robustness in Semantic Segmentation Networks**

cs.CV

accepted at ECAI 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.10239v2) [paper-pdf](http://arxiv.org/pdf/2507.10239v2)

**Authors**: Ben Hamscher, Edgar Heinert, Annika Mütze, Kira Maag, Matthias Rottmann

**Abstract**: Recent research has investigated the shape and texture biases of deep neural networks (DNNs) in image classification which influence their generalization capabilities and robustness. It has been shown that, in comparison to regular DNN training, training with stylized images reduces texture biases in image classification and improves robustness with respect to image corruptions. In an effort to advance this line of research, we examine whether style transfer can likewise deliver these two effects in semantic segmentation. To this end, we perform style transfer with style varying across artificial image areas. Those random areas are formed by a chosen number of Voronoi cells. The resulting style-transferred data is then used to train semantic segmentation DNNs with the objective of reducing their dependence on texture cues while enhancing their reliance on shape-based features. In our experiments, it turns out that in semantic segmentation, style transfer augmentation reduces texture bias and strongly increases robustness with respect to common image corruptions as well as adversarial attacks. These observations hold for convolutional neural networks and transformer architectures on the Cityscapes dataset as well as on PASCAL Context, showing the generality of the proposed method.



## **47. Quantum-Classical Hybrid Framework for Zero-Day Time-Push GNSS Spoofing Detection**

cs.LG

This work has been submitted to the IEEE Internet of Things Journal  for possible publication

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18085v1) [paper-pdf](http://arxiv.org/pdf/2508.18085v1)

**Authors**: Abyad Enan, Mashrur Chowdhury, Sagar Dasgupta, Mizanur Rahman

**Abstract**: Global Navigation Satellite Systems (GNSS) are critical for Positioning, Navigation, and Timing (PNT) applications. However, GNSS are highly vulnerable to spoofing attacks, where adversaries transmit counterfeit signals to mislead receivers. Such attacks can lead to severe consequences, including misdirected navigation, compromised data integrity, and operational disruptions. Most existing spoofing detection methods depend on supervised learning techniques and struggle to detect novel, evolved, and unseen attacks. To overcome this limitation, we develop a zero-day spoofing detection method using a Hybrid Quantum-Classical Autoencoder (HQC-AE), trained solely on authentic GNSS signals without exposure to spoofed data. By leveraging features extracted during the tracking stage, our method enables proactive detection before PNT solutions are computed. We focus on spoofing detection in static GNSS receivers, which are particularly susceptible to time-push spoofing attacks, where attackers manipulate timing information to induce incorrect time computations at the receiver. We evaluate our model against different unseen time-push spoofing attack scenarios: simplistic, intermediate, and sophisticated. Our analysis demonstrates that the HQC-AE consistently outperforms its classical counterpart, traditional supervised learning-based models, and existing unsupervised learning-based methods in detecting zero-day, unseen GNSS time-push spoofing attacks, achieving an average detection accuracy of 97.71% with an average false negative rate of 0.62% (when an attack occurs but is not detected). For sophisticated spoofing attacks, the HQC-AE attains an accuracy of 98.23% with a false negative rate of 1.85%. These findings highlight the effectiveness of our method in proactively detecting zero-day GNSS time-push spoofing attacks across various stationary GNSS receiver platforms.



## **48. Robust Federated Learning under Adversarial Attacks via Loss-Based Client Clustering**

cs.LG

16 pages, 5 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.12672v3) [paper-pdf](http://arxiv.org/pdf/2508.12672v3)

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing private data. We consider FL scenarios wherein FL clients are subject to adversarial (Byzantine) attacks, while the FL server is trusted (honest) and has a trustworthy side dataset. This may correspond to, e.g., cases where the server possesses trusted data prior to federation, or to the presence of a trusted client that temporarily assumes the server role. Our approach requires only two honest participants, i.e., the server and one client, to function effectively, without prior knowledge of the number of malicious clients. Theoretical analysis demonstrates bounded optimality gaps even under strong Byzantine attacks. Experimental results show that our algorithm significantly outperforms standard and robust FL baselines such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum under various attack strategies including label flipping, sign flipping, and Gaussian noise addition across MNIST, FMNIST, and CIFAR-10 benchmarks using the Flower framework.



## **49. FedGreed: A Byzantine-Robust Loss-Based Aggregation Method for Federated Learning**

cs.LG

8 pages, 4 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18060v1) [paper-pdf](http://arxiv.org/pdf/2508.18060v1)

**Authors**: Emmanouil Kritharakis, Antonios Makris, Dusan Jakovetic, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients while preserving data privacy by keeping local datasets on-device. In this work, we address FL settings where clients may behave adversarially, exhibiting Byzantine attacks, while the central server is trusted and equipped with a reference dataset. We propose FedGreed, a resilient aggregation strategy for federated learning that does not require any assumptions about the fraction of adversarial participants. FedGreed orders clients' local model updates based on their loss metrics evaluated against a trusted dataset on the server and greedily selects a subset of clients whose models exhibit the minimal evaluation loss. Unlike many existing approaches, our method is designed to operate reliably under heterogeneous (non-IID) data distributions, which are prevalent in real-world deployments. FedGreed exhibits convergence guarantees and bounded optimality gaps under strong adversarial behavior. Experimental evaluations on MNIST, FMNIST, and CIFAR-10 demonstrate that our method significantly outperforms standard and robust federated learning baselines, such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum, in the majority of adversarial scenarios considered, including label flipping and Gaussian noise injection attacks. All experiments were conducted using the Flower federated learning framework.



## **50. Does simple trump complex? Comparing strategies for adversarial robustness in DNNs**

cs.LG

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18019v1) [paper-pdf](http://arxiv.org/pdf/2508.18019v1)

**Authors**: William Brooks, Marelie H. Davel, Coenraad Mouton

**Abstract**: Deep Neural Networks (DNNs) have shown substantial success in various applications but remain vulnerable to adversarial attacks. This study aims to identify and isolate the components of two different adversarial training techniques that contribute most to increased adversarial robustness, particularly through the lens of margins in the input space -- the minimal distance between data points and decision boundaries. Specifically, we compare two methods that maximize margins: a simple approach which modifies the loss function to increase an approximation of the margin, and a more complex state-of-the-art method (Dynamics-Aware Robust Training) which builds upon this approach. Using a VGG-16 model as our base, we systematically isolate and evaluate individual components from these methods to determine their relative impact on adversarial robustness. We assess the effect of each component on the model's performance under various adversarial attacks, including AutoAttack and Projected Gradient Descent (PGD). Our analysis on the CIFAR-10 dataset reveals which elements most effectively enhance adversarial robustness, providing insights for designing more robust DNNs.



