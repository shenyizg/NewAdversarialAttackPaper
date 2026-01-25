# General Generative Models - Base/Interpretation
**update at 2026-01-25 10:36:50**

Sorted by classifier confidence (high to low).

## **1. Attacks on Approximate Caches in Text-to-Image Diffusion Models**

cs.CR

Accepted by Usenix Security 2026

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2508.20424v3) [paper-pdf](https://arxiv.org/pdf/2508.20424v3)

**Confidence**: 0.95

**Authors**: Desen Sun, Shuncheng Jie, Sihang Liu

**Abstract**: Diffusion models are a powerful class of generative models that produce images and other content from user prompts, but they are computationally intensive. To mitigate this cost, recent academic and industry work has adopted approximate caching, which reuses intermediate states from similar prompts in a cache. While efficient, this optimization introduces new security risks by breaking isolation among users. This paper provides a comprehensive assessment of the security vulnerabilities introduced by approximate caching. First, we demonstrate a remote covert channel established with the approximate cache, where a sender injects prompts with special keywords into the cache system and a receiver can recover that even after days, to exchange information. Second, we introduce a prompt stealing attack using the approximate cache, where an attacker can recover existing cached prompts from hits. Finally, we introduce a poisoning attack that embeds the attacker's logos into the previously stolen prompt, leading to unexpected logo rendering for the requests that hit the poisoned cache prompts. These attacks are all performed remotely through the serving system, demonstrating severe security vulnerabilities in approximate caching. The code for this work is available.



## **2. A Two-Stage Globally-Diverse Adversarial Attack for Vision-Language Pre-training Models**

cs.CV

Accepted to ICASSP 2026

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2601.12304v1) [paper-pdf](https://arxiv.org/pdf/2601.12304v1)

**Confidence**: 0.95

**Authors**: Wutao Chen, Huaqin Zou, Chen Wan, Lifeng Huang

**Abstract**: Vision-language pre-training (VLP) models are vulnerable to adversarial examples, particularly in black-box scenarios. Existing multimodal attacks often suffer from limited perturbation diversity and unstable multi-stage pipelines. To address these challenges, we propose 2S-GDA, a two-stage globally-diverse attack framework. The proposed method first introduces textual perturbations through a globally-diverse strategy by combining candidate text expansion with globally-aware replacement. To enhance visual diversity, image-level perturbations are generated using multi-scale resizing and block-shuffle rotation. Extensive experiments on VLP models demonstrate that 2S-GDA consistently improves attack success rates over state-of-the-art methods, with gains of up to 11.17\% in black-box settings. Our framework is modular and can be easily combined with existing methods to further enhance adversarial transferability.



## **3. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

cs.CV

Accepted by ICCV-2025

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2406.05491v4) [paper-pdf](https://arxiv.org/pdf/2406.05491v4)

**Confidence**: 0.95

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Hao Wu, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these methods are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also vulnerable to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC) to achieve the attack. In light that the pivotal multimodal alignment is achieved through the advanced contrastive learning technique, we devise to turn this powerful weapon against themselves, i.e., employ a malicious version of contrastive learning to train the C-PGC based on our carefully crafted positive and negative image-text pairs for essentially destroying the alignment relationship learned by VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus essentially enhancing attacks across various victim models and V+L tasks. The GitHub repository is available at https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.



## **4. Towards Effective Prompt Stealing Attack against Text-to-Image Diffusion Models**

cs.CR

This paper proposes an effective training-free, proxy-in-the-loop, and search-based prompt-stealing scheme against T2I models

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2508.06837v2) [paper-pdf](https://arxiv.org/pdf/2508.06837v2)

**Confidence**: 0.95

**Authors**: Shiqian Zhao, Chong Wang, Yiming Li, Yihao Huang, Wenjie Qu, Siew-Kei Lam, Yi Xie, Kangjie Chen, Jie Zhang, Tianwei Zhang

**Abstract**: Text-to-Image (T2I) models, represented by DALL$\cdot$E and Midjourney, have gained huge popularity for creating realistic images. The quality of these images relies on the carefully engineered prompts, which have become valuable intellectual property. While skilled prompters showcase their AI-generated art on markets to attract buyers, this business incidentally exposes them to \textit{prompt stealing attacks}. Existing state-of-the-art attack techniques reconstruct the prompts from a fixed set of modifiers (i.e., style descriptions) with model-specific training, which exhibit restricted adaptability and effectiveness to diverse showcases (i.e., target images) and diffusion models.   To alleviate these limitations, we propose Prometheus, a training-free, proxy-in-the-loop, search-based prompt-stealing attack, which reverse-engineers the valuable prompts of the showcases by interacting with a local proxy model. It consists of three innovative designs. First, we introduce dynamic modifiers, as a supplement to static modifiers used in prior works. These dynamic modifiers provide more details specific to the showcases, and we exploit NLP analysis to generate them on the fly. Second, we design a contextual matching algorithm to sort both dynamic and static modifiers. This offline process helps reduce the search space of the subsequent step. Third, we interact with a local proxy model to invert the prompts with a greedy search algorithm. Based on the feedback guidance, we refine the prompt to achieve higher fidelity. The evaluation results show that Prometheus successfully extracts prompts from popular platforms like PromptBase and AIFrog against diverse victim models, including Midjourney, Leonardo.ai, and DALL$\cdot$E, with an ASR improvement of 25.0\%. We also validate that Prometheus is resistant to extensive potential defenses, further highlighting its severity in practice.



## **5. Inference Attacks Against Graph Generative Diffusion Models**

cs.LG

This work has been accepted by USENIX Security 2026

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.03701v1) [paper-pdf](https://arxiv.org/pdf/2601.03701v1)

**Confidence**: 0.95

**Authors**: Xiuling Wang, Xin Huang, Guibo Luo, Jianliang Xu

**Abstract**: Graph generative diffusion models have recently emerged as a powerful paradigm for generating complex graph structures, effectively capturing intricate dependencies and relationships within graph data. However, the privacy risks associated with these models remain largely unexplored. In this paper, we investigate information leakage in such models through three types of black-box inference attacks. First, we design a graph reconstruction attack, which can reconstruct graphs structurally similar to those training graphs from the generated graphs. Second, we propose a property inference attack to infer the properties of the training graphs, such as the average graph density and the distribution of densities, from the generated graphs. Third, we develop two membership inference attacks to determine whether a given graph is present in the training set. Extensive experiments on three different types of graph generative diffusion models and six real-world graphs demonstrate the effectiveness of these attacks, significantly outperforming the baseline approaches. Finally, we propose two defense mechanisms that mitigate these inference attacks and achieve a better trade-off between defense strength and target model utility than existing methods. Our code is available at https://zenodo.org/records/17946102.



## **6. T2VAttack: Adversarial Attack on Text-to-Video Diffusion Models**

cs.CV

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.23953v1) [paper-pdf](https://arxiv.org/pdf/2512.23953v1)

**Confidence**: 0.95

**Authors**: Changzhen Li, Yuecong Min, Jie Zhang, Zheng Yuan, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Text-to-Video (T2V) diffusion models has driven remarkable advancements in generating high-quality, temporally coherent videos from natural language descriptions. Despite these achievements, their vulnerability to adversarial attacks remains largely unexplored. In this paper, we introduce T2VAttack, a comprehensive study of adversarial attacks on T2V diffusion models from both semantic and temporal perspectives. Considering the inherently dynamic nature of video data, we propose two distinct attack objectives: a semantic objective to evaluate video-text alignment and a temporal objective to assess the temporal dynamics. To achieve an effective and efficient attack process, we propose two adversarial attack methods: (i) T2VAttack-S, which identifies semantically or temporally critical words in prompts and replaces them with synonyms via greedy search, and (ii) T2VAttack-I, which iteratively inserts optimized words with minimal perturbation to the prompt. By combining these objectives and strategies, we conduct a comprehensive evaluation on the adversarial robustness of several state-of-the-art T2V models, including ModelScope, CogVideoX, Open-Sora, and HunyuanVideo. Our experiments reveal that even minor prompt modifications, such as the substitution or insertion of a single word, can cause substantial degradation in semantic fidelity and temporal dynamics, highlighting critical vulnerabilities in current T2V diffusion models.



## **7. Data-Chain Backdoor: Do You Trust Diffusion Models as Generative Data Supplier?**

cs.CR

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.15769v1) [paper-pdf](https://arxiv.org/pdf/2512.15769v1)

**Confidence**: 0.95

**Authors**: Junchi Lu, Xinke Li, Yuheng Liu, Qi Alfred Chen

**Abstract**: The increasing use of generative models such as diffusion models for synthetic data augmentation has greatly reduced the cost of data collection and labeling in downstream perception tasks. However, this new data source paradigm may introduce important security concerns. This work investigates backdoor propagation in such emerging generative data supply chains, namely Data-Chain Backdoor (DCB). Specifically, we find that open-source diffusion models can become hidden carriers of backdoors. Their strong distribution-fitting ability causes them to memorize and reproduce backdoor triggers during generation, which are subsequently inherited by downstream models, resulting in severe security risks. This threat is particularly concerning under clean-label attack scenarios, as it remains effective while having negligible impact on the utility of the synthetic data. Furthermore, we discover an Early-Stage Trigger Manifestation (ESTM) phenomenon: backdoor trigger patterns tend to surface more explicitly in the early, high-noise stages of the diffusion model's reverse generation process before being subtly integrated into the final samples. Overall, this work reveals a previously underexplored threat in generative data pipelines and provides initial insights toward mitigating backdoor risks in synthetic data generation.



## **8. Membership and Dataset Inference Attacks on Large Audio Generative Models**

cs.LG

NeurIPS 2025 AI for Music Workshop NeurIPS 2025 Workshop on Creativity & Generative AI

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09654v1) [paper-pdf](https://arxiv.org/pdf/2512.09654v1)

**Confidence**: 0.95

**Authors**: Jakub Proboszcz, Paweł Kochanski, Karol Korszun, Donato Crisostomi, Giorgio Strano, Emanuele Rodolà, Kamil Deja, Jan Dubinski

**Abstract**: Generative audio models, based on diffusion and autoregressive architectures, have advanced rapidly in both quality and expressiveness. This progress, however, raises pressing copyright concerns, as such models are often trained on vast corpora of artistic and commercial works. A central question is whether one can reliably verify if an artist's material was included in training, thereby providing a means for copyright holders to protect their content. In this work, we investigate the feasibility of such verification through membership inference attacks (MIA) on open-source generative audio models, which attempt to determine whether a specific audio sample was part of the training set. Our empirical results show that membership inference alone is of limited effectiveness at scale, as the per-sample membership signal is weak for models trained on large and diverse datasets. However, artists and media owners typically hold collections of works rather than isolated samples. Building on prior work in text and vision domains, in this work we focus on dataset inference (DI), which aggregates diverse membership evidence across multiple samples. We find that DI is successful in the audio domain, offering a more practical mechanism for assessing whether an artist's works contributed to model training. Our results suggest DI as a promising direction for copyright protection and dataset accountability in the era of large audio generative models.



## **9. Towards Irreversible Machine Unlearning for Diffusion Models**

cs.LG

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03564v1) [paper-pdf](https://arxiv.org/pdf/2512.03564v1)

**Confidence**: 0.95

**Authors**: Xun Yuan, Zilong Zhao, Jiayu Li, Aryan Pasikhani, Prosanta Gope, Biplab Sikdar

**Abstract**: Diffusion models are renowned for their state-of-the-art performance in generating synthetic images. However, concerns related to safety, privacy, and copyright highlight the need for machine unlearning, which can make diffusion models forget specific training data and prevent the generation of sensitive or unwanted content. Current machine unlearning methods for diffusion models are primarily designed for conditional diffusion models and focus on unlearning specific data classes or features. Among these methods, finetuning-based machine unlearning methods are recognized for their efficiency and effectiveness, which update the parameters of pre-trained diffusion models by minimizing carefully designed loss functions. However, in this paper, we propose a novel attack named Diffusion Model Relearning Attack (DiMRA), which can reverse the finetuning-based machine unlearning methods, posing a significant vulnerability of this kind of technique. Without prior knowledge of the unlearning elements, DiMRA optimizes the unlearned diffusion model on an auxiliary dataset to reverse the unlearning, enabling the model to regenerate previously unlearned elements. To mitigate this vulnerability, we propose a novel machine unlearning method for diffusion models, termed as Diffusion Model Unlearning by Memorization (DiMUM). Unlike traditional methods that focus on forgetting, DiMUM memorizes alternative data or features to replace targeted unlearning data or features in order to prevent generating such elements. In our experiments, we demonstrate the effectiveness of DiMRA in reversing state-of-the-art finetuning-based machine unlearning methods for diffusion models, highlighting the need for more robust solutions. We extensively evaluate DiMUM, demonstrating its superior ability to preserve the generative performance of diffusion models while enhancing robustness against DiMRA.



## **10. BadBlocks: Lightweight and Stealthy Backdoor Threat in Text-to-Image Diffusion Models**

cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2508.03221v4) [paper-pdf](https://arxiv.org/pdf/2508.03221v4)

**Confidence**: 0.95

**Authors**: Yu Pan, Jiahao Chen, Wenjie Wang, Bingrong Dai, Junjun Yang

**Abstract**: Diffusion models have recently achieved remarkable success in image generation, yet growing evidence shows their vulnerability to backdoor attacks, where adversaries implant covert triggers to manipulate outputs. While existing defenses can detect many such attacks via visual inspection and neural network-based analysis, we identify a more lightweight and stealthy threat, termed BadBlocks. BadBlocks selectively contaminates specific blocks within the UNet architecture while preserving the normal behavior of the remaining components. Compared with prior methods, it requires only about 30% of the computation and 20% of the GPU time, yet achieves high attack success rates with minimal perceptual degradation. Extensive experiments demonstrate that BadBlocks can effectively evade state-of-the-art defenses, particularly attention-based detection frameworks. Ablation studies further reveal that effective backdoor injection does not require fine-tuning the entire network and highlight the critical role of certain layers in backdoor mapping. Overall, BadBlocks substantially lowers the barrier for backdooring large-scale diffusion models, even on consumer-grade GPUs.



## **11. Low Resource Reconstruction Attacks Through Benign Prompts**

cs.LG

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2507.07947v3) [paper-pdf](https://arxiv.org/pdf/2507.07947v3)

**Confidence**: 0.95

**Authors**: Sol Yarkoni, Mahmood Sharif, Roi Livni

**Abstract**: Recent advances in generative models, such as diffusion models, have raised concerns related to privacy, copyright infringement, and data stewardship. To better understand and control these risks, prior work has introduced techniques and attacks that reconstruct images, or parts of images, from training data. While these results demonstrate that training data can be recovered, existing methods often rely on high computational resources, partial access to the training set, or carefully engineered prompts.   In this work, we present a new attack that requires low resources, assumes little to no access to the training data, and identifies seemingly benign prompts that can lead to potentially risky image reconstruction. We further show that such reconstructions may occur unintentionally, even for users without specialized knowledge. For example, we observe that for one existing model, the prompt ``blue Unisex T-Shirt'' generates the face of a real individual. Moreover, by combining the identified vulnerabilities with real-world prompt data, we discover prompts that reproduce memorized visual elements.   Our approach builds on insights from prior work and leverages domain knowledge to expose a fundamental vulnerability arising from the use of scraped e-commerce data, where templated layouts and images are closely tied to pattern-like textual prompts.   The code for our attack is publicly available at https://github.com/TheSolY/lr-tmi.



## **12. Towards Dataset Copyright Evasion Attack against Personalized Text-to-Image Diffusion Models**

cs.CV

Accepted by IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2505.02824v2) [paper-pdf](https://arxiv.org/pdf/2505.02824v2)

**Confidence**: 0.95

**Authors**: Kuofeng Gao, Yufei Zhu, Yiming Li, Jiawang Bai, Yong Yang, Zhifeng Li, Shu-Tao Xia

**Abstract**: Text-to-image (T2I) diffusion models enable high-quality image generation conditioned on textual prompts. However, fine-tuning these pre-trained models for personalization raises concerns about unauthorized dataset usage. To address this issue, dataset ownership verification (DOV) has recently been proposed, which embeds watermarks into fine-tuning datasets via backdoor techniques. These watermarks remain dormant on benign samples but produce owner-specified outputs when triggered. Despite its promise, the robustness of DOV against copyright evasion attacks (CEA) remains unexplored. In this paper, we investigate how adversaries can circumvent these mechanisms, enabling models trained on watermarked datasets to bypass ownership verification. We begin by analyzing the limitations of potential attacks achieved by backdoor removal, including TPD and T2IShield. In practice, TPD suffers from inconsistent effectiveness due to randomness, while T2IShield fails when watermarks are embedded as local image patches. To this end, we introduce CEAT2I, the first CEA specifically targeting DOV in T2I diffusion models. CEAT2I consists of three stages: (1) motivated by the observation that T2I models converge faster on watermarked samples with respect to intermediate features rather than training loss, we reliably detect watermarked samples; (2) we iteratively ablate tokens from the prompts of detected samples and monitor feature shifts to identify trigger tokens; and (3) we apply a closed-form concept erasure method to remove the injected watermarks. Extensive experiments demonstrate that CEAT2I effectively evades state-of-the-art DOV mechanisms while preserving model performance. The code is available at https://github.com/csyufei/CEAT2I.



## **13. Dynamic Attention Analysis for Backdoor Detection in Text-to-Image Diffusion Models**

cs.CV

Accepted by TPAMI

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2504.20518v3) [paper-pdf](https://arxiv.org/pdf/2504.20518v3)

**Confidence**: 0.95

**Authors**: Zhongqi Wang, Jie Zhang, Shiguang Shan, Xilin Chen

**Abstract**: Recent studies have revealed that text-to-image diffusion models are vulnerable to backdoor attacks, where attackers implant stealthy textual triggers to manipulate model outputs. Previous backdoor detection methods primarily focus on the static features of backdoor samples. However, a vital property of diffusion models is their inherent dynamism. This study introduces a novel backdoor detection perspective named Dynamic Attention Analysis (DAA), showing that these dynamic characteristics serve as better indicators for backdoor detection. Specifically, by examining the dynamic evolution of cross-attention maps, we observe that backdoor samples exhibit distinct feature evolution patterns at the $<$EOS$>$ token compared to benign samples. To quantify these dynamic anomalies, we first introduce DAA-I, which treats the tokens' attention maps as spatially independent and measures dynamic feature using the Frobenius norm. Furthermore, to better capture the interactions between attention maps and refine the feature, we propose a dynamical system-based approach, referred to as DAA-S. This model formulates the spatial correlations among attention maps using a graph-based state equation and we theoretically analyze the global asymptotic stability of this method. Extensive experiments across six representative backdoor attack scenarios demonstrate that our approach significantly surpasses existing detection methods, achieving an average F1 Score of 79.27% and an AUC of 86.27%. The code is available at https://github.com/Robin-WZQ/DAA.



## **14. Targeted Data Protection for Diffusion Model by Matching Training Trajectory**

cs.AI

AAAI 2026

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10433v1) [paper-pdf](https://arxiv.org/pdf/2512.10433v1)

**Confidence**: 0.95

**Authors**: Hojun Lee, Mijin Koo, Yeji Song, Nojun Kwak

**Abstract**: Recent advancements in diffusion models have made fine-tuning text-to-image models for personalization increasingly accessible, but have also raised significant concerns regarding unauthorized data usage and privacy infringement. Current protection methods are limited to passively degrading image quality, failing to achieve stable control. While Targeted Data Protection (TDP) offers a promising paradigm for active redirection toward user-specified target concepts, existing TDP attempts suffer from poor controllability due to snapshot-matching approaches that fail to account for complete learning dynamics. We introduce TAFAP (Trajectory Alignment via Fine-tuning with Adversarial Perturbations), the first method to successfully achieve effective TDP by controlling the entire training trajectory. Unlike snapshot-based methods whose protective influence is easily diluted as training progresses, TAFAP employs trajectory-matching inspired by dataset distillation to enforce persistent, verifiable transformations throughout fine-tuning. We validate our method through extensive experiments, demonstrating the first successful targeted transformation in diffusion models with simultaneous control over both identity and visual patterns. TAFAP significantly outperforms existing TDP attempts, achieving robust redirection toward target concepts while maintaining high image quality. This work enables verifiable safeguards and provides a new framework for controlling and tracing alterations in diffusion model outputs.



## **15. Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation**

cs.SD

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2507.17937v3) [paper-pdf](https://arxiv.org/pdf/2507.17937v3)

**Confidence**: 0.95

**Authors**: Jaechul Roh, Zachary Novack, Yuefeng Peng, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Amir Houmansadr

**Abstract**: Generative AI systems for music and video commonly use text-based filters to prevent the regurgitation of copyrighted material. We expose a fundamental flaw in this approach by introducing Adversarial PhoneTic Prompting (APT), a novel attack that bypasses these safeguards by exploiting phonetic memorization. The APT attack replaces iconic lyrics with homophonic but semantically unrelated alternatives (e.g., "mom's spaghetti" becomes "Bob's confetti"), preserving acoustic structure while altering meaning; we identify high-fidelity phonetic matches using CMU pronouncing dictionary. We demonstrate that leading Lyrics-to-Song (L2S) models like SUNO and YuE regenerate songs with striking melodic and rhythmic similarity to their copyrighted originals when prompted with these altered lyrics. More surprisingly, this vulnerability extends across modalities. When prompted with phonetically modified lyrics from a song, a Text-to-Video (T2V) model like Veo 3 reconstructs visual scenes from the original music video-including specific settings and character archetypes-despite the absence of any visual cues in the prompt. Our findings reveal that models memorize deep, structural patterns tied to acoustics, not just verbatim text. This phonetic-to-visual leakage represents a critical vulnerability in transcript-conditioned generative models, rendering simple copyright filters ineffective and raising urgent concerns about the secure deployment of multimodal AI systems. Demo examples are available at our project page (https://jrohsc.github.io/music_attack/).



## **16. Adversarial Attacks and Robust Defenses in Speaker Embedding based Zero-Shot Text-to-Speech System**

eess.AS

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2410.04017v2) [paper-pdf](https://arxiv.org/pdf/2410.04017v2)

**Confidence**: 0.95

**Authors**: Ze Li, Yao Shi, Yunfei Xu, Ming Li

**Abstract**: Speaker embedding based zero-shot Text-to-Speech (TTS) systems enable high-quality speech synthesis for unseen speakers using minimal data. However, these systems are vulnerable to adversarial attacks, where an attacker introduces imperceptible perturbations to the original speaker's audio waveform, leading to synthesized speech sounds like another person. This vulnerability poses significant security risks, including speaker identity spoofing and unauthorized voice manipulation. This paper investigates two primary defense strategies to address these threats: adversarial training and adversarial purification. Adversarial training enhances the model's robustness by integrating adversarial examples during the training process, thereby improving resistance to such attacks. Adversarial purification, on the other hand, employs diffusion probabilistic models to revert adversarially perturbed audio to its clean form. Experimental results demonstrate that these defense mechanisms can significantly reduce the impact of adversarial perturbations, enhancing the security and reliability of speaker embedding based zero-shot TTS systems in adversarial environments.



## **17. Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning**

cs.CV

Accepted to AAAI 2026

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2511.12735v2) [paper-pdf](https://arxiv.org/pdf/2511.12735v2)

**Confidence**: 0.85

**Authors**: Ankita Raj, Chetan Arora

**Abstract**: Open-vocabulary object detectors (OVODs) unify vision and language to detect arbitrary object categories based on text prompts, enabling strong zero-shot generalization to novel concepts. As these models gain traction in high-stakes applications such as robotics, autonomous driving, and surveillance, understanding their security risks becomes crucial. In this work, we conduct the first study of backdoor attacks on OVODs and reveal a new attack surface introduced by prompt tuning. We propose TrAP (Trigger-Aware Prompt tuning), a multi-modal backdoor injection strategy that jointly optimizes prompt parameters in both image and text modalities along with visual triggers. TrAP enables the attacker to implant malicious behavior using lightweight, learnable prompt tokens without retraining the base model weights, thus preserving generalization while embedding a hidden backdoor. We adopt a curriculum-based training strategy that progressively shrinks the trigger size, enabling effective backdoor activation using small trigger patches at inference. Experiments across multiple datasets show that TrAP achieves high attack success rates for both object misclassification and object disappearance attacks, while also improving clean image performance on downstream datasets compared to the zero-shot setting. Code: https://github.com/rajankita/TrAP



## **18. RAVEN: Erasing Invisible Watermarks via Novel View Synthesis**

cs.CV

13 pages

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08832v1) [paper-pdf](https://arxiv.org/pdf/2601.08832v1)

**Confidence**: 0.85

**Authors**: Fahad Shamshad, Nils Lukas, Karthik Nandakumar

**Abstract**: Invisible watermarking has become a critical mechanism for authenticating AI-generated image content, with major platforms deploying watermarking schemes at scale. However, evaluating the vulnerability of these schemes against sophisticated removal attacks remains essential to assess their reliability and guide robust design. In this work, we expose a fundamental vulnerability in invisible watermarks by reformulating watermark removal as a view synthesis problem. Our key insight is that generating a perceptually consistent alternative view of the same semantic content, akin to re-observing a scene from a shifted perspective, naturally removes the embedded watermark while preserving visual fidelity. This reveals a critical gap: watermarks robust to pixel-space and frequency-domain attacks remain vulnerable to semantic-preserving viewpoint transformations. We introduce a zero-shot diffusion-based framework that applies controlled geometric transformations in latent space, augmented with view-guided correspondence attention to maintain structural consistency during reconstruction. Operating on frozen pre-trained models without detector access or watermark knowledge, our method achieves state-of-the-art watermark suppression across 15 watermarking methods--outperforming 14 baseline attacks while maintaining superior perceptual quality across multiple datasets.



## **19. Prototypicality Bias Reveals Blindspots in Multimodal Evaluation Metrics**

cs.CV

First version

**SubmitDate**: 2026-01-10    [abs](http://arxiv.org/abs/2601.04946v2) [paper-pdf](https://arxiv.org/pdf/2601.04946v2)

**Confidence**: 0.85

**Authors**: Subhadeep Roy, Gagan Bhatia, Steffen Eger

**Abstract**: Automatic metrics are now central to evaluating text-to-image models, often substituting for human judgment in benchmarking and large-scale filtering. However, it remains unclear whether these metrics truly prioritize semantic correctness or instead favor visually and socially prototypical images learned from biased data distributions. We identify and study prototypicality bias as a systematic failure mode in multimodal evaluation. We introduce a controlled contrastive benchmark ProtoBias (Prototypical Bias), spanning Animals, Objects, and Demography images, where semantically correct but non-prototypical images are paired with subtly incorrect yet prototypical adversarial counterparts. This setup enables a directional evaluation of whether metrics follow textual semantics or default to prototypes. Our results show that widely used metrics, including CLIPScore, PickScore, and VQA-based scores, frequently misrank these pairs, while even LLM-as-Judge systems exhibit uneven robustness in socially grounded cases. Human evaluations consistently favour semantic correctness with larger decision margins. Motivated by these findings, we propose ProtoScore, a robust 7B-parameter metric that substantially reduces failure rates and suppresses misranking, while running at orders of magnitude faster than the inference time of GPT-5, approaching the robustness of much larger closed-source judges.



## **20. Cryptanalysis of Pseudorandom Error-Correcting Codes**

cs.CR

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17310v1) [paper-pdf](https://arxiv.org/pdf/2512.17310v1)

**Confidence**: 0.85

**Authors**: Tianrui Wang, Anyu Wang, Tianshuo Cong, Delong Ran, Jinyuan Liu, Xiaoyun Wang

**Abstract**: Pseudorandom error-correcting codes (PRC) is a novel cryptographic primitive proposed at CRYPTO 2024. Due to the dual capability of pseudorandomness and error correction, PRC has been recognized as a promising foundational component for watermarking AI-generated content. However, the security of PRC has not been thoroughly analyzed, especially with concrete parameters or even in the face of cryptographic attacks. To fill this gap, we present the first cryptanalysis of PRC. We first propose three attacks to challenge the undetectability and robustness assumptions of PRC. Among them, two attacks aim to distinguish PRC-based codewords from plain vectors, and one attack aims to compromise the decoding process of PRC. Our attacks successfully undermine the claimed security guarantees across all parameter configurations. Notably, our attack can detect the presence of a watermark with overwhelming probability at a cost of $2^{22}$ operations. We also validate our approach by attacking real-world large generative models such as DeepSeek and Stable Diffusion. To mitigate our attacks, we further propose three defenses to enhance the security of PRC, including parameter suggestions, implementation suggestions, and constructing a revised key generation algorithm. Our proposed revised key generation function effectively prevents the occurrence of weak keys. However, we highlight that the current PRC-based watermarking scheme still cannot achieve a 128-bit security under our parameter suggestions due to the inherent configurations of large generative models, such as the maximum output length of large language models.



## **21. Optimization-Guided Diffusion for Interactive Scene Generation**

cs.CV

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.07661v2) [paper-pdf](https://arxiv.org/pdf/2512.07661v2)

**Confidence**: 0.85

**Authors**: Shihao Li, Naisheng Ye, Tianyu Li, Kashyap Chitta, Tuo An, Peng Su, Boyang Wang, Haiou Liu, Chen Lv, Hongyang Li

**Abstract**: Realistic and diverse multi-agent driving scenes are crucial for evaluating autonomous vehicles, but safety-critical events which are essential for this task are rare and underrepresented in driving datasets. Data-driven scene generation offers a low-cost alternative by synthesizing complex traffic behaviors from existing driving logs. However, existing models often lack controllability or yield samples that violate physical or social constraints, limiting their usability. We present OMEGA, an optimization-guided, training-free framework that enforces structural consistency and interaction awareness during diffusion-based sampling from a scene generation model. OMEGA re-anchors each reverse diffusion step via constrained optimization, steering the generation towards physically plausible and behaviorally coherent trajectories. Building on this framework, we formulate ego-attacker interactions as a game-theoretic optimization in the distribution space, approximating Nash equilibria to generate realistic, safety-critical adversarial scenarios. Experiments on nuPlan and Waymo show that OMEGA improves generation realism, consistency, and controllability, increasing the ratio of physically and behaviorally valid scenes from 32.35% to 72.27% for free exploration capabilities, and from 11% to 80% for controllability-focused generation. Our approach can also generate $5\times$ more near-collision frames with a time-to-collision under three seconds while maintaining the overall scene realism.



## **22. Rethinking Security in Semantic Communication: Latent Manipulation as a New Threat**

cs.CR

8 pages, 6 figures

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03361v1) [paper-pdf](https://arxiv.org/pdf/2512.03361v1)

**Confidence**: 0.85

**Authors**: Zhiyuan Xi, Kun Zhu

**Abstract**: Deep learning-based semantic communication (SemCom) has emerged as a promising paradigm for next-generation wireless networks, offering superior transmission efficiency by extracting and conveying task-relevant semantic latent representations rather than raw data. However, the openness of the wireless medium and the intrinsic vulnerability of semantic latent representations expose such systems to previously unrecognized security risks. In this paper, we uncover a fundamental latent-space vulnerability that enables Man-in-the-Middle (MitM) attacker to covertly manipulate the transmitted semantics while preserving the statistical properties of the transmitted latent representations. We first present a Diffusion-based Re-encoding Attack (DiR), wherein the attacker employs a diffusion model to synthesize an attacker-designed semantic variant, and re-encodes it into a valid latent representation compatible with the SemCom decoder. Beyond this model-dependent pathway, we further propose a model-agnostic and training-free Test-Time Adaptation Latent Manipulation attack (TTA-LM), in which the attacker perturbs and steers the intercepted latent representation toward an attacker-specified semantic target by leveraging the gradient of a target loss function. In contrast to diffusion-based manipulation, TTA-LM does not rely on any generative model and does not impose modality-specific or task-specific assumptions, thereby enabling efficient and broadly applicable latent-space tampering across diverse SemCom architectures. Extensive experiments on representative semantic communication architectures demonstrate that both attacks can significantly alter the decoded semantics while preserving natural latent-space distributions, making the attacks covert and difficult to detect.



## **23. Counterfeit Answers: Adversarial Forgery against OCR-Free Document Visual Question Answering**

cs.CV

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04554v1) [paper-pdf](https://arxiv.org/pdf/2512.04554v1)

**Confidence**: 0.85

**Authors**: Marco Pintore, Maura Pintor, Dimosthenis Karatzas, Battista Biggio

**Abstract**: Document Visual Question Answering (DocVQA) enables end-to-end reasoning grounded on information present in a document input. While recent models have shown impressive capabilities, they remain vulnerable to adversarial attacks. In this work, we introduce a novel attack scenario that aims to forge document content in a visually imperceptible yet semantically targeted manner, allowing an adversary to induce specific or generally incorrect answers from a DocVQA model. We develop specialized attack algorithms that can produce adversarially forged documents tailored to different attackers' goals, ranging from targeted misinformation to systematic model failure scenarios. We demonstrate the effectiveness of our approach against two end-to-end state-of-the-art models: Pix2Struct, a vision-language transformer that jointly processes image and text through sequence-to-sequence modeling, and Donut, a transformer-based model that directly extracts text and answers questions from document images. Our findings highlight critical vulnerabilities in current DocVQA systems and call for the development of more robust defenses.



## **24. Exploiting Leaderboards for Large-Scale Distribution of Malicious Models**

cs.LG

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08983v1) [paper-pdf](https://arxiv.org/pdf/2507.08983v1)

**Confidence**: 0.85

**Authors**: Anshuman Suri, Harsh Chaudhari, Yuefeng Peng, Ali Naseh, Amir Houmansadr, Alina Oprea

**Abstract**: While poisoning attacks on machine learning models have been extensively studied, the mechanisms by which adversaries can distribute poisoned models at scale remain largely unexplored. In this paper, we shed light on how model leaderboards -- ranked platforms for model discovery and evaluation -- can serve as a powerful channel for adversaries for stealthy large-scale distribution of poisoned models. We present TrojanClimb, a general framework that enables injection of malicious behaviors while maintaining competitive leaderboard performance. We demonstrate its effectiveness across four diverse modalities: text-embedding, text-generation, text-to-speech and text-to-image, showing that adversaries can successfully achieve high leaderboard rankings while embedding arbitrary harmful functionalities, from backdoors to bias injection. Our findings reveal a significant vulnerability in the machine learning ecosystem, highlighting the urgent need to redesign leaderboard evaluation mechanisms to detect and filter malicious (e.g., poisoned) models, while exposing broader security implications for the machine learning community regarding the risks of adopting models from unverified sources.



## **25. Multi-Faceted Multimodal Monosemanticity**

cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2502.14888v3) [paper-pdf](https://arxiv.org/pdf/2502.14888v3)

**Confidence**: 0.85

**Authors**: Hanqi Yan, Xiangxiang Cui, Lu Yin, Paul Pu Liang, Yulan He, Yifei Wang

**Abstract**: Humans experience the world through multiple modalities, such as, vision, language, and speech, making it natural to explore the commonality and distinctions among them. In this work, we take a data-driven approach to address this question by analyzing interpretable, monosemantic features extracted from deep multimodal models. Specifically, we investigate CLIP, a prominent visual-language representation model trained on massive image-text pairs. Building on prior research in single-modal interpretability, we develop a set of multi-modal interpretability tools and measures designed to disentangle and analyze features learned from CLIP. Specifically, we introduce the Modality Dominance Score (MDS) to attribute each CLIP feature to a specific modality. We then map CLIP features into a more interpretable space, enabling us to categorize them into three distinct classes: vision features (single-modal), language features (single-modal), and visual-language features (cross-modal). Interestingly, this data-driven categorization closely aligns with human intuitive understandings of different modalities. We further show that this modality decomposition can benefit multiple downstream tasks, including reducing bias in gender detection, generating cross-modal adversarial examples, and enabling modal-specific feature control in text-to-image generation. These results indicate that large-scale multimodal models, when equipped with task-agnostic interpretability tools, can offer valuable insights into the relationships between different data modalities.



