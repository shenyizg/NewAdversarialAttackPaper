# Latest Adversarial Attack Papers
**update at 2025-05-24 16:13:15**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. When Are Concepts Erased From Diffusion Models?**

cs.LG

Project Page:  https://nyu-dice-lab.github.io/when-are-concepts-erased/

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17013v1) [paper-pdf](http://arxiv.org/pdf/2505.17013v1)

**Authors**: Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, Niv Cohen

**Abstract**: Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.



## **2. Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability**

cs.CV

15 pages. 7 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2504.10804v2) [paper-pdf](http://arxiv.org/pdf/2504.10804v2)

**Authors**: Jiani Liu, Zhiyuan Wang, Zeliang Zhang, Chao Huang, Susan Liang, Yunlong Tang, Chenliang Xu

**Abstract**: Vision Transformers (ViTs) have demonstrated impressive performance across a range of applications, including many safety-critical tasks. However, their unique architectural properties raise new challenges and opportunities in adversarial robustness. In particular, we observe that adversarial examples crafted on ViTs exhibit higher transferability compared to those crafted on CNNs, suggesting that ViTs contain structural characteristics favorable for transferable attacks. In this work, we investigate the role of computational redundancy in ViTs and its impact on adversarial transferability. Unlike prior studies that aim to reduce computation for efficiency, we propose to exploit this redundancy to improve the quality and transferability of adversarial examples. Through a detailed analysis, we identify two forms of redundancy, including the data-level and model-level, that can be harnessed to amplify attack effectiveness. Building on this insight, we design a suite of techniques, including attention sparsity manipulation, attention head permutation, clean token regularization, ghost MoE diversification, and test-time adversarial training. Extensive experiments on the ImageNet-1k dataset validate the effectiveness of our approach, showing that our methods significantly outperform existing baselines in both transferability and generality across diverse model architectures.



## **3. Invisible Prompts, Visible Threats: Malicious Font Injection in External Resources for Large Language Models**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16957v1) [paper-pdf](http://arxiv.org/pdf/2505.16957v1)

**Authors**: Junjie Xiong, Changjia Zhu, Shuhang Lin, Chong Zhang, Yongfeng Zhang, Yao Liu, Lingyao Li

**Abstract**: Large Language Models (LLMs) are increasingly equipped with capabilities of real-time web search and integrated with protocols like Model Context Protocol (MCP). This extension could introduce new security vulnerabilities. We present a systematic investigation of LLM vulnerabilities to hidden adversarial prompts through malicious font injection in external resources like webpages, where attackers manipulate code-to-glyph mapping to inject deceptive content which are invisible to users. We evaluate two critical attack scenarios: (1) "malicious content relay" and (2) "sensitive data leakage" through MCP-enabled tools. Our experiments reveal that indirect prompts with injected malicious font can bypass LLM safety mechanisms through external resources, achieving varying success rates based on data sensitivity and prompt design. Our research underscores the urgent need for enhanced security measures in LLM deployments when processing external content.



## **4. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16947v1) [paper-pdf](http://arxiv.org/pdf/2505.16947v1)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.



## **5. CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16888v1) [paper-pdf](http://arxiv.org/pdf/2505.16888v1)

**Authors**: Viet Pham, Thai Le

**Abstract**: Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.



## **6. Safe RLHF-V: Safe Reinforcement Learning from Multi-modal Human Feedback**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2503.17682v2) [paper-pdf](http://arxiv.org/pdf/2503.17682v2)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Conghui Zhang, Han Zhu, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Yida Tang, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are essential for building general-purpose AI assistants; however, they pose increasing safety risks. How can we ensure safety alignment of MLLMs to prevent undesired behaviors? Going further, it is critical to explore how to fine-tune MLLMs to preserve capabilities while meeting safety constraints. Fundamentally, this challenge can be formulated as a min-max optimization problem. However, existing datasets have not yet disentangled single preference signals into explicit safety constraints, hindering systematic investigation in this direction. Moreover, it remains an open question whether such constraints can be effectively incorporated into the optimization process for multi-modal models. In this work, we present the first exploration of the Safe RLHF-V -- the first multimodal safety alignment framework. The framework consists of: $\mathbf{(I)}$ BeaverTails-V, the first open-source dataset featuring dual preference annotations for helpfulness and safety, supplemented with multi-level safety labels (minor, moderate, severe); $\mathbf{(II)}$ Beaver-Guard-V, a multi-level guardrail system to proactively defend against unsafe queries and adversarial attacks. Applying the guard model over five rounds of filtering and regeneration significantly enhances the precursor model's overall safety by an average of 40.9%. $\mathbf{(III)}$ Based on dual preference, we initiate the first exploration of multi-modal safety alignment within a constrained optimization. Experimental results demonstrate that Safe RLHF effectively improves both model helpfulness and safety. Specifically, Safe RLHF-V enhances model safety by 34.2% and helpfulness by 34.3%.



## **7. Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability**

cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16789v1) [paper-pdf](http://arxiv.org/pdf/2505.16789v1)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models gain popularity, their vulnerability to adversarial attacks remains a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Misalignment, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity within our experimental datasets. We then evaluate the adversarial performance of these fine-tuned models and assess how dataset factors correlate with attack success rates. Lastly, we explore potential causal links, offering new insights into adversarial defense strategies and highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_misalignment.



## **8. Experimental robustness benchmark of quantum neural network on a superconducting quantum processor**

quant-ph

There are 8 pages with 5 figures in the main text and 15 pages with  14 figures in the supplementary information

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16714v1) [paper-pdf](http://arxiv.org/pdf/2505.16714v1)

**Authors**: Hai-Feng Zhang, Zhao-Yun Chen, Peng Wang, Liang-Liang Guo, Tian-Le Wang, Xiao-Yan Yang, Ren-Ze Zhao, Ze-An Zhao, Sheng Zhang, Lei Du, Hao-Ran Tao, Zhi-Long Jia, Wei-Cheng Kong, Huan-Yu Liu, Athanasios V. Vasilakos, Yang Yang, Yu-Chun Wu, Ji Guan, Peng Duan, Guo-Ping Guo

**Abstract**: Quantum machine learning (QML) models, like their classical counterparts, are vulnerable to adversarial attacks, hindering their secure deployment. Here, we report the first systematic experimental robustness benchmark for 20-qubit quantum neural network (QNN) classifiers executed on a superconducting processor. Our benchmarking framework features an efficient adversarial attack algorithm designed for QNNs, enabling quantitative characterization of adversarial robustness and robustness bounds. From our analysis, we verify that adversarial training reduces sensitivity to targeted perturbations by regularizing input gradients, significantly enhancing QNN's robustness. Additionally, our analysis reveals that QNNs exhibit superior adversarial robustness compared to classical neural networks, an advantage attributed to inherent quantum noise. Furthermore, the empirical upper bound extracted from our attack experiments shows a minimal deviation ($3 \times 10^{-3}$) from the theoretical lower bound, providing strong experimental confirmation of the attack's effectiveness and the tightness of fidelity-based robustness bounds. This work establishes a critical experimental framework for assessing and improving quantum adversarial robustness, paving the way for secure and reliable QML applications.



## **9. BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization**

cs.CR

19 pages, 12 figures, 6 tables

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16640v1) [paper-pdf](http://arxiv.org/pdf/2505.16640v1)

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Hechang Wang, Pan Zhou, Lichao Sun

**Abstract**: Vision-Language-Action (VLA) models have advanced robotic control by enabling end-to-end decision-making directly from multimodal inputs. However, their tightly coupled architectures expose novel security vulnerabilities. Unlike traditional adversarial perturbations, backdoor attacks represent a stealthier, persistent, and practically significant threat-particularly under the emerging Training-as-a-Service paradigm-but remain largely unexplored in the context of VLA models. To address this gap, we propose BadVLA, a backdoor attack method based on Objective-Decoupled Optimization, which for the first time exposes the backdoor vulnerabilities of VLA models. Specifically, it consists of a two-stage process: (1) explicit feature-space separation to isolate trigger representations from benign inputs, and (2) conditional control deviations that activate only in the presence of the trigger, while preserving clean-task performance. Empirical results on multiple VLA benchmarks demonstrate that BadVLA consistently achieves near-100% attack success rates with minimal impact on clean task accuracy. Further analyses confirm its robustness against common input perturbations, task transfers, and model fine-tuning, underscoring critical security vulnerabilities in current VLA deployments. Our work offers the first systematic investigation of backdoor vulnerabilities in VLA models, highlighting an urgent need for secure and trustworthy embodied model design practices. We have released the project page at https://badvla-project.github.io/.



## **10. Finetuning-Activated Backdoors in LLMs**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16567v1) [paper-pdf](http://arxiv.org/pdf/2505.16567v1)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs.



## **11. On the Lack of Robustness of Binary Function Similarity Systems**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.04163v2) [paper-pdf](http://arxiv.org/pdf/2412.04163v2)

**Authors**: Gianluca Capozzi, Tong Tang, Jie Wan, Ziqi Yang, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Lorenzo Cavallaro, Leonardo Querzoni

**Abstract**: Binary function similarity, which often relies on learning-based algorithms to identify what functions in a pool are most similar to a given query function, is a sought-after topic in different communities, including machine learning, software engineering, and security. Its importance stems from the impact it has in facilitating several crucial tasks, from reverse engineering and malware analysis to automated vulnerability detection. Whereas recent work cast light around performance on this long-studied problem, the research landscape remains largely lackluster in understanding the resiliency of the state-of-the-art machine learning models against adversarial attacks. As security requires to reason about adversaries, in this work we assess the robustness of such models through a simple yet effective black-box greedy attack, which modifies the topology and the content of the control flow of the attacked functions. We demonstrate that this attack is successful in compromising all the models, achieving average attack success rates of 57.06% and 95.81% depending on the problem settings (targeted and untargeted attacks). Our findings are insightful: top performance on clean data does not necessarily relate to top robustness properties, which explicitly highlights performance-robustness trade-offs one should consider when deploying such models, calling for further research.



## **12. Implicit Jailbreak Attacks via Cross-Modal Information Concealment on Vision-Language Models**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16446v1) [paper-pdf](http://arxiv.org/pdf/2505.16446v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities. However, the expanded input space introduces new attack surfaces. Previous jailbreak attacks often inject malicious instructions from text into less aligned modalities, such as vision. As MLLMs increasingly incorporate cross-modal consistency and alignment mechanisms, such explicit attacks become easier to detect and block. In this work, we propose a novel implicit jailbreak framework termed IJA that stealthily embeds malicious instructions into images via least significant bit steganography and couples them with seemingly benign, image-related textual prompts. To further enhance attack effectiveness across diverse MLLMs, we incorporate adversarial suffixes generated by a surrogate model and introduce a template optimization module that iteratively refines both the prompt and embedding based on model feedback. On commercial models like GPT-4o and Gemini-1.5 Pro, our method achieves attack success rates of over 90% using an average of only 3 queries.



## **13. AdvReal: Adversarial Patch Generation Framework with Application to Adversarial Safety Evaluation of Object Detection Systems**

cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16402v1) [paper-pdf](http://arxiv.org/pdf/2505.16402v1)

**Authors**: Yuanhao Huang, Yilong Ren, Jinlei Wang, Lujia Huo, Xuesong Bai, Jinchuan Zhang, Haiyan Yu

**Abstract**: Autonomous vehicles are typical complex intelligent systems with artificial intelligence at their core. However, perception methods based on deep learning are extremely vulnerable to adversarial samples, resulting in safety accidents. How to generate effective adversarial examples in the physical world and evaluate object detection systems is a huge challenge. In this study, we propose a unified joint adversarial training framework for both 2D and 3D samples to address the challenges of intra-class diversity and environmental variations in real-world scenarios. Building upon this framework, we introduce an adversarial sample reality enhancement approach that incorporates non-rigid surface modeling and a realistic 3D matching mechanism. We compare with 5 advanced adversarial patches and evaluate their attack performance on 8 object detecotrs, including single-stage, two-stage, and transformer-based models. Extensive experiment results in digital and physical environments demonstrate that the adversarial textures generated by our method can effectively mislead the target detection model. Moreover, proposed method demonstrates excellent robustness and transferability under multi-angle attacks, varying lighting conditions, and different distance in the physical world. The demo video and code can be obtained at https://github.com/Huangyh98/AdvReal.git.



## **14. Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems**

cs.IR

7 pages,3 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16367v1) [paper-pdf](http://arxiv.org/pdf/2505.16367v1)

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan

**Abstract**: Retrieval-augmented generation (RAG) systems can effectively mitigate the hallucination problem of large language models (LLMs),but they also possess inherent vulnerabilities. Identifying these weaknesses before the large-scale real-world deployment of RAG systems is of great importance, as it lays the foundation for building more secure and robust RAG systems in the future. Existing adversarial attack methods typically exploit knowledge base poisoning to probe the vulnerabilities of RAG systems, which can effectively deceive standard RAG models. However, with the rapid advancement of deep reasoning capabilities in modern LLMs, previous approaches that merely inject incorrect knowledge are inadequate when attacking RAG systems equipped with deep reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this paper extracts reasoning process templates from R1-based RAG systems, uses these templates to wrap erroneous knowledge into adversarial documents, and injects them into the knowledge base to attack RAG systems. The key idea of our approach is that adversarial documents, by simulating the chain-of-thought patterns aligned with the model's training signals, may be misinterpreted by the model as authentic historical reasoning processes, thus increasing their likelihood of being referenced. Experiments conducted on the MS MARCO passage ranking dataset demonstrate the effectiveness of our proposed method.



## **15. SuperPure: Efficient Purification of Localized and Distributed Adversarial Patches via Super-Resolution GAN Models**

cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16318v1) [paper-pdf](http://arxiv.org/pdf/2505.16318v1)

**Authors**: Hossein Khalili, Seongbin Park, Venkat Bollapragada, Nader Sehatbakhsh

**Abstract**: As vision-based machine learning models are increasingly integrated into autonomous and cyber-physical systems, concerns about (physical) adversarial patch attacks are growing. While state-of-the-art defenses can achieve certified robustness with minimal impact on utility against highly-concentrated localized patch attacks, they fall short in two important areas: (i) State-of-the-art methods are vulnerable to low-noise distributed patches where perturbations are subtly dispersed to evade detection or masking, as shown recently by the DorPatch attack; (ii) Achieving high robustness with state-of-the-art methods is extremely time and resource-consuming, rendering them impractical for latency-sensitive applications in many cyber-physical systems.   To address both robustness and latency issues, this paper proposes a new defense strategy for adversarial patch attacks called SuperPure. The key novelty is developing a pixel-wise masking scheme that is robust against both distributed and localized patches. The masking involves leveraging a GAN-based super-resolution scheme to gradually purify the image from adversarial patches. Our extensive evaluations using ImageNet and two standard classifiers, ResNet and EfficientNet, show that SuperPure advances the state-of-the-art in three major directions: (i) it improves the robustness against conventional localized patches by more than 20%, on average, while also improving top-1 clean accuracy by almost 10%; (ii) It achieves 58% robustness against distributed patch attacks (as opposed to 0% in state-of-the-art method, PatchCleanser); (iii) It decreases the defense end-to-end latency by over 98% compared to PatchCleanser. Our further analysis shows that SuperPure is robust against white-box attacks and different patch sizes. Our code is open-source.



## **16. Accelerating Targeted Hard-Label Adversarial Attacks in Low-Query Black-Box Settings**

cs.CV

This paper contains 11 pages, 7 figures and 3 tables. For associated  supplementary code, see https://github.com/mdppml/TEA

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16313v1) [paper-pdf](http://arxiv.org/pdf/2505.16313v1)

**Authors**: Arjhun Swaminathan, Mete Akgün

**Abstract**: Deep neural networks for image classification remain vulnerable to adversarial examples -- small, imperceptible perturbations that induce misclassifications. In black-box settings, where only the final prediction is accessible, crafting targeted attacks that aim to misclassify into a specific target class is particularly challenging due to narrow decision regions. Current state-of-the-art methods often exploit the geometric properties of the decision boundary separating a source image and a target image rather than incorporating information from the images themselves. In contrast, we propose Targeted Edge-informed Attack (TEA), a novel attack that utilizes edge information from the target image to carefully perturb it, thereby producing an adversarial image that is closer to the source image while still achieving the desired target classification. Our approach consistently outperforms current state-of-the-art methods across different models in low query settings (nearly 70\% fewer queries are used), a scenario especially relevant in real-world applications with limited queries and black-box access. Furthermore, by efficiently generating a suitable adversarial example, TEA provides an improved target initialization for established geometry-based attacks.



## **17. Timestamp Manipulation: Timestamp-based Nakamoto-style Blockchains are Vulnerable**

cs.CR

26 pages, 6 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.05328v3) [paper-pdf](http://arxiv.org/pdf/2505.05328v3)

**Authors**: Junjie Hu, Na Ruan, Sisi Duan

**Abstract**: Nakamoto consensus are the most widely adopted decentralized consensus mechanism in cryptocurrency systems. Since it was proposed in 2008, many studies have focused on analyzing its security. Most of them focus on maximizing the profit of the adversary. Examples include the selfish mining attack [FC '14] and the recent riskless uncle maker (RUM) attack [CCS '23]. In this work, we introduce the Staircase-Unrestricted Uncle Maker (SUUM), the first block withholding attack targeting the timestamp-based Nakamoto-style blockchain. Through block withholding, timestamp manipulation, and difficulty risk control, SUUM adversaries are capable of launching persistent attacks with zero cost and minimal difficulty risk characteristics, indefinitely exploiting rewards from honest participants. This creates a self-reinforcing cycle that threatens the security of blockchains. We conduct a comprehensive and systematic evaluation of SUUM, including the attack conditions, its impact on blockchains, and the difficulty risks. Finally, we further discuss four feasible mitigation measures against SUUM.



## **18. Decentralized Nonconvex Robust Optimization over Unsafe Multiagent Systems: System Modeling, Utility, Resilience, and Privacy Analysis**

math.OC

15 pages, 15 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2409.18632v7) [paper-pdf](http://arxiv.org/pdf/2409.18632v7)

**Authors**: Jinhui Hu, Guo Chen, Huaqing Li, Huqiang Cheng, Xiaoyu Guo, Tingwen Huang

**Abstract**: Privacy leakage and Byzantine failures are two adverse factors to the intelligent decision-making process of multi-agent systems (MASs). Considering the presence of these two issues, this paper targets the resolution of a class of nonconvex optimization problems under the Polyak-{\L}ojasiewicz (P-{\L}) condition. To address this problem, we first identify and construct the adversary system model. To enhance the robustness of stochastic gradient descent methods, we mask the local gradients with Gaussian noises and adopt a resilient aggregation method self-centered clipping (SCC) to design a differentially private (DP) decentralized Byzantine-resilient algorithm, namely DP-SCC-PL, which simultaneously achieves differential privacy and Byzantine resilience. The convergence analysis of DP-SCC-PL is challenging since the convergence error can be contributed jointly by privacy-preserving and Byzantine-resilient mechanisms, as well as the nonconvex relaxation, which is addressed via seeking the contraction relationships among the disagreement measure of reliable agents before and after aggregation, together with the optimal gap. Theoretical results reveal that DP-SCC-PL achieves consensus among all reliable agents and sublinear (inexact) convergence with well-designed step-sizes. It has also been proved that if there are no privacy issues and Byzantine agents, then the asymptotic exact convergence can be recovered. Numerical experiments verify the utility, resilience, and differential privacy of DP-SCC-PL by tackling a nonconvex optimization problem satisfying the P-{\L} condition under various Byzantine attacks.



## **19. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

cs.IR

29 pages

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.12574v3) [paper-pdf](http://arxiv.org/pdf/2505.12574v3)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.



## **20. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.17038v4) [paper-pdf](http://arxiv.org/pdf/2412.17038v4)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Yueyun Shang, Zhihong Tian

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.



## **21. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.13862v2) [paper-pdf](http://arxiv.org/pdf/2505.13862v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.



## **22. SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning**

cs.AI

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16186v1) [paper-pdf](http://arxiv.org/pdf/2505.16186v1)

**Authors**: Kaiwen Zhou, Xuandong Zhao, Gaowen Liu, Jayanth Srinivasa, Aosong Feng, Dawn Song, Xin Eric Wang

**Abstract**: Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations.



## **23. TRAIL: Transferable Robust Adversarial Images via Latent diffusion**

cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16166v1) [paper-pdf](http://arxiv.org/pdf/2505.16166v1)

**Authors**: Yuhao Xue, Zhifei Zhang, Xinyang Jiang, Yifei Shen, Junyao Gao, Wentao Gu, Jiale Zhao, Miaojing Shi, Cairong Zhao

**Abstract**: Adversarial attacks exploiting unrestricted natural perturbations present severe security risks to deep learning systems, yet their transferability across models remains limited due to distribution mismatches between generated adversarial features and real-world data. While recent works utilize pre-trained diffusion models as adversarial priors, they still encounter challenges due to the distribution shift between the distribution of ideal adversarial samples and the natural image distribution learned by the diffusion model. To address the challenge, we propose Transferable Robust Adversarial Images via Latent Diffusion (TRAIL), a test-time adaptation framework that enables the model to generate images from a distribution of images with adversarial features and closely resembles the target images. To mitigate the distribution shift, during attacks, TRAIL updates the diffusion U-Net's weights by combining adversarial objectives (to mislead victim models) and perceptual constraints (to preserve image realism). The adapted model then generates adversarial samples through iterative noise injection and denoising guided by these objectives. Experiments demonstrate that TRAIL significantly outperforms state-of-the-art methods in cross-model attack transferability, validating that distribution-aligned adversarial feature synthesis is critical for practical black-box attacks.



## **24. Adaptive Honeypot Allocation in Multi-Attacker Networks via Bayesian Stackelberg Games**

cs.GT

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.16043v1) [paper-pdf](http://arxiv.org/pdf/2505.16043v1)

**Authors**: Dongyoung Park, Gaby G. Dagher

**Abstract**: Defending against sophisticated cyber threats demands strategic allocation of limited security resources across complex network infrastructures. When the defender has limited defensive resources, the complexity of coordinating honeypot placements across hundreds of nodes grows exponentially. In this paper, we present a multi-attacker Bayesian Stackelberg framework modeling concurrent adversaries attempting to breach a directed network of system components. Our approach uniquely characterizes each adversary through distinct target preferences, exploit capabilities, and associated costs, while enabling defenders to strategically deploy honeypots at critical network positions. By integrating a multi-follower Stackelberg formulation with dynamic Bayesian belief updates, our framework allows defenders to continuously refine their understanding of attacker intentions based on actions detected through Intrusion Detection Systems (IDS). Experimental results show that the proposed method prevents attack success within a few rounds and scales well up to networks of 500 nodes with more than 1,500 edges, maintaining tractable run times.



## **25. SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders**

cs.LG

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2501.18052v3) [paper-pdf](http://arxiv.org/pdf/2501.18052v3)

**Authors**: Bartosz Cywiński, Kamil Deja

**Abstract**: Diffusion models, while powerful, can inadvertently generate harmful or undesirable content, raising significant ethical and safety concerns. Recent machine unlearning approaches offer potential solutions but often lack transparency, making it difficult to understand the changes they introduce to the base model. In this work, we introduce SAeUron, a novel method leveraging features learned by sparse autoencoders (SAEs) to remove unwanted concepts in text-to-image diffusion models. First, we demonstrate that SAEs, trained in an unsupervised manner on activations from multiple denoising timesteps of the diffusion model, capture sparse and interpretable features corresponding to specific concepts. Building on this, we propose a feature selection method that enables precise interventions on model activations to block targeted content while preserving overall performance. Our evaluation shows that SAeUron outperforms existing approaches on the UnlearnCanvas benchmark for concepts and style unlearning, and effectively eliminates nudity when evaluated with I2P. Moreover, we show that with a single SAE, we can remove multiple concepts simultaneously and that in contrast to other methods, SAeUron mitigates the possibility of generating unwanted content under adversarial attack. Code and checkpoints are available at https://github.com/cywinski/SAeUron.



## **26. Ranking Free RAG: Replacing Re-ranking with Selection in RAG for Sensitive Domains**

cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.16014v1) [paper-pdf](http://arxiv.org/pdf/2505.16014v1)

**Authors**: Yash Saxena, Anpur Padia, Mandar S Chaudhary, Kalpa Gunaratna, Srinivasan Parthasarathy, Manas Gaur

**Abstract**: Traditional Retrieval-Augmented Generation (RAG) pipelines rely on similarity-based retrieval and re-ranking, which depend on heuristics such as top-k, and lack explainability, interpretability, and robustness against adversarial content. To address this gap, we propose a novel method METEORA that replaces re-ranking in RAG with a rationale-driven selection approach. METEORA operates in two stages. First, a general-purpose LLM is preference-tuned to generate rationales conditioned on the input query using direct preference optimization. These rationales guide the evidence chunk selection engine, which selects relevant chunks in three stages: pairing individual rationales with corresponding retrieved chunks for local relevance, global selection with elbow detection for adaptive cutoff, and context expansion via neighboring chunks. This process eliminates the need for top-k heuristics. The rationales are also used for consistency check using a Verifier LLM to detect and filter poisoned or misleading content for safe generation. The framework provides explainable and interpretable evidence flow by using rationales consistently across both selection and verification. Our evaluation across six datasets spanning legal, financial, and academic research domains shows that METEORA improves generation accuracy by 33.34% while using approximately 50% fewer chunks than state-of-the-art re-ranking methods. In adversarial settings, METEORA significantly improves the F1 score from 0.10 to 0.44 over the state-of-the-art perplexity-based defense baseline, demonstrating strong resilience to poisoning attacks. Code available at: https://anonymous.4open.science/r/METEORA-DC46/README.md



## **27. Reverse Engineering Human Preferences with Reinforcement Learning**

cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15795v1) [paper-pdf](http://arxiv.org/pdf/2505.15795v1)

**Authors**: Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo

**Abstract**: The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.



## **28. Scalable Defense against In-the-wild Jailbreaking Attacks with Safety Context Retrieval**

cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15753v1) [paper-pdf](http://arxiv.org/pdf/2505.15753v1)

**Authors**: Taiye Chen, Zeming Wei, Ang Li, Yisen Wang

**Abstract**: Large Language Models (LLMs) are known to be vulnerable to jailbreaking attacks, wherein adversaries exploit carefully engineered prompts to induce harmful or unethical responses. Such threats have raised critical concerns about the safety and reliability of LLMs in real-world deployment. While existing defense mechanisms partially mitigate such risks, subsequent advancements in adversarial techniques have enabled novel jailbreaking methods to circumvent these protections, exposing the limitations of static defense frameworks. In this work, we explore defending against evolving jailbreaking threats through the lens of context retrieval. First, we conduct a preliminary study demonstrating that even a minimal set of safety-aligned examples against a particular jailbreak can significantly enhance robustness against this attack pattern. Building on this insight, we further leverage the retrieval-augmented generation (RAG) techniques and propose Safety Context Retrieval (SCR), a scalable and robust safeguarding paradigm for LLMs against jailbreaking. Our comprehensive experiments demonstrate how SCR achieves superior defensive performance against both established and emerging jailbreaking tactics, contributing a new paradigm to LLM safety. Our code will be available upon publication.



## **29. Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses**

cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15738v1) [paper-pdf](http://arxiv.org/pdf/2505.15738v1)

**Authors**: Xiaoxue Yang, Bozhidar Stevanoski, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs.



## **30. Beyond Classification: Evaluating Diffusion Denoised Smoothing for Security-Utility Trade off**

cs.LG

Paper accepted at the 33rd European Signal Processing Conference  (EUSIPCO 2025)

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15594v1) [paper-pdf](http://arxiv.org/pdf/2505.15594v1)

**Authors**: Yury Belousov, Brian Pulfer, Vitaliy Kinakh, Slava Voloshynovskiy

**Abstract**: While foundation models demonstrate impressive performance across various tasks, they remain vulnerable to adversarial inputs. Current research explores various approaches to enhance model robustness, with Diffusion Denoised Smoothing emerging as a particularly promising technique. This method employs a pretrained diffusion model to preprocess inputs before model inference. Yet, its effectiveness remains largely unexplored beyond classification. We aim to address this gap by analyzing three datasets with four distinct downstream tasks under three different adversarial attack algorithms. Our findings reveal that while foundation models maintain resilience against conventional transformations, applying high-noise diffusion denoising to clean images without any distortions significantly degrades performance by as high as 57%. Low-noise diffusion settings preserve performance but fail to provide adequate protection across all attack types. Moreover, we introduce a novel attack strategy specifically targeting the diffusion process itself, capable of circumventing defenses in the low-noise regime. Our results suggest that the trade-off between adversarial robustness and performance remains a challenge to be addressed.



## **31. Red-Teaming for Inducing Societal Bias in Large Language Models**

cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2405.04756v2) [paper-pdf](http://arxiv.org/pdf/2405.04756v2)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Bharat Bhimshetty, Kashyap Murali, Murli Jadhav, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Ensuring the safe deployment of AI systems is critical in industry settings where biased outputs can lead to significant operational, reputational, and regulatory risks. Thorough evaluation before deployment is essential to prevent these hazards. Red-teaming addresses this need by employing adversarial attacks to develop guardrails that detect and reject biased or harmful queries, enabling models to be retrained or steered away from harmful outputs. However, most red-teaming efforts focus on harmful or unethical instructions rather than addressing social bias, leaving this critical area under-explored despite its significant real-world impact, especially in customer-facing systems. We propose two bias-specific red-teaming methods, Emotional Bias Probe (EBP) and BiasKG, to evaluate how standard safety measures for harmful content affect bias. For BiasKG, we refactor natural language stereotypes into a knowledge graph. We use these attacking strategies to induce biased responses from several open- and closed-source language models. Unlike prior work, these methods specifically target social bias. We find our method increases bias in all models, even those trained with safety guardrails. Our work emphasizes uncovering societal bias in LLMs through rigorous evaluation, and recommends measures ensure AI safety in high-stakes industry deployments.



## **32. ModSec-AdvLearn: Countering Adversarial SQL Injections with Robust Machine Learning**

cs.LG

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2308.04964v4) [paper-pdf](http://arxiv.org/pdf/2308.04964v4)

**Authors**: Giuseppe Floris, Christian Scano, Biagio Montaruli, Luca Demetrio, Andrea Valenza, Luca Compagna, Davide Ariu, Luca Piras, Davide Balzarotti, Battista Biggio

**Abstract**: Many Web Application Firewalls (WAFs) leverage the OWASP CRS to block incoming malicious requests. The CRS consists of different sets of rules designed by domain experts to detect well-known web attack patterns. Both the set of rules and the weights used to combine them are manually defined, yielding four different default configurations of the CRS. In this work, we focus on the detection of SQLi attacks, and show that the manual configurations of the CRS typically yield a suboptimal trade-off between detection and false alarm rates. Furthermore, we show that these configurations are not robust to adversarial SQLi attacks, i.e., carefully-crafted attacks that iteratively refine the malicious SQLi payload by querying the target WAF to bypass detection. To overcome these limitations, we propose (i) using machine learning to automate the selection of the set of rules to be combined along with their weights, i.e., customizing the CRS configuration based on the monitored web services; and (ii) leveraging adversarial training to significantly improve its robustness to adversarial SQLi manipulations. Our experiments, conducted using the well-known open-source ModSecurity WAF equipped with the CRS rules, show that our approach, named ModSec-AdvLearn, can (i) increase the detection rate up to 30%, while retaining negligible false alarm rates and discarding up to 50% of the CRS rules; and (ii) improve robustness against adversarial SQLi attacks up to 85%, marking a significant stride toward designing more effective and robust WAFs. We release our open-source code at https://github.com/pralab/modsec-advlearn.



## **33. Audio Jailbreak: An Open Comprehensive Benchmark for Jailbreaking Large Audio-Language Models**

cs.SD

We release AJailBench, including both static and optimized  adversarial data, to facilitate future research:  https://github.com/mbzuai-nlp/AudioJailbreak

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15406v1) [paper-pdf](http://arxiv.org/pdf/2505.15406v1)

**Authors**: Zirui Song, Qian Jiang, Mingxuan Cui, Mingzhe Li, Lang Gao, Zeyu Zhang, Zixiang Xu, Yanbo Wang, Chenxi Wang, Guangxian Ouyang, Zhenhao Chen, Xiuying Chen

**Abstract**: The rise of Large Audio Language Models (LAMs) brings both potential and risks, as their audio outputs may contain harmful or unethical content. However, current research lacks a systematic, quantitative evaluation of LAM safety especially against jailbreak attacks, which are challenging due to the temporal and semantic nature of speech. To bridge this gap, we introduce AJailBench, the first benchmark specifically designed to evaluate jailbreak vulnerabilities in LAMs. We begin by constructing AJailBench-Base, a dataset of 1,495 adversarial audio prompts spanning 10 policy-violating categories, converted from textual jailbreak attacks using realistic text to speech synthesis. Using this dataset, we evaluate several state-of-the-art LAMs and reveal that none exhibit consistent robustness across attacks. To further strengthen jailbreak testing and simulate more realistic attack conditions, we propose a method to generate dynamic adversarial variants. Our Audio Perturbation Toolkit (APT) applies targeted distortions across time, frequency, and amplitude domains. To preserve the original jailbreak intent, we enforce a semantic consistency constraint and employ Bayesian optimization to efficiently search for perturbations that are both subtle and highly effective. This results in AJailBench-APT, an extended dataset of optimized adversarial audio samples. Our findings demonstrate that even small, semantically preserved perturbations can significantly reduce the safety performance of leading LAMs, underscoring the need for more robust and semantically aware defense mechanisms.



## **34. My Face Is Mine, Not Yours: Facial Protection Against Diffusion Model Face Swapping**

cs.CV

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15336v1) [paper-pdf](http://arxiv.org/pdf/2505.15336v1)

**Authors**: Hon Ming Yam, Zhongliang Guo, Chun Pong Lau

**Abstract**: The proliferation of diffusion-based deepfake technologies poses significant risks for unauthorized and unethical facial image manipulation. While traditional countermeasures have primarily focused on passive detection methods, this paper introduces a novel proactive defense strategy through adversarial attacks that preemptively protect facial images from being exploited by diffusion-based deepfake systems. Existing adversarial protection methods predominantly target conventional generative architectures (GANs, AEs, VAEs) and fail to address the unique challenges presented by diffusion models, which have become the predominant framework for high-quality facial deepfakes. Current diffusion-specific adversarial approaches are limited by their reliance on specific model architectures and weights, rendering them ineffective against the diverse landscape of diffusion-based deepfake implementations. Additionally, they typically employ global perturbation strategies that inadequately address the region-specific nature of facial manipulation in deepfakes.



## **35. BadSR: Stealthy Label Backdoor Attacks on Image Super-Resolution**

cs.CV

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15308v1) [paper-pdf](http://arxiv.org/pdf/2505.15308v1)

**Authors**: Ji Guo, Xiaolei Wen, Wenbo Jiang, Cheng Huang, Jinjin Li, Hongwei Li

**Abstract**: With the widespread application of super-resolution (SR) in various fields, researchers have begun to investigate its security. Previous studies have demonstrated that SR models can also be subjected to backdoor attacks through data poisoning, affecting downstream tasks. A backdoor SR model generates an attacker-predefined target image when given a triggered image while producing a normal high-resolution (HR) output for clean images. However, prior backdoor attacks on SR models have primarily focused on the stealthiness of poisoned low-resolution (LR) images while ignoring the stealthiness of poisoned HR images, making it easy for users to detect anomalous data. To address this problem, we propose BadSR, which improves the stealthiness of poisoned HR images. The key idea of BadSR is to approximate the clean HR image and the pre-defined target image in the feature space while ensuring that modifications to the clean HR image remain within a constrained range. The poisoned HR images generated by BadSR can be integrated with existing triggers. To further improve the effectiveness of BadSR, we design an adversarially optimized trigger and a backdoor gradient-driven poisoned sample selection method based on a genetic algorithm. The experimental results show that BadSR achieves a high attack success rate in various models and data sets, significantly affecting downstream tasks.



## **36. Robustness Evaluation of Graph-based News Detection Using Network Structural Information**

cs.SI

Accepted to Proceedings of the ACM SIGKDD Conference on Knowledge  Discovery and Data Mining 2025 (KDD 2025). 14 pages, 7 figures, 10 tables

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.14453v2) [paper-pdf](http://arxiv.org/pdf/2505.14453v2)

**Authors**: Xianghua Zeng, Hao Peng, Angsheng Li

**Abstract**: Although Graph Neural Networks (GNNs) have shown promising potential in fake news detection, they remain highly vulnerable to adversarial manipulations within social networks. Existing methods primarily establish connections between malicious accounts and individual target news to investigate the vulnerability of graph-based detectors, while they neglect the structural relationships surrounding targets, limiting their effectiveness in robustness evaluation. In this work, we propose a novel Structural Information principles-guided Adversarial Attack Framework, namely SI2AF, which effectively challenges graph-based detectors and further probes their detection robustness. Specifically, structural entropy is introduced to quantify the dynamic uncertainty in social engagements and identify hierarchical communities that encompass all user accounts and news posts. An influence metric is presented to measure each account's probability of engaging in random interactions, facilitating the design of multiple agents that manage distinct malicious accounts. For each target news, three attack strategies are developed through multi-agent collaboration within the associated subgraph to optimize evasion against black-box detectors. By incorporating the adversarial manipulations generated by SI2AF, we enrich the original network structure and refine graph-based detectors to improve their robustness against adversarial attacks. Extensive evaluations demonstrate that SI2AF significantly outperforms state-of-the-art baselines in attack effectiveness with an average improvement of 16.71%, and enhances GNN-based detection robustness by 41.54% on average.



## **37. Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs**

cs.CV

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15265v1) [paper-pdf](http://arxiv.org/pdf/2505.15265v1)

**Authors**: Zihao Pan, Yu Tong, Weibin Wu, Jingyi Wang, Lifeng Chen, Zhe Zhao, Jiajia Wei, Yitong Qiao, Zibin Zheng

**Abstract**: Adversarial attacks aim to generate malicious inputs that mislead deep models, but beyond causing model failure, they cannot provide certain interpretable information such as ``\textit{What content in inputs make models more likely to fail?}'' However, this information is crucial for researchers to specifically improve model robustness. Recent research suggests that models may be particularly sensitive to certain semantics in visual inputs (such as ``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this paper we conducted the first exploration on large vision-language models (LVLMs) and found that LVLMs indeed are susceptible to hallucinations and various errors when facing specific semantic concepts in images. To efficiently search for these sensitive concepts, we integrated large language models (LLMs) and text-to-image (T2I) models to propose a novel semantic evolution framework. Randomly initialized semantic concepts undergo LLM-based crossover and mutation operations to form image descriptions, which are then converted by T2I models into visual inputs for LVLMs. The task-specific performance of LVLMs on each input is quantified as fitness scores for the involved semantics and serves as reward signals to further guide LLMs in exploring concepts that induce LVLMs. Extensive experiments on seven mainstream LVLMs and two multimodal tasks demonstrate the effectiveness of our method. Additionally, we provide interesting findings about the sensitive semantics of LVLMs, aiming to inspire further in-depth research.



## **38. From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios**

cs.AI

New version of the paper

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2502.02145v3) [paper-pdf](http://arxiv.org/pdf/2502.02145v3)

**Authors**: Yuan Gao, Mattia Piccinini, Korbinian Moller, Amr Alanwar, Johannes Betz

**Abstract**: Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.



## **39. Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks**

cs.LG

11 pages, 11 figures, accepted by IJCAI 2025

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.04046v2) [paper-pdf](http://arxiv.org/pdf/2505.04046v2)

**Authors**: Xuyang Wang, Siyuan Duan, Qizhi Li, Guiduo Duan, Yuan Sun, Dezhong Peng

**Abstract**: Trustworthy multi-view learning has attracted extensive attention because evidence learning can provide reliable uncertainty estimation to enhance the credibility of multi-view predictions. Existing trusted multi-view learning methods implicitly assume that multi-view data is secure. However, in safety-sensitive applications such as autonomous driving and security monitoring, multi-view data often faces threats from adversarial perturbations, thereby deceiving or disrupting multi-view models. This inevitably leads to the adversarial unreliability problem (AUP) in trusted multi-view learning. To overcome this tricky problem, we propose a novel multi-view learning framework, namely Reliable Disentanglement Multi-view Learning (RDML). Specifically, we first propose evidential disentanglement learning to decompose each view into clean and adversarial parts under the guidance of corresponding evidences, which is extracted by a pretrained evidence extractor. Then, we employ the feature recalibration module to mitigate the negative impact of adversarial perturbations and extract potential informative features from them. Finally, to further ignore the irreparable adversarial interferences, a view-level evidential attention mechanism is designed. Extensive experiments on multi-view classification tasks with adversarial attacks show that RDML outperforms the state-of-the-art methods by a relatively large margin. Our code is available at https://github.com/Willy1005/2025-IJCAI-RDML.



## **40. Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models**

cs.LG

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15130v1) [paper-pdf](http://arxiv.org/pdf/2505.15130v1)

**Authors**: Sajjad Ghiasvand, Haniyeh Ehsani Oskouie, Mahnoosh Alizadeh, Ramtin Pedarsani

**Abstract**: Vision-Language Models (VLMs) such as CLIP have shown remarkable performance in cross-modal tasks through large-scale contrastive pre-training. To adapt these large transformer-based models efficiently for downstream tasks, Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA have emerged as scalable alternatives to full fine-tuning, especially in few-shot scenarios. However, like traditional deep neural networks, VLMs are highly vulnerable to adversarial attacks, where imperceptible perturbations can significantly degrade model performance. Adversarial training remains the most effective strategy for improving model robustness in PEFT. In this work, we propose AdvCLIP-LoRA, the first algorithm designed to enhance the adversarial robustness of CLIP models fine-tuned with LoRA in few-shot settings. Our method formulates adversarial fine-tuning as a minimax optimization problem and provides theoretical guarantees for convergence under smoothness and nonconvex-strong-concavity assumptions. Empirical results across eight datasets using ViT-B/16 and ViT-B/32 models show that AdvCLIP-LoRA significantly improves robustness against common adversarial attacks (e.g., FGSM, PGD), without sacrificing much clean accuracy. These findings highlight AdvCLIP-LoRA as a practical and theoretically grounded approach for robust adaptation of VLMs in resource-constrained settings.



## **41. A Survey On Secure Machine Learning**

cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15124v1) [paper-pdf](http://arxiv.org/pdf/2505.15124v1)

**Authors**: Taobo Liao, Taoran Li, Prathamesh Nadkarni

**Abstract**: In this survey, we will explore the interaction between secure multiparty computation and the area of machine learning. Recent advances in secure multiparty computation (MPC) have significantly improved its applicability in the realm of machine learning (ML), offering robust solutions for privacy-preserving collaborative learning. This review explores key contributions that leverage MPC to enable multiple parties to engage in ML tasks without compromising the privacy of their data. The integration of MPC with ML frameworks facilitates the training and evaluation of models on combined datasets from various sources, ensuring that sensitive information remains encrypted throughout the process. Innovations such as specialized software frameworks and domain-specific languages streamline the adoption of MPC in ML, optimizing performance and broadening its usage. These frameworks address both semi-honest and malicious threat models, incorporating features such as automated optimizations and cryptographic auditing to ensure compliance and data integrity. The collective insights from these studies highlight MPC's potential in fostering collaborative yet confidential data analysis, marking a significant stride towards the realization of secure and efficient computational solutions in privacy-sensitive industries. This paper investigates a spectrum of SecureML libraries that includes cryptographic protocols, federated learning frameworks, and privacy-preserving algorithms. By surveying the existing literature, this paper aims to examine the efficacy of these libraries in preserving data privacy, ensuring model confidentiality, and fortifying ML systems against adversarial attacks. Additionally, the study explores an innovative application domain for SecureML techniques: the integration of these methodologies in gaming environments utilizing ML.



## **42. AudioJailbreak: Jailbreak Attacks against End-to-End Large Audio-Language Models**

cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.14103v2) [paper-pdf](http://arxiv.org/pdf/2505.14103v2)

**Authors**: Guangke Chen, Fu Song, Zhe Zhao, Xiaojun Jia, Yang Liu, Yanchen Qiao, Weizhe Zhang

**Abstract**: Jailbreak attacks to Large audio-language models (LALMs) are studied recently, but they achieve suboptimal effectiveness, applicability, and practicability, particularly, assuming that the adversary can fully manipulate user prompts. In this work, we first conduct an extensive experiment showing that advanced text jailbreak attacks cannot be easily ported to end-to-end LALMs via text-to speech (TTS) techniques. We then propose AudioJailbreak, a novel audio jailbreak attack, featuring (1) asynchrony: the jailbreak audio does not need to align with user prompts in the time axis by crafting suffixal jailbreak audios; (2) universality: a single jailbreak perturbation is effective for different prompts by incorporating multiple prompts into perturbation generation; (3) stealthiness: the malicious intent of jailbreak audios will not raise the awareness of victims by proposing various intent concealment strategies; and (4) over-the-air robustness: the jailbreak audios remain effective when being played over the air by incorporating the reverberation distortion effect with room impulse response into the generation of the perturbations. In contrast, all prior audio jailbreak attacks cannot offer asynchrony, universality, stealthiness, or over-the-air robustness. Moreover, AudioJailbreak is also applicable to the adversary who cannot fully manipulate user prompts, thus has a much broader attack scenario. Extensive experiments with thus far the most LALMs demonstrate the high effectiveness of AudioJailbreak. We highlight that our work peeks into the security implications of audio jailbreak attacks against LALMs, and realistically fosters improving their security robustness. The implementation and audio samples are available at our website https://audiojailbreak.github.io/AudioJailbreak.



## **43. MrGuard: A Multilingual Reasoning Guardrail for Universal LLM Safety**

cs.CL

Preprint

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2504.15241v2) [paper-pdf](http://arxiv.org/pdf/2504.15241v2)

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual settings, where multilingual safety-aligned data is often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we introduce a multilingual guardrail with reasoning for prompt classification. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-based Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail, MrGuard, consistently outperforms recent baselines across both in-domain and out-of-domain languages by more than 15%. We also evaluate MrGuard's robustness to multilingual variations, such as code-switching and low-resource language distractors in the prompt, and demonstrate that it preserves safety judgments under these challenging conditions. The multilingual reasoning capability of our guardrail enables it to generate explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.



## **44. Lessons from Defending Gemini Against Indirect Prompt Injections**

cs.CR

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14534v1) [paper-pdf](http://arxiv.org/pdf/2505.14534v1)

**Authors**: Chongyang Shi, Sharon Lin, Shuang Song, Jamie Hayes, Ilia Shumailov, Itay Yona, Juliette Pluto, Aneesh Pappu, Christopher A. Choquette-Choo, Milad Nasr, Chawin Sitawarin, Gena Gibson, Andreas Terzis, John "Four" Flynn

**Abstract**: Gemini is increasingly used to perform tasks on behalf of users, where function-calling and tool-use capabilities enable the model to access user data. Some tools, however, require access to untrusted data introducing risk. Adversaries can embed malicious instructions in untrusted data which cause the model to deviate from the user's expectations and mishandle their data or permissions. In this report, we set out Google DeepMind's approach to evaluating the adversarial robustness of Gemini models and describe the main lessons learned from the process. We test how Gemini performs against a sophisticated adversary through an adversarial evaluation framework, which deploys a suite of adaptive attack techniques to run continuously against past, current, and future versions of Gemini. We describe how these ongoing evaluations directly help make Gemini more resilient against manipulation.



## **45. Adverseness vs. Equilibrium: Exploring Graph Adversarial Resilience through Dynamic Equilibrium**

cs.LG

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14463v1) [paper-pdf](http://arxiv.org/pdf/2505.14463v1)

**Authors**: Xinxin Fan, Wenxiong Chen, Mengfan Li, Wenqi Wei, Ling Liu

**Abstract**: Adversarial attacks to graph analytics are gaining increased attention. To date, two lines of countermeasures have been proposed to resist various graph adversarial attacks from the perspectives of either graph per se or graph neural networks. Nevertheless, a fundamental question lies in whether there exists an intrinsic adversarial resilience state within a graph regime and how to find out such a critical state if exists. This paper contributes to tackle the above research questions from three unique perspectives: i) we regard the process of adversarial learning on graph as a complex multi-object dynamic system, and model the behavior of adversarial attack; ii) we propose a generalized theoretical framework to show the existence of critical adversarial resilience state; and iii) we develop a condensed one-dimensional function to capture the dynamic variation of graph regime under perturbations, and pinpoint the critical state through solving the equilibrium point of dynamic system. Multi-facet experiments are conducted to show our proposed approach can significantly outperform the state-of-the-art defense methods under five commonly-used real-world datasets and three representative attacks.



## **46. Vulnerability of Transfer-Learned Neural Networks to Data Reconstruction Attacks in Small-Data Regime**

cs.CR

27 pages

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14323v1) [paper-pdf](http://arxiv.org/pdf/2505.14323v1)

**Authors**: Tomasz Maciążek, Robert Allison

**Abstract**: Training data reconstruction attacks enable adversaries to recover portions of a released model's training data. We consider the attacks where a reconstructor neural network learns to invert the (random) mapping between training data and model weights. Prior work has shown that an informed adversary with access to released model's weights and all but one training data point can achieve high-quality reconstructions in this way. However, differential privacy can defend against such an attack with little to no loss in model's utility when the amount of training data is sufficiently large. In this work we consider a more realistic adversary who only knows the distribution from which a small training dataset has been sampled and who attacks a transfer-learned neural network classifier that has been trained on this dataset. We exhibit an attack that works in this realistic threat model and demonstrate that in the small-data regime it cannot be defended against by DP-SGD without severely damaging the classifier accuracy. This raises significant concerns about the use of such transfer-learned classifiers when protection of training-data is paramount. We demonstrate the effectiveness and robustness of our attack on VGG, EfficientNet and ResNet image classifiers transfer-learned on MNIST, CIFAR-10 and CelebA respectively. Additionally, we point out that the commonly used (true-positive) reconstruction success rate metric fails to reliably quantify the actual reconstruction effectiveness. Instead, we make use of the Neyman-Pearson lemma to construct the receiver operating characteristic curve and consider the associated true-positive reconstruction rate at a fixed level of the false-positive reconstruction rate.



## **47. Exploring Jailbreak Attacks on LLMs through Intent Concealment and Diversion**

cs.CR

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14316v1) [paper-pdf](http://arxiv.org/pdf/2505.14316v1)

**Authors**: Tiehan Cui, Yanxu Mao, Peipei Liu, Congying Liu, Datao You

**Abstract**: Although large language models (LLMs) have achieved remarkable advancements, their security remains a pressing concern. One major threat is jailbreak attacks, where adversarial prompts bypass model safeguards to generate harmful or objectionable content. Researchers study jailbreak attacks to understand security and robustness of LLMs. However, existing jailbreak attack methods face two main challenges: (1) an excessive number of iterative queries, and (2) poor generalization across models. In addition, recent jailbreak evaluation datasets focus primarily on question-answering scenarios, lacking attention to text generation tasks that require accurate regeneration of toxic content. To tackle these challenges, we propose two contributions: (1) ICE, a novel black-box jailbreak method that employs Intent Concealment and divErsion to effectively circumvent security constraints. ICE achieves high attack success rates (ASR) with a single query, significantly improving efficiency and transferability across different models. (2) BiSceneEval, a comprehensive dataset designed for assessing LLM robustness in question-answering and text-generation tasks. Experimental results demonstrate that ICE outperforms existing jailbreak techniques, revealing critical vulnerabilities in current defense mechanisms. Our findings underscore the necessity of a hybrid security strategy that integrates predefined security mechanisms with real-time semantic decomposition to enhance the security of LLMs.



## **48. EVA: Red-Teaming GUI Agents via Evolving Indirect Prompt Injection**

cs.AI

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14289v1) [paper-pdf](http://arxiv.org/pdf/2505.14289v1)

**Authors**: Yijie Lu, Tianjie Ju, Manman Zhao, Xinbei Ma, Yuan Guo, ZhuoSheng Zhang

**Abstract**: As multimodal agents are increasingly trained to operate graphical user interfaces (GUIs) to complete user tasks, they face a growing threat from indirect prompt injection, attacks in which misleading instructions are embedded into the agent's visual environment, such as popups or chat messages, and misinterpreted as part of the intended task. A typical example is environmental injection, in which GUI elements are manipulated to influence agent behavior without directly modifying the user prompt. To address these emerging attacks, we propose EVA, a red teaming framework for indirect prompt injection which transforms the attack into a closed loop optimization by continuously monitoring an agent's attention distribution over the GUI and updating adversarial cues, keywords, phrasing, and layout, in response. Compared with prior one shot methods that generate fixed prompts without regard for how the model allocates visual attention, EVA dynamically adapts to emerging attention hotspots, yielding substantially higher attack success rates and far greater transferability across diverse GUI scenarios. We evaluate EVA on six widely used generalist and specialist GUI agents in realistic settings such as popup manipulation, chat based phishing, payments, and email composition. Experimental results show that EVA substantially improves success rates over static baselines. Under goal agnostic constraints, where the attacker does not know the agent's task intent, EVA still discovers effective patterns. Notably, we find that injection styles transfer well across models, revealing shared behavioral biases in GUI agents. These results suggest that evolving indirect prompt injection is a powerful tool not only for red teaming agents, but also for uncovering common vulnerabilities in their multimodal decision making.



## **49. Universal Acoustic Adversarial Attacks for Flexible Control of Speech-LLMs**

cs.CL

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14286v1) [paper-pdf](http://arxiv.org/pdf/2505.14286v1)

**Authors**: Rao Ma, Mengjie Qian, Vyas Raina, Mark Gales, Kate Knill

**Abstract**: The combination of pre-trained speech encoders with large language models has enabled the development of speech LLMs that can handle a wide range of spoken language processing tasks. While these models are powerful and flexible, this very flexibility may make them more vulnerable to adversarial attacks. To examine the extent of this problem, in this work we investigate universal acoustic adversarial attacks on speech LLMs. Here a fixed, universal, adversarial audio segment is prepended to the original input audio. We initially investigate attacks that cause the model to either produce no output or to perform a modified task overriding the original prompt. We then extend the nature of the attack to be selective so that it activates only when specific input attributes, such as a speaker gender or spoken language, are present. Inputs without the targeted attribute should be unaffected, allowing fine-grained control over the model outputs. Our findings reveal critical vulnerabilities in Qwen2-Audio and Granite-Speech and suggest that similar speech LLMs may be susceptible to universal adversarial attacks. This highlights the need for more robust training strategies and improved resistance to adversarial attacks.



## **50. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

cs.CR

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.12442v2) [paper-pdf](http://arxiv.org/pdf/2505.12442v2)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.



