# Latest Adversarial Attack Papers
**update at 2025-09-17 09:53:34**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. JANUS: A Dual-Constraint Generative Framework for Stealthy Node Injection Attacks**

JANUS：一个用于隐形节点注入攻击的双约束生成框架 cs.LG

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13266v1) [paper-pdf](http://arxiv.org/pdf/2509.13266v1)

**Authors**: Jiahao Zhang, Xiaobing Pei, Zhaokun Zhong, Wenqiang Hao, Zhenghao Tang

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable performance across various applications, yet they are vulnerable to sophisticated adversarial attacks, particularly node injection attacks. The success of such attacks heavily relies on their stealthiness, the ability to blend in with the original graph and evade detection. However, existing methods often achieve stealthiness by relying on indirect proxy metrics, lacking consideration for the fundamental characteristics of the injected content, or focusing only on imitating local structures, which leads to the problem of local myopia. To overcome these limitations, we propose a dual-constraint stealthy node injection framework, called Joint Alignment of Nodal and Universal Structures (JANUS). At the local level, we introduce a local feature manifold alignment strategy to achieve geometric consistency in the feature space. At the global level, we incorporate structured latent variables and maximize the mutual information with the generated structures, ensuring the injected structures are consistent with the semantic patterns of the original graph. We model the injection attack as a sequential decision process, which is optimized by a reinforcement learning agent. Experiments on multiple standard datasets demonstrate that the JANUS framework significantly outperforms existing methods in terms of both attack effectiveness and stealthiness.

摘要: 图形神经网络（GNN）在各种应用程序中表现出了出色的性能，但它们很容易受到复杂的对抗攻击，尤其是节点注入攻击。此类攻击的成功在很大程度上依赖于它们的隐蔽性、融入原始图表并逃避检测的能力。然而，现有的方法往往通过依赖间接代理指标来实现隐蔽性，缺乏对注入内容的基本特征的考虑，或者只专注于模仿局部结构，从而导致局部近视的问题。为了克服这些限制，我们提出了一种双约束隐形节点注入框架，称为节点和通用结构联合对齐（JANUS）。在局部层次上，我们引入了局部特征流形对齐策略，以实现特征空间的几何一致性。在全局层次上，我们将结构化的潜变量和最大化的互信息与生成的结构，确保注入的结构是一致的原始图的语义模式。我们将注入攻击建模为一个顺序决策过程，并通过强化学习代理进行优化。在多个标准数据集上的实验表明，JANUS框架在攻击有效性和隐蔽性方面明显优于现有方法。



## **2. Chernoff Information as a Privacy Constraint for Adversarial Classification and Membership Advantage**

冲突信息作为对抗性分类和成员优势的隐私约束 cs.IT

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2403.10307v3) [paper-pdf](http://arxiv.org/pdf/2403.10307v3)

**Authors**: Ayşe Ünsal, Melek Önen

**Abstract**: This work inspects a privacy metric based on Chernoff information, namely Chernoff differential privacy, due to its significance in characterization of the optimal classifier's performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we characterize the relationship between $\varepsilon\textrm{-}$differential privacy, the best error exponent of one of the errors (when the other is fixed) and the best average error exponent. Accordingly, we re-derive Chernoff differential privacy in connection with $\varepsilon\textrm{-}$differential privacy using the Radon-Nikodym derivative, and prove its relation with Kullback-Leibler (KL) differential privacy. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$ and the impact of the adversary's attack in Laplace mechanisms. Lastly, we introduce a new upper bound on adversary's membership advantage in membership inference attacks using Chernoff DP and numerically compare its performance with existing alternatives based on $(\varepsilon, \delta)\textrm{-}$differential privacy in the literature.

摘要: 这项工作检查了基于Timoff信息的隐私指标，即Timoff差异隐私，因为它在描述最佳分类器性能方面具有重要意义。对抗性分类，就像任何其他分类问题一样，在二元分类的情况下，都是围绕决定任何一个类别时（平均或正确检测）错误概率的最小化而建立的。与经典假设测试问题不同，其中虚警和误判概率分别处理，导致最佳错误指数的不对称行为，在这项工作中，我们描述了$\varepð\texttrm {-}$差异隐私、其中一个错误的最佳错误指数（当另一个错误是固定的时）和最佳平均错误指数之间的关系。因此，我们使用Radon-Nikodym衍生物重新推导与$\varepð\textrm{-}$差异隐私相关的Timoff差异隐私，并证明其与Kullback-Leibler（KL）差异隐私的关系。随后，我们给出了数值评估结果，该结果表明，作为隐私参数$\varepð $和对手攻击在拉普拉斯机制中的影响的函数，Lattoff信息优于Kullback-Leibler分歧。最后，我们在使用Deliverff DP的成员资格推断攻击中引入了对手成员资格优势的新上限，并将其性能与文献中基于$（\varepð，\delta）\texttrm {-}$差异隐私的现有替代方案进行了数字比较。



## **3. Detection of Synthetic Face Images: Accuracy, Robustness, Generalization**

合成人脸图像检测：准确性、鲁棒性、概括性 cs.CV

The paper was presented at the DAGM German Conference on Pattern  Recognition (GCPR), 2025

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2406.17547v2) [paper-pdf](http://arxiv.org/pdf/2406.17547v2)

**Authors**: Nela Petrzelkova, Jan Cech

**Abstract**: An experimental study on detecting synthetic face images is presented. We collected a dataset, called FF5, of five fake face image generators, including recent diffusion models. We find that a simple model trained on a specific image generator can achieve near-perfect accuracy in separating synthetic and real images. The model handles common image distortions (reduced resolution, compression) by using data augmentation. Moreover, partial manipulations, where synthetic images are blended into real ones by inpainting, are identified and the area of the manipulation is localized by a simple model of YOLO architecture. However, the model turned out to be vulnerable to adversarial attacks and does not generalize to unseen generators. Failure to generalize to detect images produced by a newer generator also occurs for recent state-of-the-art methods, which we tested on Realistic Vision, a fine-tuned version of StabilityAI's Stable Diffusion image generator.

摘要: 进行了合成人脸图像检测的实验研究。我们收集了一个名为FF 5的数据集，包含五个假面部图像生成器，包括最近的扩散模型。我们发现，在特定图像生成器上训练的简单模型可以在分离合成图像和真实图像方面实现近乎完美的准确性。该模型通过使用数据增强来处理常见的图像失真（分辨率降低、压缩）。此外，还可以识别部分操纵（通过修补将合成图像混合到真实图像中），并通过YOLO架构的简单模型来本地化操纵区域。然而，事实证明，该模型很容易受到对抗攻击，并且不能推广到看不见的生成器。最近的最先进方法也会出现无法概括检测由较新生成器产生的图像的情况，我们在Realistic Vision上进行了测试，Realistic Vision是StabilityAI的Stable Dispatch图像生成器的微调版本。



## **4. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

大型多模式模型的鲁棒适应用于检索增强仇恨模因检测 cs.CL

EMNLP 2025 Main (Oral)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2502.13061v4) [paper-pdf](http://arxiv.org/pdf/2502.13061v4)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL

摘要: 仇恨模因已成为互联网上的一个重要问题，需要强大的自动化检测系统。虽然大型多模式模型（LSYS）在仇恨模因检测方面表现出了希望，但它们面临着显着的挑战，例如次优的性能和有限的域外概括能力。最近的研究进一步揭示了在这种环境下将监督微调（SFT）和上下文学习应用于LSYS时的局限性。为了解决这些问题，我们提出了一个用于仇恨模因检测的鲁棒适应框架，该框架可以增强领域内准确性和跨领域概括性，同时保留Letts的一般视觉语言能力。分析表明，与SFT模型相比，我们的方法在对抗攻击下实现了更好的鲁棒性。对六个模因分类数据集的实验表明，我们的方法实现了最先进的性能，优于更大的代理系统。此外，与标准SFT相比，我们的方法为解释仇恨内容生成了更高质量的理由，增强了模型的可解释性。代码可访问https://github.com/JingbiaoMei/RGCL



## **5. Bridging Threat Models and Detections: Formal Verification via CADP**

桥梁威胁模型和检测：通过CADP进行正式验证 cs.CR

In Proceedings FROM 2025, arXiv:2509.11877

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13035v1) [paper-pdf](http://arxiv.org/pdf/2509.13035v1)

**Authors**: Dumitru-Bogdan Prelipcean, Cătălin Dima

**Abstract**: Threat detection systems rely on rule-based logic to identify adversarial behaviors, yet the conformance of these rules to high-level threat models is rarely verified formally. We present a formal verification framework that models both detection logic and attack trees as labeled transition systems (LTSs), enabling automated conformance checking via bisimulation and weak trace inclusion. Detection rules specified in the Generic Threat Detection Language (GTDL, a general-purpose detection language we formalize in this work) are assigned a compositional operational semantics, and threat models expressed as attack trees are interpreted as LTSs through a structural trace semantics. Both representations are translated to LNT, a modeling language supported by the CADP toolbox. This common semantic domain enables systematic and automated verification of detection coverage. We evaluate our approach on real-world malware scenarios such as LokiBot and Emotet and provide scalability analysis through parametric synthetic models. Results confirm that our methodology identifies semantic mismatches between threat models and detection rules, supports iterative refinement, and scales to realistic threat landscapes.

摘要: 威胁检测系统依赖基于规则的逻辑来识别对抗行为，但这些规则与高级威胁模型的一致性很少得到正式验证。我们提出了一个正式的验证框架，将检测逻辑和攻击树建模为标记转移系统（LTS），通过互模拟和弱跟踪包含实现自动一致性检查。通用威胁检测语言（GTDL，我们在本文中形式化的通用检测语言）中指定的检测规则被分配了组合操作语义，并通过结构跟踪语义将表示为攻击树的威胁模型解释为LTS。这两种表示都被翻译为LNT，这是CADP工具箱支持的建模语言。这个通用的语义域能够系统化、自动化地验证检测覆盖范围。我们评估我们针对LokiBot和Objetet等现实世界恶意软件场景的方法，并通过参数合成模型提供可扩展性分析。结果证实，我们的方法可以识别威胁模型和检测规则之间的语义不匹配，支持迭代细化，并扩展到现实的威胁格局。



## **6. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

语音认证和反欺骗系统威胁调查 cs.CR

This paper is submitted to the IEEE IoT Journal

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.16843v4) [paper-pdf](http://arxiv.org/pdf/2508.16843v4)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.

摘要: 语音认证发生了重大变化，从依赖手工声学特征的传统系统到可以提取稳健的说话者嵌入的深度学习模型。这一进步扩大了其在金融、智能设备、执法等领域的应用。然而，随着采用率的增加，威胁也随之增加。本调查全面回顾了针对语音认证系统（PAS）和反欺骗对策（CM）的现代威胁格局，包括数据中毒、对抗性、深度伪造和对抗性欺骗攻击。我们按时间顺序追踪语音认证的发展，并研究漏洞如何随着技术进步而演变。对于每种类型的攻击，我们总结了方法论，强调常用的数据集，比较性能和局限性，并使用广泛接受的分类法组织现有文献。通过强调新出现的风险和公开挑战，本调查旨在支持开发更安全、更有弹性的语音认证系统。



## **7. Sy-FAR: Symmetry-based Fair Adversarial Robustness**

Sy-FAR：基于对称性的公平对抗鲁棒性 cs.LG

20 pages, 11 figures

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12939v1) [paper-pdf](http://arxiv.org/pdf/2509.12939v1)

**Authors**: Haneen Najjar, Eyal Ronen, Mahmood Sharif

**Abstract**: Security-critical machine-learning (ML) systems, such as face-recognition systems, are susceptible to adversarial examples, including real-world physically realizable attacks. Various means to boost ML's adversarial robustness have been proposed; however, they typically induce unfair robustness: It is often easier to attack from certain classes or groups than from others. Several techniques have been developed to improve adversarial robustness while seeking perfect fairness between classes. Yet, prior work has focused on settings where security and fairness are less critical. Our insight is that achieving perfect parity in realistic fairness-critical tasks, such as face recognition, is often infeasible -- some classes may be highly similar, leading to more misclassifications between them. Instead, we suggest that seeking symmetry -- i.e., attacks from class $i$ to $j$ would be as successful as from $j$ to $i$ -- is more tractable. Intuitively, symmetry is a desirable because class resemblance is a symmetric relation in most domains. Additionally, as we prove theoretically, symmetry between individuals induces symmetry between any set of sub-groups, in contrast to other fairness notions where group-fairness is often elusive. We develop Sy-FAR, a technique to encourage symmetry while also optimizing adversarial robustness and extensively evaluate it using five datasets, with three model architectures, including against targeted and untargeted realistic attacks. The results show Sy-FAR significantly improves fair adversarial robustness compared to state-of-the-art methods. Moreover, we find that Sy-FAR is faster and more consistent across runs. Notably, Sy-FAR also ameliorates another type of unfairness we discover in this work -- target classes that adversarial examples are likely to be classified into become significantly less vulnerable after inducing symmetry.

摘要: 面部识别系统等对安全至关重要的机器学习（ML）系统容易受到对抗性示例的影响，包括现实世界的物理可实现攻击。人们提出了各种提高ML对抗鲁棒性的方法;然而，它们通常会导致不公平的鲁棒性：从某些类或组进行攻击通常比从其他类或组更容易。人们开发了多种技术来提高对抗鲁棒性，同时寻求类之间的完美公平性。然而，之前的工作重点是安全性和公平性不那么重要的环境。我们的见解是，在现实的公平性关键任务（例如面部识别）中实现完美对等通常是不可行的--某些类别可能高度相似，从而导致它们之间出现更多的错误分类。相反，我们建议寻求对称性--即，从类$i$到$j$的攻击与从$j$到$i$ -的攻击一样成功。直观地说，对称性是可取的，因为类相似性在大多数领域中都是对称关系。此外，正如我们从理论上证明的那样，个体之间的对称性会导致任何一组子群体之间的对称性，这与其他公平性概念相反，在这些公平性概念中，群体公平性往往是难以捉摸的。我们开发了Sy-FAR，这是一种鼓励对称性同时优化对抗鲁棒性的技术，并使用五个数据集，三个模型架构对其进行了广泛的评估，包括针对有针对性和无针对性的现实攻击。结果表明，与最先进的方法相比，Sy-FAR显着提高了公平对抗鲁棒性。此外，我们发现Sy-FAR在运行中更快，更一致。值得注意的是，Sy-FAR还改善了我们在这项工作中发现的另一种类型的不公平性--在诱导对称性后，对抗性示例可能被归类到的目标类别变得明显不那么脆弱。



## **8. Adversarial Prompt Distillation for Vision-Language Models**

视觉语言模型的对抗性提示蒸馏 cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.15244v3) [paper-pdf](http://arxiv.org/pdf/2411.15244v3)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.

摘要: 对比图像预训练（CLIP）等大型预训练视觉语言模型（VLM）已被证明容易受到对抗攻击，这引发了人们对其在自动驾驶和医疗诊断等安全关键应用中部署的担忧。对抗性提示调整（APT）是对预训练的VLM进行鲁棒化的一种有希望的方法，它在提示调整的过程中应用对抗性训练。然而，现有的APT方法大多是单模式方法，仅为视觉或文本模式设计提示，从而限制了其稳健性或清晰准确性的有效性。在这项工作中，我们提出了对抗性提示蒸馏（APT），这是一个双峰知识蒸馏框架，通过将APT与多模式知识转移集成来增强APT。APT优化视觉和文本模式的提示，同时从干净的预培训教师CLIP模型中提取知识。对多个基准数据集的广泛实验证明了我们的APT方法在对抗稳健性和精确性方面优于当前最先进的APT方法。APT的有效性也验证了使用非稳健教师来提高微调后的VLM的通用性和稳健性的可能性。



## **9. DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval**

迪夫哈希：通过针对深度哈希图像检索的扩散模型进行文本引导定向攻击 cs.IR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12824v1) [paper-pdf](http://arxiv.org/pdf/2509.12824v1)

**Authors**: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang

**Abstract**: Deep hashing models have been widely adopted to tackle the challenges of large-scale image retrieval. However, these approaches face serious security risks due to their vulnerability to adversarial examples. Despite the increasing exploration of targeted attacks on deep hashing models, existing approaches still suffer from a lack of multimodal guidance, reliance on labeling information and dependence on pixel-level operations for attacks. To address these limitations, we proposed DiffHash, a novel diffusion-based targeted attack for deep hashing. Unlike traditional pixel-based attacks that directly modify specific pixels and lack multimodal guidance, our approach focuses on optimizing the latent representations of images, guided by text information generated by a Large Language Model (LLM) for the target image. Furthermore, we designed a multi-space hash alignment network to align the high-dimension image space and text space to the low-dimension binary hash space. During reconstruction, we also incorporated text-guided attention mechanisms to refine adversarial examples, ensuring them aligned with the target semantics while maintaining visual plausibility. Extensive experiments have demonstrated that our method outperforms state-of-the-art (SOTA) targeted attack methods, achieving better black-box transferability and offering more excellent stability across datasets.

摘要: 深度哈希模型已被广泛采用来应对大规模图像检索的挑战。然而，由于这些方法容易受到对抗示例的影响，因此面临严重的安全风险。尽管人们越来越多地探索深度哈希模型的有针对性的攻击，但现有方法仍然缺乏多模式指导、依赖标签信息以及依赖像素级操作进行攻击。为了解决这些限制，我们提出了迪夫哈希，这是一种新型的基于扩散的深度哈希定向攻击。与直接修改特定像素且缺乏多模式指导的传统基于像素的攻击不同，我们的方法重点是优化图像的潜在表示，并由目标图像的大型语言模型（LLM）生成的文本信息指导。此外，我们设计了一个多空间哈希对齐网络，将多维图像空间和文本空间与低维二进制哈希空间对齐。在重建过程中，我们还结合了文本引导的注意力机制来完善对抗性示例，确保它们与目标语义保持一致，同时保持视觉可信性。大量实验表明，我们的方法优于最先进的（SOTA）定向攻击方法，实现了更好的黑匣子可转移性，并在数据集之间提供更出色的稳定性。



## **10. Gradient-Free Adversarial Purification with Diffusion Models**

采用扩散模型的无干扰对抗净化 cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2501.13336v2) [paper-pdf](http://arxiv.org/pdf/2501.13336v2)

**Authors**: Xuelong Dai, Dong Wang, Xiuzhen Cheng, Bin Xiao

**Abstract**: Adversarial training and adversarial purification are two widely used defense strategies for enhancing model robustness against adversarial attacks. However, adversarial training requires costly retraining, while adversarial purification often suffers from low efficiency. More critically, existing defenses are primarily designed under the perturbation-based adversarial threat model, which is ineffective against recently introduced unrestricted adversarial attacks. In this paper, we propose an effective and efficient defense framework that counters both perturbation-based and unrestricted adversarial attacks. Our approach is motivated by the observation that adversarial examples typically lie near the decision boundary and are highly sensitive to pixel-level perturbations. To address this, we introduce adversarial anti-aliasing, a preprocessing technique that mitigates adversarial noise by reducing the magnitude of pixel-level perturbations. In addition, we propose adversarial super-resolution, which leverages prior knowledge from clean datasets to benignly restore high-quality images from adversarially degraded ones. Unlike image synthesis methods that generate entirely new images, adversarial super-resolution focuses on image restoration, making it more suitable for purification. Importantly, both techniques require no additional training and are computationally efficient since they do not rely on gradient computations. To further improve robustness across diverse datasets, we introduce a contrastive learning-based adversarial deblurring fine-tuning method. By incorporating adversarial priors during fine-tuning on the target dataset, this method enhances purification effectiveness without the need to retrain diffusion models.

摘要: 对抗训练和对抗净化是两种广泛使用的防御策略，用于增强模型针对对抗攻击的稳健性。然而，对抗性训练需要昂贵的再培训，而对抗性净化往往效率低下。更关键的是，现有的防御系统主要是在基于扰动的对抗性威胁模型下设计的，该模型对最近引入的无限制对抗性攻击无效。在本文中，我们提出了一个有效且高效的防御框架，可以对抗基于扰动的和无限制的对抗性攻击。我们的方法的动机是这样一个观察：对抗性示例通常位于决策边界附近，并且对像素级扰动高度敏感。为了解决这个问题，我们引入了对抗性抗锯齿，这是一种预处理技术，通过减少像素级扰动的幅度来减轻对抗性噪音。此外，我们还提出了对抗性超分辨率，它利用来自干净数据集的先验知识，从对抗性退化的图像中良性恢复高质量图像。与生成全新图像的图像合成方法不同，对抗性超分辨率专注于图像恢复，使其更适合净化。重要的是，这两种技术都不需要额外的训练，并且计算效率高，因为它们不依赖于梯度计算。为了进一步提高不同数据集的稳健性，我们引入了一种基于对比学习的对抗去模糊微调方法。通过在对目标数据集进行微调期间纳入对抗先验，该方法增强了净化有效性，而无需重新训练扩散模型。



## **11. Defense-to-Attack: Bypassing Weak Defenses Enables Stronger Jailbreaks in Vision-Language Models**

防御到攻击：击败弱防御，在视觉语言模型中实现更强的越狱 cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12724v1) [paper-pdf](http://arxiv.org/pdf/2509.12724v1)

**Authors**: Yunhan Zhao, Xiang Zheng, Xingjun Ma

**Abstract**: Despite their superb capabilities, Vision-Language Models (VLMs) have been shown to be vulnerable to jailbreak attacks. While recent jailbreaks have achieved notable progress, their effectiveness and efficiency can still be improved. In this work, we reveal an interesting phenomenon: incorporating weak defense into the attack pipeline can significantly enhance both the effectiveness and the efficiency of jailbreaks on VLMs. Building on this insight, we propose Defense2Attack, a novel jailbreak method that bypasses the safety guardrails of VLMs by leveraging defensive patterns to guide jailbreak prompt design. Specifically, Defense2Attack consists of three key components: (1) a visual optimizer that embeds universal adversarial perturbations with affirmative and encouraging semantics; (2) a textual optimizer that refines the input using a defense-styled prompt; and (3) a red-team suffix generator that enhances the jailbreak through reinforcement fine-tuning. We empirically evaluate our method on four VLMs and four safety benchmarks. The results demonstrate that Defense2Attack achieves superior jailbreak performance in a single attempt, outperforming state-of-the-art attack methods that often require multiple tries. Our work offers a new perspective on jailbreaking VLMs.

摘要: 尽管视觉语言模型（VLM）具有出色的功能，但已被证明很容易受到越狱攻击。虽然最近的越狱取得了显着进展，但其有效性和效率仍有待提高。在这项工作中，我们揭示了一个有趣的现象：将弱防御纳入攻击管道中可以显着提高VLM越狱的有效性和效率。基于这一见解，我们提出了Defense 2Attack，这是一种新颖的越狱方法，通过利用防御模式来指导越狱提示设计，绕过了VLM的安全护栏。具体来说，Defense 2Attack由三个关键组件组成：（1）视觉优化器，它嵌入具有肯定和鼓励性语义的通用对抗性扰动;（2）文本优化器，它使用防御风格的提示来细化输入;（3）红队后缀生成器，它通过强化微调来增强越狱。我们根据四个VLM和四个安全基准对我们的方法进行了经验评估。结果表明，Defense 2Attack只需一次尝试即可实现卓越的越狱性能，优于通常需要多次尝试的最先进的攻击方法。我们的工作为越狱VLM提供了新的视角。



## **12. Revisiting Transferable Adversarial Images: Systemization, Evaluation, and New Insights**

重新审视可转移对抗图像：系统化、评估和新见解 cs.CR

TPAMI 2025. Code is available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2310.11850v2) [paper-pdf](http://arxiv.org/pdf/2310.11850v2)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes, Qi Li, Qian Wang, Chao Shen

**Abstract**: Transferable adversarial images raise critical security concerns for computer vision systems in real-world, black-box attack scenarios. Although many transfer attacks have been proposed, existing research lacks a systematic and comprehensive evaluation. In this paper, we systemize transfer attacks into five categories around the general machine learning pipeline and provide the first comprehensive evaluation, with 23 representative attacks against 11 representative defenses, including the recent, transfer-oriented defense and the real-world Google Cloud Vision. In particular, we identify two main problems of existing evaluations: (1) for attack transferability, lack of intra-category analyses with fair hyperparameter settings, and (2) for attack stealthiness, lack of diverse measures. Our evaluation results validate that these problems have indeed caused misleading conclusions and missing points, and addressing them leads to new, \textit{consensus-challenging} insights, such as (1) an early attack, DI, even outperforms all similar follow-up ones, (2) the state-of-the-art (white-box) defense, DiffPure, is even vulnerable to (black-box) transfer attacks, and (3) even under the same $L_p$ constraint, different attacks yield dramatically different stealthiness results regarding diverse imperceptibility metrics, finer-grained measures, and a user study. We hope that our analyses will serve as guidance on properly evaluating transferable adversarial images and advance the design of attacks and defenses. Code is available at https://github.com/ZhengyuZhao/TransferAttackEval.

摘要: 可传输的对抗图像在现实世界的黑匣子攻击场景中给计算机视觉系统带来了关键的安全问题。尽管已经提出了很多转移攻击，但现有的研究缺乏系统、全面的评估。在本文中，我们围绕通用机器学习管道将转移攻击系统化为五类，并提供了首次全面评估，其中包含针对11种代表性防御的23种代表性攻击，包括最近的面向转移的防御和现实世界的Google Cloud Vision。特别是，我们发现了现有评估的两个主要问题：（1）对于攻击可转移性，缺乏具有公平超参数设置的类别内分析，以及（2）对于攻击隐蔽性，缺乏多样化的措施。我们的评估结果证实，这些问题确实导致了误导性结论和缺失点，解决这些问题会带来新的、具有挑战性的见解，例如（1）早期攻击，DI，甚至优于所有类似的后续攻击，（2）最先进的（白盒）防御，DiffPure，甚至容易受到（黑盒）转移攻击，以及（3）即使在相同的$L_p$约束下，不同的攻击产生关于不同的不可感知性度量的显著不同的隐蔽性结果，更细粒度的测量和用户研究。我们希望我们的分析能够指导正确评估可转移的对抗图像并推进攻击和防御的设计。代码可在https://github.com/ZhengyuZhao/TransferAttackEval上获取。



## **13. SoK: How Sensor Attacks Disrupt Autonomous Vehicles: An End-to-end Analysis, Challenges, and Missed Threats**

SoK：传感器攻击如何扰乱自动驾驶车辆：端到端分析、挑战和错过的威胁 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11120v2) [paper-pdf](http://arxiv.org/pdf/2509.11120v2)

**Authors**: Qingzhao Zhang, Shaocheng Luo, Z. Morley Mao, Miroslav Pajic, Michael K. Reiter

**Abstract**: Autonomous vehicles, including self-driving cars, robotic ground vehicles, and drones, rely on complex sensor pipelines to ensure safe and reliable operation. However, these safety-critical systems remain vulnerable to adversarial sensor attacks that can compromise their performance and mission success. While extensive research has demonstrated various sensor attack techniques, critical gaps remain in understanding their feasibility in real-world, end-to-end systems. This gap largely stems from the lack of a systematic perspective on how sensor errors propagate through interconnected modules in autonomous systems when autonomous vehicles interact with the physical world.   To bridge this gap, we present a comprehensive survey of autonomous vehicle sensor attacks across platforms, sensor modalities, and attack methods. Central to our analysis is the System Error Propagation Graph (SEPG), a structured demonstration tool that illustrates how sensor attacks propagate through system pipelines, exposing the conditions and dependencies that determine attack feasibility. With the aid of SEPG, our study distills seven key findings that highlight the feasibility challenges of sensor attacks and uncovers eleven previously overlooked attack vectors exploiting inter-module interactions, several of which we validate through proof-of-concept experiments. Additionally, we demonstrate how large language models (LLMs) can automate aspects of SEPG construction and cross-validate expert analysis, showcasing the promise of AI-assisted security evaluation.

摘要: 自动驾驶汽车、机器人地面车辆和无人机等自动驾驶车辆依赖复杂的传感器管道来确保安全可靠的运行。然而，这些安全关键系统仍然容易受到对抗性传感器攻击，这可能会损害其性能和任务成功。虽然广泛的研究已经证明了各种传感器攻击技术，但在了解其在现实世界的端到端系统中的可行性方面仍然存在重大差距。这一差距很大程度上源于缺乏对自动驾驶汽车与物理世界互动时传感器误差如何通过自动驾驶系统中的互连模块传播的系统视角。   为了弥合这一差距，我们对跨平台、传感器模式和攻击方法的自动驾驶汽车传感器攻击进行了全面调查。我们分析的核心是系统错误传播图（SEPG），这是一种结构化演示工具，它说明了传感器攻击如何通过系统管道传播，揭示了决定攻击可行性的条件和依赖性。在SEPG的帮助下，我们的研究提炼了七个关键发现，这些发现凸显了传感器攻击的可行性挑战，并揭示了11个以前被忽视的利用模块间交互的攻击载体，其中一些我们通过概念验证实验进行了验证。此外，我们还展示了大型语言模型（LLM）如何自动化SEPG构建的各个方面和交叉验证专家分析，展示了人工智能辅助安全评估的前景。



## **14. Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content**

迈向包容性有毒内容审核：解决毒性分类器中对抗攻击的脆弱性，以应对LLM生成的内容 cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12672v1) [paper-pdf](http://arxiv.org/pdf/2509.12672v1)

**Authors**: Shaz Furniturewala, Arkaitz Zubiaga

**Abstract**: The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.

摘要: 由于大型语言模型（LLM）的广泛使用，在线机器生成内容的数量急剧增长，这给内容审核系统带来了新的挑战。传统的内容审核分类器通常根据人类生成的文本进行训练，但由于LLM生成的文本偏离其训练数据以及旨在避免检测的对抗性攻击而遭受错误分类。当今的防御策略是被动的，而不是主动的，因为它们依赖于对抗训练或外部检测模型来识别攻击。在这项工作中，我们的目标是识别毒性分类器中导致错误分类的脆弱组件，提出一种基于机械解释性技术的新型策略。我们的研究重点是微调的BERT和RoBERTa分类器，对跨越各种少数群体的不同数据集进行测试。我们使用对抗攻击技术来识别脆弱的电路。最后，我们抑制了这些脆弱的电路，提高了对抗攻击的性能。我们还提供了对这些脆弱电路的人口统计学层面的见解，揭示了模型训练中的公平性和稳健性差距。我们发现模型具有不同的头部，这些头部要么对性能至关重要，要么容易受到攻击，而抑制脆弱的头部可以提高对抗性输入的性能。我们还发现，不同的头部导致了不同人口群体的脆弱性，这可以为毒性检测模型的更具包容性的开发提供信息。



## **15. CIARD: Cyclic Iterative Adversarial Robustness Distillation**

CIARD：循环迭代对抗稳健蒸馏 cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12633v1) [paper-pdf](http://arxiv.org/pdf/2509.12633v1)

**Authors**: Liming Lu, Shuchao Pang, Xu Zheng, Xiang Gu, Anan Du, Yunhuai Liu, Yongbin Zhou

**Abstract**: Adversarial robustness distillation (ARD) aims to transfer both performance and robustness from teacher model to lightweight student model, enabling resilient performance on resource-constrained scenarios. Though existing ARD approaches enhance student model's robustness, the inevitable by-product leads to the degraded performance on clean examples. We summarize the causes of this problem inherent in existing methods with dual-teacher framework as: 1. The divergent optimization objectives of dual-teacher models, i.e., the clean and robust teachers, impede effective knowledge transfer to the student model, and 2. The iteratively generated adversarial examples during training lead to performance deterioration of the robust teacher model. To address these challenges, we propose a novel Cyclic Iterative ARD (CIARD) method with two key innovations: a. A multi-teacher framework with contrastive push-loss alignment to resolve conflicts in dual-teacher optimization objectives, and b. Continuous adversarial retraining to maintain dynamic teacher robustness against performance degradation from the varying adversarial examples. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrate that CIARD achieves remarkable performance with an average 3.53 improvement in adversarial defense rates across various attack scenarios and a 5.87 increase in clean sample accuracy, establishing a new benchmark for balancing model robustness and generalization. Our code is available at https://github.com/eminentgu/CIARD

摘要: 对抗稳健性蒸馏（ARD）旨在将性能和稳健性从教师模型转移到轻量级学生模型，从而在资源受限的场景中实现弹性性能。尽管现有的ARD方法增强了学生模型的稳健性，但不可避免的副产品导致干净示例的性能下降。我们将现有方法中固有的双教师框架中存在的这个问题的原因总结为：1.双教师模型的不同优化目标，即干净而强大的教师阻碍了知识向学生模式的有效转移，2.训练期间迭代生成的对抗性示例导致稳健教师模型的性能恶化。为了应对这些挑战，我们提出了一种新型的循环迭代ARD（CIARD）方法，具有两个关键创新：a.具有对比推-损失对齐的多教师框架，以解决双教师优化目标中的冲突，以及b。持续的对抗性再培训，以保持动态教师鲁棒性，防止不同对抗性示例带来的绩效下降。CIFAR-10、CIFAR-100和Tiny-ImageNet上的大量实验表明，CIARD实现了非凡的性能，在各种攻击场景中的对抗防御率平均提高了3.53，干净样本准确性提高了5.87，为平衡模型稳健性和概括性建立了新的基准。我们的代码可在https://github.com/eminentgu/CIARD上获取



## **16. Your Compiler is Backdooring Your Model: Understanding and Exploiting Compilation Inconsistency Vulnerabilities in Deep Learning Compilers**

您的编译器正在为您的模型做后门：了解和利用深度学习编译器中的编译不一致漏洞 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11173v2) [paper-pdf](http://arxiv.org/pdf/2509.11173v2)

**Authors**: Simin Chen, Jinjun Peng, Yixin He, Junfeng Yang, Baishakhi Ray

**Abstract**: Deep learning (DL) compilers are core infrastructure in modern DL systems, offering flexibility and scalability beyond vendor-specific libraries. This work uncovers a fundamental vulnerability in their design: can an official, unmodified compiler alter a model's semantics during compilation and introduce hidden backdoors? We study both adversarial and natural settings. In the adversarial case, we craft benign models where triggers have no effect pre-compilation but become effective backdoors after compilation. Tested on six models, three commercial compilers, and two hardware platforms, our attack yields 100% success on triggered inputs while preserving normal accuracy and remaining undetected by state-of-the-art detectors. The attack generalizes across compilers, hardware, and floating-point settings. In the natural setting, we analyze the top 100 HuggingFace models (including one with 220M+ downloads) and find natural triggers in 31 models. This shows that compilers can introduce risks even without adversarial manipulation.   Our results reveal an overlooked threat: unmodified DL compilers can silently alter model semantics. To our knowledge, this is the first work to expose inherent security risks in DL compiler design, opening a new direction for secure and trustworthy ML.

摘要: 深度学习（DL）编译器是现代DL系统的核心基础设施，提供超出供应商特定库的灵活性和可扩展性。这项工作揭示了他们设计中的一个根本漏洞：官方的、未经修改的编译器能否在编译期间改变模型的语义并引入隐藏的后门？我们研究对抗环境和自然环境。在对抗性的情况下，我们构建了良性模型，其中触发器对预编译没有影响，但在编译后成为有效的后门。经过六种型号、三种商业编译器和两个硬件平台的测试，我们的攻击在触发的输入上取得了100%的成功，同时保持正常的准确性并保持未被最先进的检测器检测到。该攻击跨越编译器、硬件和浮点设置进行推广。在自然环境中，我们分析了排名前100的HuggingFace模型（包括下载量超过2.2亿的模型），并在31个模型中找到自然触发因素。这表明，即使没有对抗性操纵，编译器也可能引入风险。   我们的结果揭示了一个被忽视的威胁：未修改的DL编译器可以悄悄改变模型语义。据我们所知，这是第一个暴露DL编译器设计中固有安全风险的工作，为安全和可信的ML开辟了新的方向。



## **17. DisorientLiDAR: Physical Attacks on LiDAR-based Localization**

DisorientLiDART：对基于LiDART的本地化的物理攻击 cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12595v1) [paper-pdf](http://arxiv.org/pdf/2509.12595v1)

**Authors**: Yizhen Lao, Yu Zhang, Ziting Wang, Chengbo Wang, Yifei Xue, Wanpeng Shao

**Abstract**: Deep learning models have been shown to be susceptible to adversarial attacks with visually imperceptible perturbations. Even this poses a serious security challenge for the localization of self-driving cars, there has been very little exploration of attack on it, as most of adversarial attacks have been applied to 3D perception. In this work, we propose a novel adversarial attack framework called DisorientLiDAR targeting LiDAR-based localization. By reverse-engineering localization models (e.g., feature extraction networks), adversaries can identify critical keypoints and strategically remove them, thereby disrupting LiDAR-based localization. Our proposal is first evaluated on three state-of-the-art point-cloud registration models (HRegNet, D3Feat, and GeoTransformer) using the KITTI dataset. Experimental results demonstrate that removing regions containing Top-K keypoints significantly degrades their registration accuracy. We further validate the attack's impact on the Autoware autonomous driving platform, where hiding merely a few critical regions induces noticeable localization drift. Finally, we extended our attacks to the physical world by hiding critical regions with near-infrared absorptive materials, thereby successfully replicate the attack effects observed in KITTI data. This step has been closer toward the realistic physical-world attack that demonstrate the veracity and generality of our proposal.

摘要: 深度学习模型已被证明容易受到具有视觉上难以感知的干扰的对抗攻击。即使这对自动驾驶汽车的本地化构成了严重的安全挑战，但对其攻击的探索却很少，因为大多数对抗性攻击都应用于3D感知。在这项工作中，我们提出了一种名为DisorientLiDART的新型对抗攻击框架，目标是基于LiDART的本地化。通过反向工程本地化模型（例如，特征提取网络），对手可以识别关键关键点并从战略上删除它们，从而破坏基于激光雷达的定位。我们的建议首先使用KITTI数据集在三个最先进的点云配准模型（HRegNet，D3 Feat和GeoTransformer）上进行评估。实验结果表明，删除包含Top-K关键点的区域显着降低其配准精度。我们进一步验证了攻击对Autoware自动驾驶平台的影响，在该平台上，仅隐藏几个关键区域会引起明显的定位漂移。最后，我们通过用近红外吸收材料隐藏关键区域，将攻击扩展到物理世界，从而成功复制了KITTI数据中观察到的攻击效果。这一步更接近现实的物理世界攻击，证明了我们提案的真实性和普遍性。



## **18. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

EmantSleuth：通过语义意图不变性检测提示注入 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.20890v2) [paper-pdf](http://arxiv.org/pdf/2508.20890v2)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.

摘要: 大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，从虚拟助手到自治代理。然而，它们的灵活性也引入了新的攻击向量，特别是提示注入（PI），其中攻击者通过精心制作的输入操纵模型行为。随着攻击者不断地使用释义、混淆甚至多任务注入策略，现有的基准不再足以捕获所有新兴威胁。   为了解决这一差距，我们构建了一个新的基准，系统地扩展了以前的努力。我们的基准涵盖了两种广泛使用的现有基准，同时引入了新的操纵技术和多任务场景，从而提供了更全面的评估设置。我们发现，现有的防御虽然在原始基准上有效，但在我们的基准下表现出明显的弱点，这凸显了对更强大解决方案的需求。我们的关键见解是，虽然攻击形式可能会有所不同，但对手的意图（注入未经授权的任务）保持不变。在这一观察的基础上，我们提出了EmittSleuth，这是一个面向语义的防御框架，它通过对任务级意图而不是表面特征进行推理来检测提示注入。在最先进的基准测试中进行评估后，AktSleuth始终优于现有的防御，同时保持相当的运行时间和成本效率。这些结果表明，基于意图的语义推理提供了一个强大的，有效的，和可推广的策略，以抵御不断发展的即时注入威胁的LLM。



## **19. Exploiting Timing Side-Channels in Quantum Circuits Simulation Via ML-Based Methods**

通过基于ML的方法在量子电路模拟中利用定时边通道 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12535v1) [paper-pdf](http://arxiv.org/pdf/2509.12535v1)

**Authors**: Ben Dong, Hui Feng, Qian Wang

**Abstract**: As quantum computing advances, quantum circuit simulators serve as critical tools to bridge the current gap caused by limited quantum hardware availability. These simulators are typically deployed on cloud platforms, where users submit proprietary circuit designs for simulation. In this work, we demonstrate a novel timing side-channel attack targeting cloud- based quantum simulators. A co-located malicious process can observe fine-grained execution timing patterns to extract sensitive information about concurrently running quantum circuits. We systematically analyze simulator behavior using the QASMBench benchmark suite, profiling timing and memory characteristics across various circuit executions. Our experimental results show that timing profiles exhibit circuit-dependent patterns that can be effectively classified using pattern recognition techniques, enabling the adversary to infer circuit identities and compromise user confidentiality. We were able to achieve 88% to 99.9% identification rate of quantum circuits based on different datasets. This work highlights previously unexplored security risks in quantum simulation environments and calls for stronger isolation mechanisms to protect user workloads

摘要: 随着量子计算的进步，量子电路模拟器成为弥合当前量子硬件可用性有限造成的差距的关键工具。这些模拟器通常部署在云平台上，用户在云平台上提交专有电路设计进行模拟。在这项工作中，我们展示了一种针对基于云的量子模拟器的新型定时侧通道攻击。位于同一位置的恶意进程可以观察细粒度的执行计时模式，以提取有关并发运行的量子电路的敏感信息。我们使用QASMBench基准套件系统地分析模拟器行为，分析各种电路执行中的计时和内存特征。我们的实验结果表明，时间分布呈现出与电路相关的模式，可以使用模式识别技术有效分类，使对手能够推断电路身份并损害用户的机密性。我们能够基于不同数据集实现88%至99.9%的量子电路识别率。这项工作强调了量子模拟环境中之前未探索的安全风险，并呼吁更强大的隔离机制来保护用户工作负载



## **20. How to Beat Nakamoto in the Race**

如何在比赛中击败中本聪 cs.CR

To be presented at the 2025 ACM Conference on Computer and  Communications Security (CCS)

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2508.16202v2) [paper-pdf](http://arxiv.org/pdf/2508.16202v2)

**Authors**: Shu-Jie Cao, Dongning Guo

**Abstract**: This paper studies proof-of-work Nakamoto consensus protocols under bounded network delays, settling two long-standing questions in blockchain security: What is the most effective attack on block safety under a given block confirmation latency? And what is the resulting probability of safety violation? A Markov decision process (MDP) framework is introduced to precisely characterize the system state (including the blocktree and timings of all blocks mined), the adversary's potential actions, and the state transitions due to the adversarial action and the random block arrival processes. An optimal attack, called bait-and-switch, is proposed and proved to maximize the adversary's chance of violating block safety by "beating Nakamoto in the race". The exact probability of this violation is calculated for any given confirmation depth using Markov chain analysis, offering fresh insights into the interplay of network delay, confirmation rules, and blockchain security.

摘要: 本文研究了有限网络延迟下的工作量证明中本聪共识协议，解决了区块链安全中的两个长期存在的问题：在给定的块确认延迟下，对块安全性的最有效攻击是什么？由此产生的安全违规可能性是多少？引入了马尔科夫决策过程（MDP）框架来精确描述系统状态（包括区块树和所有挖掘的块的时间）、对手的潜在行为以及由于对抗行为和随机块到达过程而导致的状态转变。提出并证明了一种称为诱饵和开关的最佳攻击，可以通过“在比赛中击败中本聪”来最大化对手违反区块安全的机会。使用马尔科夫链分析针对任何给定的确认深度计算这种违规的确切概率，为网络延迟、确认规则和区块链安全性的相互作用提供了新的见解。



## **21. Time-Constrained Intelligent Adversaries for Automation Vulnerability Testing: A Multi-Robot Patrol Case Study**

用于自动化漏洞测试的时间约束智能对手：多机器人巡逻案例研究 cs.RO

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11971v1) [paper-pdf](http://arxiv.org/pdf/2509.11971v1)

**Authors**: James C. Ward, Alex Bott, Connor York, Edmund R. Hunt

**Abstract**: Simulating hostile attacks of physical autonomous systems can be a useful tool to examine their robustness to attack and inform vulnerability-aware design. In this work, we examine this through the lens of multi-robot patrol, by presenting a machine learning-based adversary model that observes robot patrol behavior in order to attempt to gain undetected access to a secure environment within a limited time duration. Such a model allows for evaluation of a patrol system against a realistic potential adversary, offering insight into future patrol strategy design. We show that our new model outperforms existing baselines, thus providing a more stringent test, and examine its performance against multiple leading decentralized multi-robot patrol strategies.

摘要: 模拟物理自治系统的敌对攻击可以成为检查其对攻击的稳健性并为可预见性设计提供信息的有用工具。在这项工作中，我们通过多机器人巡逻的视角来检查这一点，提出了一个基于机器学习的对手模型，该模型观察机器人巡逻行为，以尝试在有限的时间内获得对安全环境的未被发现的访问。这样的模型允许针对现实的潜在对手评估巡逻系统，从而深入了解未来的巡逻策略设计。我们表明，我们的新模型优于现有的基线，从而提供了更严格的测试，并针对多种领先的去中心化多机器人巡逻策略检查了其性能。



## **22. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

NeuronStrike：对对齐的LLM的神经元级攻击 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11864v1) [paper-pdf](http://arxiv.org/pdf/2509.11864v1)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.

摘要: 安全一致对于大型语言模型（LLM）的道德部署至关重要，指导它们避免生成有害或不道德的内容。当前的对齐技术，例如有监督的微调和来自人类反馈的强化学习，仍然很脆弱，可以被精心设计的对抗提示绕过。不幸的是，此类攻击依赖于试错，缺乏跨模型的通用性，并且受到可扩展性和可靠性的限制。   本文介绍了NeuroStrike，这是一种新颖且可推广的攻击框架，它利用了对齐技术引入的一个基本漏洞：依赖于负责检测和抑制有害输入的稀疏、专门的安全神经元。我们将NeuroStrike应用于白盒和黑盒设置：在白盒设置中，NeuroStrike通过反馈激活分析识别安全神经元，并在推理期间修剪它们以禁用安全机制。在黑匣子环境中，我们提出了第一次LLM剖析攻击，该攻击通过在开权重代理模型上训练对抗提示生成器，然后将它们部署到黑匣子和专有目标上来利用安全神经元的可移植性。我们对来自主要LLM开发商的20多个开量级LLM进行了评估。通过删除目标层中不到0.6%的神经元，NeuroStrike仅使用普通恶意提示即可实现76.9%的平均攻击成功率（ASB）。此外，Neurostrike将四种多模式LLM推广到对不安全图像输入具有100%的ASB。安全神经元在架构之间有效转移，使11个微调模型的ASB达到78.5%，5个提炼模型的ASB达到77.7%。黑匣子LLM分析攻击在包括Google Gemini系列在内的五种黑匣子型号中实现了63.7%的平均ASB。



## **23. A Practical Adversarial Attack against Sequence-based Deep Learning Malware Classifiers**

针对基于序列的深度学习恶意软件分类器的实用对抗攻击 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11836v1) [paper-pdf](http://arxiv.org/pdf/2509.11836v1)

**Authors**: Kai Tan, Dongyang Zhan, Lin Ye, Hongli Zhang, Binxing Fang

**Abstract**: Sequence-based deep learning models (e.g., RNNs), can detect malware by analyzing its behavioral sequences. Meanwhile, these models are susceptible to adversarial attacks. Attackers can create adversarial samples that alter the sequence characteristics of behavior sequences to deceive malware classifiers. The existing methods for generating adversarial samples typically involve deleting or replacing crucial behaviors in the original data sequences, or inserting benign behaviors that may violate the behavior constraints. However, these methods that directly manipulate sequences make adversarial samples difficult to implement or apply in practice. In this paper, we propose an adversarial attack approach based on Deep Q-Network and a heuristic backtracking search strategy, which can generate perturbation sequences that satisfy practical conditions for successful attacks. Subsequently, we utilize a novel transformation approach that maps modifications back to the source code, thereby avoiding the need to directly modify the behavior log sequences. We conduct an evaluation of our approach, and the results confirm its effectiveness in generating adversarial samples from real-world malware behavior sequences, which have a high success rate in evading anomaly detection models. Furthermore, our approach is practical and can generate adversarial samples while maintaining the functionality of the modified software.

摘要: 基于序列的深度学习模型（例如，RNN），可以通过分析恶意软件的行为序列来检测恶意软件。与此同时，这些模型很容易受到对抗攻击。攻击者可以创建对抗样本，这些样本改变行为序列的序列特征，以欺骗恶意软件分类器。生成对抗样本的现有方法通常涉及删除或替换原始数据序列中的关键行为，或者插入可能违反行为约束的良性行为。然而，这些直接操纵序列的方法使得对抗性样本难以在实践中实现或应用。本文提出了一种基于Deep Q网络的对抗性攻击方法和启发式回溯搜索策略，该策略可以生成满足成功攻击实际条件的扰动序列。随后，我们利用一种新颖的转换方法，将修改映射回源代码，从而避免了直接修改行为日志序列的需要。我们对我们的方法进行了评估，结果证实了它在从现实世界的恶意软件行为序列生成对抗样本方面的有效性，这些样本在逃避异常检测模型方面具有很高的成功率。此外，我们的方法很实用，可以生成对抗样本，同时保持修改后软件的功能。



## **24. Removal Attack and Defense on AI-generated Content Latent-based Watermarking**

对人工智能生成的内容基于潜伏的水印的删除攻击和防御 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11745v1) [paper-pdf](http://arxiv.org/pdf/2509.11745v1)

**Authors**: De Zhang Lee, Han Fang, Hanyi Wang, Ee-Chien Chang

**Abstract**: Digital watermarks can be embedded into AI-generated content (AIGC) by initializing the generation process with starting points sampled from a secret distribution. When combined with pseudorandom error-correcting codes, such watermarked outputs can remain indistinguishable from unwatermarked objects, while maintaining robustness under whitenoise. In this paper, we go beyond indistinguishability and investigate security under removal attacks. We demonstrate that indistinguishability alone does not necessarily guarantee resistance to adversarial removal. Specifically, we propose a novel attack that exploits boundary information leaked by the locations of watermarked objects. This attack significantly reduces the distortion required to remove watermarks -- by up to a factor of $15 \times$ compared to a baseline whitenoise attack under certain settings. To mitigate such attacks, we introduce a defense mechanism that applies a secret transformation to hide the boundary, and prove that the secret transformation effectively rendering any attacker's perturbations equivalent to those of a naive whitenoise adversary. Our empirical evaluations, conducted on multiple versions of Stable Diffusion, validate the effectiveness of both the attack and the proposed defense, highlighting the importance of addressing boundary leakage in latent-based watermarking schemes.

摘要: 通过使用从秘密分发中采样的起点初始化生成过程，数字水印可以嵌入到人工智能生成的内容（AIGC）中。当与伪随机错误纠正码结合时，此类带水印的输出可以与未带水印的对象保持不可区分，同时在白噪音下保持鲁棒性。在本文中，我们超越了不可撤销性，并调查了删除攻击下的安全性。我们证明，仅靠不可撤销性并不一定保证对对抗性清除的抵抗。具体来说，我们提出了一种新颖的攻击，该攻击利用带有水印的对象位置泄露的边界信息。与某些设置下的基线白噪音攻击相比，此攻击显着减少了删除水印所需的失真，最多可减少15美元\x $。为了减轻此类攻击，我们引入了一种防御机制，该机制应用秘密转换来隐藏边界，并证明秘密转换有效地使任何攻击者的扰动等效于天真白噪音对手的扰动。我们对多个版本的稳定扩散进行了经验评估，验证了攻击和拟议防御的有效性，强调了解决基于潜伏的水印方案中边界泄漏的重要性。



## **25. DARD: Dice Adversarial Robustness Distillation against Adversarial Attacks**

DARD：对抗性攻击的对抗鲁棒性蒸馏 cs.LG

Accepted at SecureComm 2025, 15 pages, 4 figures

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11525v1) [paper-pdf](http://arxiv.org/pdf/2509.11525v1)

**Authors**: Jing Zou, Shungeng Zhang, Meikang Qiu, Chong Li

**Abstract**: Deep learning models are vulnerable to adversarial examples, posing critical security challenges in real-world applications. While Adversarial Training (AT ) is a widely adopted defense mechanism to enhance robustness, it often incurs a trade-off by degrading performance on unperturbed, natural data. Recent efforts have highlighted that larger models exhibit enhanced robustness over their smaller counterparts. In this paper, we empirically demonstrate that such robustness can be systematically distilled from large teacher models into compact student models. To achieve better performance, we introduce Dice Adversarial Robustness Distillation (DARD), a novel method designed to transfer robustness through a tailored knowledge distillation paradigm. Additionally, we propose Dice Projected Gradient Descent (DPGD), an adversarial example generalization method optimized for effective attack. Our extensive experiments demonstrate that the DARD approach consistently outperforms adversarially trained networks with the same architecture, achieving superior robustness and standard accuracy.

摘要: 深度学习模型容易受到对抗性示例的影响，从而在现实世界的应用程序中构成了关键的安全挑战。虽然对抗训练（AT）是一种广泛采用的增强稳健性的防御机制，但它经常会降低未受干扰的自然数据的性能，从而导致权衡。最近的努力强调，较大的模型比较小的模型表现出更强的稳健性。在本文中，我们经验证明，这种稳健性可以从大型教师模型系统地提炼到紧凑的学生模型中。为了实现更好的性能，我们引入了骰子对抗鲁棒性蒸馏（DARD），这是一种新颖的方法，旨在通过定制的知识蒸馏范式转移鲁棒性。此外，我们还提出了骰子投影梯度下降（DPVD），这是一种针对有效攻击而优化的对抗性示例概括方法。我们广泛的实验表明，DARD方法始终优于具有相同架构的对抗训练网络，从而实现了卓越的鲁棒性和标准准确性。



## **26. Pulse-to-Circuit Characterization of Stealthy Crosstalk Attack on Multi-Tenant Superconducting Quantum Hardware**

多租户超导量子硬件隐形串话攻击的脉冲对电路特征 quant-ph

Will appear in the Proceedings of the 2025 Quantum Security and  Privacy Workshop (QSec '25)

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11407v1) [paper-pdf](http://arxiv.org/pdf/2509.11407v1)

**Authors**: Syed Emad Uddin Shubha, Tasnuva Farheen

**Abstract**: Hardware crosstalk in multi-tenant superconducting quantum computers constitutes a significant security threat, enabling adversaries to inject targeted errors across tenant boundaries. We present the first end-to-end framework for mapping physical pulse-level attacks to interpretable logical error channels, integrating density-matrix simulation, quantum process tomography (QPT), and a novel isometry-based circuit extraction method. Our pipeline reconstructs the complete induced error channel and fits an effective logical circuit model, revealing a fundamentally asymmetric attack mechanism: one adversarial qubit acts as a driver to set the induced logical rotation, while a second, the catalyst, refines the attack's coherence. Demonstrated on a linear three-qubit system, our approach shows that such attacks can significantly disrupt diverse quantum protocols, sometimes reducing accuracy to random guessing, while remaining effective and stealthy even under realistic hardware parameter variations. We further propose a protocol-level detection strategy based on observable attack signatures, showing that stealthy attacks can be exposed through targeted monitoring and providing a foundation for future defense-in-depth in quantum cloud platforms.

摘要: 多租户高温量子计算机中的硬件串烧构成了重大的安全威胁，使对手能够跨越租户边界注入有针对性的错误。我们提出了第一个将物理脉冲级攻击映射到可解释的逻辑错误通道的端到端框架，集成了密度矩阵模拟、量子过程断层扫描（QPT）和新型的基于等距的电路提取方法。我们的管道重建了完整的诱导错误通道，并符合有效的逻辑电路模型，揭示了一种根本不对称的攻击机制：一个对抗量子位充当设置诱导逻辑旋转的驱动器，而第二个量子位（催化剂）则细化了攻击的一致性。在线性三量子位系统上进行了演示，我们的方法表明，此类攻击可以显着破坏各种量子协议，有时会降低随机猜测的准确性，同时即使在现实的硬件参数变化下也保持有效和隐蔽性。我们进一步提出了一种基于可观察攻击特征的协议级检测策略，表明可以通过有针对性的监控来暴露隐形攻击，并为未来量子云平台的深度防御提供基础。



## **27. From Firewalls to Frontiers: AI Red-Teaming is a Domain-Specific Evolution of Cyber Red-Teaming**

从防火墙到前沿：人工智能红色团队是网络红色团队的特定领域进化 cs.LG

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11398v1) [paper-pdf](http://arxiv.org/pdf/2509.11398v1)

**Authors**: Anusha Sinha, Keltin Grimes, James Lucassen, Michael Feffer, Nathan VanHoudnos, Zhiwei Steven Wu, Hoda Heidari

**Abstract**: A red team simulates adversary attacks to help defenders find effective strategies to defend their systems in a real-world operational setting. As more enterprise systems adopt AI, red-teaming will need to evolve to address the unique vulnerabilities and risks posed by AI systems. We take the position that AI systems can be more effectively red-teamed if AI red-teaming is recognized as a domain-specific evolution of cyber red-teaming. Specifically, we argue that existing Cyber Red Teams who adopt this framing will be able to better evaluate systems with AI components by recognizing that AI poses new risks, has new failure modes to exploit, and often contains unpatchable bugs that re-prioritize disclosure and mitigation strategies. Similarly, adopting a cybersecurity framing will allow existing AI Red Teams to leverage a well-tested structure to emulate realistic adversaries, promote mutual accountability with formal rules of engagement, and provide a pattern to mature the tooling necessary for repeatable, scalable engagements. In these ways, the merging of AI and Cyber Red Teams will create a robust security ecosystem and best position the community to adapt to the rapidly changing threat landscape.

摘要: 红队模拟对手攻击，以帮助防御者找到有效的策略来在现实世界的操作环境中保护其系统。随着越来越多的企业系统采用人工智能，红色团队需要发展以解决人工智能系统带来的独特漏洞和风险。我们的立场是，如果人工智能红色团队被认为是网络红色团队的特定领域演变，那么人工智能系统就可以更有效地进行红色团队。具体来说，我们认为，采用这种框架的现有网红团队将能够通过认识到人工智能带来新的风险、有新的故障模式可供利用，并且通常包含无法修补的错误来更好地评估包含人工智能组件的系统，这些错误重新优先考虑披露和缓解策略。同样，采用网络安全框架将使现有的人工智能红团队能够利用经过良好测试的结构来模拟现实的对手，通过正式的参与规则促进相互问责，并提供一种模式来成熟可重复、可扩展的参与所需的工具。通过这些方式，人工智能和网络红团队的合并将创建一个强大的安全生态系统，并使社区能够最好地适应快速变化的威胁格局。



## **28. On the Escaping Efficiency of Distributed Adversarial Training Algorithms**

分布式对抗训练算法的逃避效率研究 cs.LG

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11337v1) [paper-pdf](http://arxiv.org/pdf/2509.11337v1)

**Authors**: Ying Cao, Kun Yuan, Ali H. Sayed

**Abstract**: Adversarial training has been widely studied in recent years due to its role in improving model robustness against adversarial attacks. This paper focuses on comparing different distributed adversarial training algorithms--including centralized and decentralized strategies--within multi-agent learning environments. Previous studies have highlighted the importance of model flatness in determining robustness. To this end, we develop a general theoretical framework to study the escaping efficiency of these algorithms from local minima, which is closely related to the flatness of the resulting models. We show that when the perturbation bound is sufficiently small (i.e., when the attack strength is relatively mild) and a large batch size is used, decentralized adversarial training algorithms--including consensus and diffusion--are guaranteed to escape faster from local minima than the centralized strategy, thereby favoring flatter minima. However, as the perturbation bound increases, this trend may no longer hold. In the simulation results, we illustrate our theoretical findings and systematically compare the performance of models obtained through decentralized and centralized adversarial training algorithms. The results highlight the potential of decentralized strategies to enhance the robustness of models in distributed settings.

摘要: 近年来，对抗训练因其在提高模型针对对抗攻击的稳健性方面所发挥的作用而受到广泛研究。本文重点比较多代理学习环境中不同的分布式对抗训练算法（包括集中式和分散式策略）。之前的研究强调了模型平坦性在确定稳健性方面的重要性。为此，我们开发了一个通用理论框架来研究这些算法摆脱局部极小值的效率，这与所得模型的平坦性密切相关。我们表明，当扰动界足够小时（即，当攻击强度相对较弱时）并且使用大批量时，去中心化对抗训练算法（包括共识和扩散）保证比集中式策略更快地摆脱局部极小值，从而有利于更平坦的极小值。然而，随着扰动界限的增加，这种趋势可能不再保持。在模拟结果中，我们说明了我们的理论发现，并系统地比较了通过分散式和集中式对抗训练算法获得的模型的性能。结果强调了去中心化策略在增强分布式环境中模型稳健性方面的潜力。



## **29. ANROT-HELANet: Adverserially and Naturally Robust Attention-Based Aggregation Network via The Hellinger Distance for Few-Shot Classification**

ANROT-HELANet：通过Hellinger距离连续、自然地扩展基于注意力的聚合网络，用于少镜头分类 cs.CV

Preprint version. The manuscript has been submitted to a journal. All  changes will be transferred to the final version if accepted. Also an  erratum: In Figure 10 and 11, the $\epsilon = 0.005$ value should be  $\epsilon = 0.05$

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11220v1) [paper-pdf](http://arxiv.org/pdf/2509.11220v1)

**Authors**: Gao Yu Lee, Tanmoy Dam, Md Meftahul Ferdaus, Daniel Puiu Poenar, Vu N. Duong

**Abstract**: Few-Shot Learning (FSL), which involves learning to generalize using only a few data samples, has demonstrated promising and superior performances to ordinary CNN methods. While Bayesian based estimation approaches using Kullback-Leibler (KL) divergence have shown improvements, they remain vulnerable to adversarial attacks and natural noises. We introduce ANROT-HELANet, an Adversarially and Naturally RObusT Hellinger Aggregation Network that significantly advances the state-of-the-art in FSL robustness and performance. Our approach implements an adversarially and naturally robust Hellinger distance-based feature class aggregation scheme, demonstrating resilience to adversarial perturbations up to $\epsilon=0.30$ and Gaussian noise up to $\sigma=0.30$. The network achieves substantial improvements across benchmark datasets, including gains of 1.20\% and 1.40\% for 1-shot and 5-shot scenarios on miniImageNet respectively. We introduce a novel Hellinger Similarity contrastive loss function that generalizes cosine similarity contrastive loss for variational few-shot inference scenarios. Our approach also achieves superior image reconstruction quality with a FID score of 2.75, outperforming traditional VAE (3.43) and WAE (3.38) approaches. Extensive experiments conducted on four few-shot benchmarked datasets verify that ANROT-HELANet's combination of Hellinger distance-based feature aggregation, attention mechanisms, and our novel loss function establishes new state-of-the-art performance while maintaining robustness against both adversarial and natural perturbations. Our code repository will be available at https://github.com/GreedYLearner1146/ANROT-HELANet/tree/main.

摘要: Few-Shot学习（FSL）涉及仅使用少数数据样本学习进行概括，它已表现出比普通CNN方法有希望且更优越的性能。虽然使用Kullback-Leibler（KL）方差的基于Bayesian的估计方法已经显示出改进，但它们仍然容易受到对抗性攻击和自然噪音的影响。我们引入ANROT-HELANet，这是一种对抗性和自然性的RObusT Hellinger聚合网络，它显着提高了FSL稳健性和性能的最新水平。我们的方法实现了一个基于对抗性和自然鲁棒性的Hellinger距离的特征类聚合方案，展示了对高达$\=0.30$的对抗性扰动和高达$\sigma=0.30$的高斯噪音的弹性。该网络在基准数据集中实现了重大改进，包括miniImageNet上的1次和5次场景分别获得1.20%和1.40%的收益。我们引入了一种新颖的Hellinger相似度对比损失函数，它将Cosine相似度对比损失推广到变分少镜头推理场景。我们的方法还实现了卓越的图像重建质量，DID评分为2.75，优于传统的VAE（3.43）和WAE（3.38）方法。在四个几次基准数据集上进行的大量实验验证了ANROT-HELANet将基于Hellinger距离的特征聚合、注意力机制和我们新颖的损失函数相结合，建立了新的最先进性能，同时保持了针对对抗性和自然扰动的鲁棒性。我们的代码存储库将在https://github.com/GreedYLearner1146/ANROT-HELANet/tree/main上提供。



## **30. Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs**

以毒攻毒（F3）：LVLM中一种无需培训且高效的视觉对抗示例净化方法 cs.CV

Accepted by ACM Multimedia 2025 BNI track (Oral)

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.01064v3) [paper-pdf](http://arxiv.org/pdf/2506.01064v3)

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive ``fighting fire with fire'' strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code is available at https://github.com/btzyd/F3.

摘要: 大型视觉语言模型（LVLM）的最新进展展示了它们在广泛的多模式视觉语言任务中的非凡能力。然而，这些模型仍然容易受到视觉对抗攻击，这可能会极大地损害其性能。在本文中，我们介绍了F3，这是一种新型的对抗净化框架，它采用了违反直觉的“以毒攻毒”策略：有意地向对抗性示例引入简单的扰动以减轻其有害影响。具体来说，F3利用从随机干扰的对手示例中获得的跨模式注意力作为参考目标。通过向这些对抗性示例中注入噪音，F3有效地细化了他们的注意力，从而产生更干净、更可靠的模型输出。值得注意的是，这种看似矛盾的利用噪音来抵消对抗攻击的方法产生了令人印象深刻的净化结果。此外，F3具有几个明显的优势：无需训练且易于实施，并且与现有的纯化方法相比，计算效率显着提高。这些属性使F3特别适合大规模工业应用，其中稳健的性能和运营效率都是关键优先事项。该代码可在https://github.com/btzyd/F3上获取。



## **31. DMLDroid: Deep Multimodal Fusion Framework for Android Malware Detection with Resilience to Code Obfuscation and Adversarial Perturbations**

DMLDroid：用于Android恶意软件检测的深度多模式融合框架，具有代码混淆和对抗性扰动的弹性 cs.CR

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11187v1) [paper-pdf](http://arxiv.org/pdf/2509.11187v1)

**Authors**: Doan Minh Trung, Tien Duc Anh Hao, Luong Hoang Minh, Nghi Hoang Khoa, Nguyen Tan Cam, Van-Hau Pham, Phan The Duy

**Abstract**: In recent years, learning-based Android malware detection has seen significant advancements, with detectors generally falling into three categories: string-based, image-based, and graph-based approaches. While these methods have shown strong detection performance, they often struggle to sustain robustness in real-world settings, particularly when facing code obfuscation and adversarial examples (AEs). Deep multimodal learning has emerged as a promising solution, leveraging the strengths of multiple feature types to enhance robustness and generalization. However, a systematic investigation of multimodal fusion for both accuracy and resilience remains underexplored. In this study, we propose DMLDroid, an Android malware detection based on multimodal fusion that leverages three different representations of malware features, including permissions & intents (tabular-based), DEX file representations (image-based), and API calls (graph-derived sequence-based). We conduct exhaustive experiments independently on each feature, as well as in combination, using different fusion strategies. Experimental results on the CICMalDroid 2020 dataset demonstrate that our multimodal approach with the dynamic weighted fusion mechanism achieves high performance, reaching 97.98% accuracy and 98.67% F1-score on original malware detection. Notably, the proposed method maintains strong robustness, sustaining over 98% accuracy and 98% F1-score under both obfuscation and adversarial attack scenarios. Our findings highlight the benefits of multimodal fusion in improving both detection accuracy and robustness against evolving Android malware threats.

摘要: 近年来，基于学习的Android恶意软件检测取得了重大进展，检测器通常分为三类：基于字符串、基于图像和基于图形的方法。虽然这些方法表现出了强大的检测性能，但它们在现实世界环境中常常难以维持稳健性，特别是当面临代码混淆和对抗性示例（AE）时。深度多模式学习已成为一种有前途的解决方案，可以利用多种特征类型的优势来增强稳健性和概括性。然而，对多模式融合的准确性和弹性的系统研究仍然不足。在这项研究中，我们提出了DMLDroid，这是一种基于多模式融合的Android恶意软件检测，它利用了恶意软件功能的三种不同表示，包括权限和意图（基于表格）、SEN文件表示（基于图像）和API调用（基于图形的序列）。我们使用不同的融合策略对每个特征独立进行详尽的实验，以及组合进行详尽的实验。CICMalDroid 2020数据集的实验结果表明，我们采用动态加权融合机制的多模式方法实现了高性能，原始恶意软件检测的准确率达到97.98%，F1得分达到98.67%。值得注意的是，所提出的方法保持了很强的鲁棒性，在模糊和对抗攻击场景下都保持超过98%的准确性和98%的F1评分。我们的研究结果强调了多模式融合在提高检测准确性和针对不断变化的Android恶意软件威胁的稳健性方面的优势。



## **32. Feature Space Topology Control via Hopkins Loss**

通过霍普金斯损失进行特征空间布局控制 cs.LG

Accepted for publication in Proc. IEEE ICTAI 2025, Athens, Greece

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11154v1) [paper-pdf](http://arxiv.org/pdf/2509.11154v1)

**Authors**: Einari Vaaras, Manu Airaksinen

**Abstract**: Feature space topology refers to the organization of samples within the feature space. Modifying this topology can be beneficial in machine learning applications, including dimensionality reduction, generative modeling, transfer learning, and robustness to adversarial attacks. This paper introduces a novel loss function, Hopkins loss, which leverages the Hopkins statistic to enforce a desired feature space topology, which is in contrast to existing topology-related methods that aim to preserve input feature topology. We evaluate the effectiveness of Hopkins loss on speech, text, and image data in two scenarios: classification and dimensionality reduction using nonlinear bottleneck autoencoders. Our experiments show that integrating Hopkins loss into classification or dimensionality reduction has only a small impact on classification performance while providing the benefit of modifying feature topology.

摘要: 特征空间布局是指特征空间内样本的组织。修改此拓扑在机器学习应用中可能是有益的，包括降维、生成式建模、迁移学习和对抗攻击的鲁棒性。本文引入了一种新颖的损失函数Hopkins损失，它利用Hopkins统计量来实施所需的特征空间布局，这与现有的旨在保留输入特征布局的与topology相关的方法形成鲜明对比。我们在两种情况下评估了霍普金斯损失对语音、文本和图像数据的有效性：使用非线性瓶颈自动编码器进行分类和降维。我们的实验表明，将霍普金斯损失集成到分类或降维中，对分类性能的影响很小，同时还提供了修改特征布局的好处。



## **33. Character-Level Perturbations Disrupt LLM Watermarks**

初级扰动扰乱LLM水印 cs.CR

accepted by Network and Distributed System Security (NDSS) Symposium  2026

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.09112v2) [paper-pdf](http://arxiv.org/pdf/2509.09112v2)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.   To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.

摘要: 大型语言模型（LLM）水印将可检测信号嵌入到生成的文本中，以实现版权保护、防止滥用和内容检测。虽然之前的研究使用水印去除攻击来评估稳健性，但这些方法通常不是最优的，从而产生了一种误解，即有效的去除需要大的扰动或强大的对手。   为了弥合差距，我们首先形式化LLM水印的系统模型，并描述了两个受有限访问水印检测器限制的现实威胁模型。然后，我们分析不同类型的扰动在其攻击范围内的变化，即，一次编辑可以影响的代币数量。我们观察到字符级扰动（例如，拼写错误、互换、删除、同字形）可以通过扰乱标记化过程同时影响多个标记。我们证明，在最严格的威胁模型下，字符级扰动对于水印去除来说明显更有效。我们进一步提出了基于遗传算法（GA）的引导删除攻击，使用一个参考检测器进行优化。在一个实际的威胁模型与有限的黑盒查询的水印检测器，我们的方法表现出强大的去除性能。实验证实了字符级扰动的优越性和遗传算法在现实约束条件下去除水印的有效性。此外，我们认为有一个对抗性的困境时，考虑潜在的防御：任何固定的防御可以绕过一个合适的扰动策略。受此原则的启发，我们提出了一种自适应复合字符级攻击。实验结果表明，这种方法可以有效地击败防御。我们的研究结果强调了现有LLM水印方案中的重大漏洞，并强调了开发新的稳健机制的紧迫性。



## **34. ENJ: Optimizing Noise with Genetic Algorithms to Jailbreak LSMs**

ENJ：用遗传算法优化噪音以越狱LSM cs.SD

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11128v1) [paper-pdf](http://arxiv.org/pdf/2509.11128v1)

**Authors**: Yibo Zhang, Liang Lin

**Abstract**: The widespread application of Large Speech Models (LSMs) has made their security risks increasingly prominent. Traditional speech adversarial attack methods face challenges in balancing effectiveness and stealth. This paper proposes Evolutionary Noise Jailbreak (ENJ), which utilizes a genetic algorithm to transform environmental noise from a passive interference into an actively optimizable attack carrier for jailbreaking LSMs. Through operations such as population initialization, crossover fusion, and probabilistic mutation, this method iteratively evolves a series of audio samples that fuse malicious instructions with background noise. These samples sound like harmless noise to humans but can induce the model to parse and execute harmful commands. Extensive experiments on multiple mainstream speech models show that ENJ's attack effectiveness is significantly superior to existing baseline methods. This research reveals the dual role of noise in speech security and provides new critical insights for model security defense in complex acoustic environments.

摘要: 大型语音模型（LSM）的广泛应用使其安全风险日益突出。传统的语音对抗攻击方法在平衡有效性和隐蔽性方面面临挑战。本文提出了进化噪音越狱（ENJ），利用遗传算法将环境噪音从被动干扰转化为可主动优化的攻击载体，用于越狱LSM。该方法通过种群初始化、交叉融合和概率变异等操作，迭代进化一系列音频样本，将恶意指令与背景噪音融合。这些样本对人类来说听起来像是无害的噪音，但可能会导致模型解析和执行有害命令。对多个主流语音模型的大量实验表明，ENJ的攻击有效性明显优于现有的基线方法。这项研究揭示了噪音在语音安全中的双重作用，并为复杂声学环境中的模型安全防御提供了新的关键见解。



## **35. Nonreciprocal RIS-Aided Covert Channel Reciprocity Attacks and Countermeasures**

非互惠的RIS辅助隐蔽通道互惠攻击及对策 eess.SP

submitted to IEEE Trans for review

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11117v1) [paper-pdf](http://arxiv.org/pdf/2509.11117v1)

**Authors**: Haoyu Wang, Jiawei Hu, Jiaqi Xu, Ying Ju, A. Lee Swindlehurst

**Abstract**: Reconfigurable intelligent surface (RIS) technology enhances wireless communication performance, but it also introduces new vulnerabilities that can be exploited by adversaries. This paper investigates channel reciprocity attack (CRACK) threats in multi-antenna wireless systems operating in time-division duplexing mode using a physically consistent non-reciprocal RIS (NR-RIS) model. CRACK can degrade communication rate and facilitate passive eavesdropping behavior by distorting the downlink precoding, without requiring any additional signal transmission or channel state information (CSI). Unlike conventional RIS jamming strategies, the NR-RIS does not need synchronization with the legitimate system and thus can operate with slow or fixed configurations to implement CRACK, obscuring the distinction between the direct and RIS-induced channels and thereby complicating corresponding defensive precoding designs. To counter the CRACK threat posed by NR-RIS, we develop ``SecureCoder,'' a deep reinforcement learning-based framework that can mitigate CRACK and determine an improved downlink precoder matrix using the estimated uplink CSI and rate feedback from the users. Simulation results demonstrate the severe performance degradation caused by NR-RIS CRACK and validate the effectiveness of SecureCoder in improving both throughput and reducing security threats, thereby enhancing system robustness.

摘要: 可重新配置智能表面（RIS）技术增强了无线通信性能，但也引入了可被对手利用的新漏洞。本文使用物理一致的非互惠RIS（NR-RIS）模型研究了以时分交织模式运行的多天线无线系统中的通道互惠攻击（CRACK）威胁。CRACK可以通过失真下行链路预编码来降低通信速率并促进被动窃听行为，而不需要任何额外的信号传输或信道状态信息（SI）。与传统的RIS干扰策略不同，NR-RIS不需要与合法系统同步，因此可以以缓慢或固定的配置运行来实现CRACK，模糊了直接和RIS诱导的通道之间的区别，从而使相应的防御预编码设计变得复杂。为了应对NR-RIS构成的CRACK威胁，我们开发了“SecureCoder”，这是一个基于深度强化学习的框架，可以缓解CRACK并使用估计的上行链路SI和用户的速率反馈来确定改进的下行链路预编码器矩阵。仿真结果证明了NR-RIS CRACK导致的严重性能下降，并验证了SecureCoder在提高吞吐量和减少安全威胁方面的有效性，从而增强系统稳健性。



## **36. Adversarial control of synchronization in complex oscillator networks**

复振子网络同步的对抗控制 nlin.AO

10 pages, 4 figures

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.02403v3) [paper-pdf](http://arxiv.org/pdf/2506.02403v3)

**Authors**: Yasutoshi Nagahama, Kosuke Miyazato, Kazuhiro Takemoto

**Abstract**: This study investigates perturbation strategies inspired by adversarial attack principles from deep learning, designed to control synchronization dynamics through strategically crafted weak perturbations. We propose a gradient-based optimization method that identifies small phase perturbations to dramatically enhance or suppress collective synchronization in Kuramoto oscillator networks. Our approach formulates synchronization control as an optimization problem, computing gradients of the order parameter with respect to oscillator phases to determine optimal perturbation directions. Results demonstrate that extremely small phase perturbations applied to network oscillators can achieve significant synchronization control across diverse network architectures. Our analysis reveals that synchronization enhancement is achievable across various network sizes, while synchronization suppression becomes particularly effective in larger networks, with effectiveness scaling favorably with network size. The method is systematically validated on canonical model networks including scale-free and small-world topologies, and real-world networks representing power grids and brain connectivity patterns. This adversarial framework represents a novel paradigm for synchronization management by introducing deep learning concepts to networked dynamical systems.

摘要: 这项研究调查了受深度学习对抗攻击原则启发的扰动策略，旨在通过策略性精心设计的弱扰动来控制同步动态。我们提出了一种基于梯度的优化方法，该方法识别小的相扰动，以显着增强或抑制仓本振荡器网络中的集体同步。我们的方法将同步控制制定为优化问题，计算阶参数相对于振荡器相的梯度以确定最佳扰动方向。结果表明，应用于网络振荡器的极小的相扰动可以实现跨不同网络架构的显着同步控制。我们的分析表明，同步增强可以在不同的网络规模中实现，而同步抑制在更大的网络中变得特别有效，并且有效性随着网络规模的增加而有利地扩展。该方法在规范模型网络（包括无标度和小世界布局）以及代表电网和大脑连接模式的现实世界网络上进行了系统验证。这种对抗性框架通过将深度学习概念引入网络动态系统，代表了一种新颖的同步管理范式。



## **37. Five Minutes of DDoS Brings down Tor: DDoS Attacks on the Tor Directory Protocol and Mitigations**

五分钟的DDOS攻击Tor：对Tor目录协议的DDOS攻击和缓解措施 cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10755v1) [paper-pdf](http://arxiv.org/pdf/2509.10755v1)

**Authors**: Zhongtang Luo, Jianting Zhang, Akshat Neerati, Aniket Kate

**Abstract**: The Tor network offers network anonymity to its users by routing their traffic through a sequence of relays. A group of nine directory authorities maintains information about all available relay nodes using a distributed directory protocol. We observe that the current protocol makes a steep synchrony assumption, which makes it vulnerable to natural as well as adversarial non-synchronous communication scenarios over the Internet. In this paper, we show that it is possible to cause a failure in the Tor directory protocol by targeting a majority of the authorities for only five minutes using a well-executed distributed denial-of-service (DDoS) attack. We demonstrate this attack in a controlled environment and show that it is cost-effective for as little as \$53.28 per month to disrupt the protocol and to effectively bring down the entire Tor network. To mitigate this problem, we consider the popular partial synchrony assumption for the Tor directory protocol that ensures that the protocol security is hampered even when the network delays are large and unknown. We design a new Tor directory protocol that leverages any standard partial-synchronous consensus protocol to solve this problem, while also proving its security. We have implemented a prototype in Rust, demonstrating comparable performance to the current protocol while resisting similar attacks.

摘要: Tor网络通过一系列中继路由用户的流量，为其用户提供网络匿名性。由九个目录机构组成的小组使用分布式目录协议维护有关所有可用中继节点的信息。我们观察到，当前的协议做出了严格的同步假设，这使得它容易受到互联网上自然和对抗性非同步通信场景的影响。在本文中，我们表明，使用执行良好的分布式拒绝服务（DDOS）攻击仅针对大多数权威机构五分钟，就可能导致Tor目录协议失败。我们在受控环境中演示了这种攻击，并表明每月只需53.28英镑就可以破坏协议并有效地摧毁整个Tor网络。为了缓解这个问题，我们考虑了Tor目录协议的流行部分同步假设，该假设确保即使网络延迟很大且未知，协议安全性也会受到阻碍。我们设计了一种新的Tor目录协议，该协议利用任何标准的部分同步共识协议来解决这个问题，同时也证明了其安全性。我们在Rust中实现了一个原型，展示了与当前协议相当的性能，同时还能抵抗类似的攻击。



## **38. Feature-Centric Approaches to Android Malware Analysis: A Survey**

以设备为中心的Android恶意软件分析方法：一项调查 cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10709v1) [paper-pdf](http://arxiv.org/pdf/2509.10709v1)

**Authors**: Shama Maganur, Yili Jiang, Jiaqi Huang, Fangtian Zhong

**Abstract**: Sophisticated malware families exploit the openness of the Android platform to infiltrate IoT networks, enabling large-scale disruption, data exfiltration, and denial-of-service attacks. This systematic literature review (SLR) examines cutting-edge approaches to Android malware analysis with direct implications for securing IoT infrastructures. We analyze feature extraction techniques across static, dynamic, hybrid, and graph-based methods, highlighting their trade-offs: static analysis offers efficiency but is easily evaded through obfuscation; dynamic analysis provides stronger resistance to evasive behaviors but incurs high computational costs, often unsuitable for lightweight IoT devices; hybrid approaches balance accuracy with resource considerations; and graph-based methods deliver superior semantic modeling and adversarial robustness. This survey contributes a structured comparison of existing methods, exposes research gaps, and outlines a roadmap for future directions to enhance scalability, adaptability, and long-term security in IoT-driven Android malware detection.

摘要: 复杂的恶意软件家族利用Android平台的开放性渗透物联网网络，导致大规模破坏、数据泄露和拒绝服务攻击。这篇系统性文献评论（SLR）探讨了Android恶意软件分析的前沿方法，对确保物联网基础设施的安全有直接影响。我们分析了静态、动态、混合和基于图形的方法中的特征提取技术，强调了它们的权衡：静态分析提供了效率，但很容易通过模糊来规避;动态分析对规避行为提供了更强的抵抗力，但会产生很高的计算成本，通常不适合轻量级物联网设备;混合方法平衡准确性与资源考虑;基于图形的方法提供了卓越的语义建模和对抗鲁棒性。这项调查对现有方法进行了结构化比较，揭示了研究差距，并概述了未来方向的路线图，以增强物联网驱动的Android恶意软件检测的可扩展性、适应性和长期安全性。



## **39. Privacy-Preserving Decentralized Federated Learning via Explainable Adaptive Differential Privacy**

通过可解释的自适应差异隐私保护隐私的去中心联邦学习 cs.CR

21 pages

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10691v1) [paper-pdf](http://arxiv.org/pdf/2509.10691v1)

**Authors**: Fardin Jalil Piran, Zhiling Chen, Yang Zhang, Qianyu Zhou, Jiong Tang, Farhad Imani

**Abstract**: Decentralized federated learning faces privacy risks because model updates can leak data through inference attacks and membership inference, a concern that grows over many client exchanges. Differential privacy offers principled protection by injecting calibrated noise so confidential information remains secure on resource-limited IoT devices. Yet without transparency, black-box training cannot track noise already injected by previous clients and rounds, which forces worst-case additions and harms accuracy. We propose PrivateDFL, an explainable framework that joins hyperdimensional computing with differential privacy and keeps an auditable account of cumulative noise so each client adds only the difference between the required noise and what has already been accumulated. We evaluate on MNIST, ISOLET, and UCI-HAR to span image, signal, and tabular modalities, and we benchmark against transformer-based and deep learning-based baselines trained centrally with Differentially Private Stochastic Gradient Descent (DP-SGD) and Renyi Differential Privacy (RDP). PrivateDFL delivers higher accuracy, lower latency, and lower energy across IID and non-IID partitions while preserving formal (epsilon, delta) guarantees and operating without a central server. For example, under non-IID partitions, PrivateDFL achieves 24.42% higher accuracy than the Vision Transformer on MNIST while using about 10x less training time, 76x lower inference latency, and 11x less energy, and on ISOLET it exceeds Transformer accuracy by more than 80% with roughly 10x less training time, 40x lower inference latency, and 36x less training energy. Future work will extend the explainable accounting to adversarial clients and adaptive topologies with heterogeneous privacy budgets.

摘要: 去中心化联邦学习面临隐私风险，因为模型更新可能会通过推理攻击和成员资格推理泄露数据，这一担忧在许多客户端交换中日益加剧。差异隐私通过注入校准噪音来提供原则性保护，以便机密信息在资源有限的物联网设备上保持安全。然而，如果没有透明度，黑匣子训练就无法跟踪之前的客户和回合已经注入的噪音，这会迫使最坏的情况进行添加并损害准确性。我们提出PrivateDFL，这是一个可解释的框架，它将多维计算与差异隐私结合起来，并保留累积噪音的可审计帐户，以便每个客户端仅添加所需噪音与已经累积的噪音之间的差异。我们在MNIST、ISOLET和UCI-HAR上进行评估，以跨越图像、信号和表格模式，并以基于变换器和基于深度学习的基线为基准，这些基线采用差异私人随机梯度下降（DP-BCD）和雷尼差异隐私（SDP）集中训练。PrivateDFL在IID和非IID分区之间提供更高的准确性、更低的延迟和更低的功耗，同时保留正式（NPS、Delta）保证并在没有中央服务器的情况下运行。例如，在非IID分区下，PrivateDFL的准确性比MNIST上的Vision Transformer高出24.42%，同时使用的训练时间减少了约10倍，推理延迟减少了76倍，能量减少了11倍，而在ISOLET上，它比Transformer准确性高出80%以上，训练时间减少了约10倍，推理延迟减少了40倍，训练能量减少了36倍。未来的工作将将可解释的会计扩展到敌对客户端和具有不同隐私预算的自适应布局。



## **40. Multi-Agent Systems Execute Arbitrary Malicious Code**

多代理系统执行任意恶意代码 cs.CR

33 pages, 5 figures, 7 tables

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2503.12188v2) [paper-pdf](http://arxiv.org/pdf/2503.12188v2)

**Authors**: Harold Triedman, Rishi Jha, Vitaly Shmatikov

**Abstract**: Multi-agent systems coordinate LLM-based agents to perform tasks on users' behalf. In real-world applications, multi-agent systems will inevitably interact with untrusted inputs, such as malicious Web content, files, email attachments, and more.   Using several recently proposed multi-agent frameworks as concrete examples, we demonstrate that adversarial content can hijack control and communication within the system to invoke unsafe agents and functionalities. This results in a complete security breach, up to execution of arbitrary malicious code on the user's device or exfiltration of sensitive data from the user's containerized environment. For example, when agents are instantiated with GPT-4o, Web-based attacks successfully cause the multi-agent system execute arbitrary malicious code in 58-90\% of trials (depending on the orchestrator). In some model-orchestrator configurations, the attack success rate is 100\%. We also demonstrate that these attacks succeed even if individual agents are not susceptible to direct or indirect prompt injection, and even if they refuse to perform harmful actions. We hope that these results will motivate development of trust and security models for multi-agent systems before they are widely deployed.

摘要: 多代理系统协调基于LLM的代理代表用户执行任务。在现实世界的应用程序中，多代理系统将不可避免地与不受信任的输入进行交互，例如恶意Web内容、文件、电子邮件附件等。   使用最近提出的几个多代理框架作为具体示例，我们证明对抗性内容可以劫持系统内的控制和通信，以调用不安全的代理和功能。这会导致完全的安全漏洞，甚至在用户设备上执行任意恶意代码或从用户的容器化环境中泄露敏感数据。例如，当使用GPT-4 o实例化代理时，基于Web的攻击成功导致多代理系统在58- 90%的尝试中执行任意恶意代码（取决于编排器）。在某些模型编排器配置中，攻击成功率为100%。我们还证明，即使个别特工不容易受到直接或间接的即时注射，并且即使他们拒绝执行有害行为，这些攻击也会成功。我们希望这些结果将激励多代理系统在广泛部署之前开发信任和安全模型。



## **41. SAIF: Sparse Adversarial and Imperceptible Attack Framework**

SAIF：稀疏对抗和不可感知的攻击框架 cs.CV

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2212.07495v4) [paper-pdf](http://arxiv.org/pdf/2212.07495v4)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Masih Eskandar, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.

摘要: 对抗性攻击通过干扰输入信号来阻碍神经网络的决策能力。例如，向图像添加计算出的微小失真可能会欺骗训练有素的图像分类网络。在这项工作中，我们提出了一种新型攻击技术，称为稀疏对抗和可解释攻击框架（SAIF）。具体来说，我们设计了不可感知的攻击，其中包含少量像素的低幅度扰动，并利用这些稀疏攻击来揭示分类器的漏洞。我们使用Frank-Wolfe（条件梯度）算法同时优化有界幅度和稀疏性的攻击扰动，并具有$O（1/\SQRT{T}）$收敛。经验结果表明，SAIF可以计算高度不可感知且可解释的对抗示例，并且在ImageNet数据集上优于最先进的稀疏攻击方法。



## **42. Attacking Attention of Foundation Models Disrupts Downstream Tasks**

攻击基础模型的注意力会扰乱下游任务 cs.CR

Paper published at CVPR 2025 Workshop Advml

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2506.05394v3) [paper-pdf](http://arxiv.org/pdf/2506.05394v3)

**Authors**: Hondamunige Prasanna Silva, Federico Becattini, Lorenzo Seidenari

**Abstract**: Foundation models represent the most prominent and recent paradigm shift in artificial intelligence. Foundation models are large models, trained on broad data that deliver high accuracy in many downstream tasks, often without fine-tuning. For this reason, models such as CLIP , DINO or Vision Transfomers (ViT), are becoming the bedrock of many industrial AI-powered applications. However, the reliance on pre-trained foundation models also introduces significant security concerns, as these models are vulnerable to adversarial attacks. Such attacks involve deliberately crafted inputs designed to deceive AI systems, jeopardizing their reliability. This paper studies the vulnerabilities of vision foundation models, focusing specifically on CLIP and ViTs, and explores the transferability of adversarial attacks to downstream tasks. We introduce a novel attack, targeting the structure of transformer-based architectures in a task-agnostic fashion. We demonstrate the effectiveness of our attack on several downstream tasks: classification, captioning, image/text retrieval, segmentation and depth estimation. Code available at:https://github.com/HondamunigePrasannaSilva/attack-attention

摘要: 基础模型代表了人工智能领域最突出、最新的范式转变。基础模型是大型模型，基于广泛的数据进行训练，可以在许多下游任务中提供高准确性，通常无需微调。因此，CLIP、DINO或Vision Transfomers（ViT）等型号正在成为许多工业人工智能应用的基石。然而，对预训练的基础模型的依赖也带来了严重的安全问题，因为这些模型容易受到对抗攻击。此类攻击涉及故意设计的输入，旨在欺骗人工智能系统，危及其可靠性。本文研究了视觉基础模型的漏洞，特别关注CLIP和ViT，并探讨了对抗性攻击到下游任务的可转移性。我们引入了一种新颖的攻击，以任务不可知的方式针对基于转换器的架构的结构。我们展示了我们对几个下游任务的攻击的有效性：分类、字幕、图像/文本检索、分割和深度估计。代码可访问：https://github.com/HondamunigePrasannaSilva/attack-attention



## **43. Immunizing Images from Text to Image Editing via Adversarial Cross-Attention**

通过对抗性交叉注意使图像从文本到图像编辑免疫 cs.CV

Accepted as Regular Paper at ACM Multimedia 2025

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10359v1) [paper-pdf](http://arxiv.org/pdf/2509.10359v1)

**Authors**: Matteo Trippodo, Federico Becattini, Lorenzo Seidenari

**Abstract**: Recent advances in text-based image editing have enabled fine-grained manipulation of visual content guided by natural language. However, such methods are susceptible to adversarial attacks. In this work, we propose a novel attack that targets the visual component of editing methods. We introduce Attention Attack, which disrupts the cross-attention between a textual prompt and the visual representation of the image by using an automatically generated caption of the source image as a proxy for the edit prompt. This breaks the alignment between the contents of the image and their textual description, without requiring knowledge of the editing method or the editing prompt. Reflecting on the reliability of existing metrics for immunization success, we propose two novel evaluation strategies: Caption Similarity, which quantifies semantic consistency between original and adversarial edits, and semantic Intersection over Union (IoU), which measures spatial layout disruption via segmentation masks. Experiments conducted on the TEDBench++ benchmark demonstrate that our attack significantly degrades editing performance while remaining imperceptible.

摘要: 基于文本的图像编辑的最新进展使得能够在自然语言指导下对视觉内容进行细粒度操纵。然而，此类方法很容易受到对抗攻击。在这项工作中，我们提出了一种针对编辑方法的视觉组件的新颖攻击。我们引入了注意力攻击，它通过使用自动生成的源图像标题作为编辑提示的代理来扰乱文本提示和图像视觉表示之间的交叉注意力。这打破了图像内容与其文本描述之间的对齐，而不需要了解编辑方法或编辑提示。考虑到现有免疫成功指标的可靠性，我们提出了两种新颖的评估策略：标题相似性，量化原始编辑和对抗编辑之间的语义一致性，以及语义联合交叉（IoU），通过分割面具测量空间布局中断。在TEDBench++基准测试上进行的实验表明，我们的攻击会显着降低编辑性能，同时保持不可察觉。



## **44. Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs**

在岩石和困难之间：利用道德推理越狱法学硕士 cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.05367v2) [paper-pdf](http://arxiv.org/pdf/2509.05367v2)

**Authors**: Shei Pern Chua, Zhen Leng Thai, Teh Kai Jun, Xiao Li, Xiaolin Hu

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.

摘要: 大型语言模型（LLM）已经进行了安全调整工作，以减轻有害输出。然而，随着LLM的推理变得更加复杂，他们的智能可能会带来新的安全风险。虽然传统的越狱攻击依赖于单步攻击，但动态适应上下文的多回合越狱策略仍然没有得到充分的研究。在这项工作中，我们引入了TRAL（交互式攻击逻辑的电车问题推理），这是一个利用LLM道德推理来绕过其保障措施的框架。TRAL将对抗目标嵌入以电车问题为模型的道德困境中。TriAL展示了开放和封闭源模型的高越狱成功率。我们的研究结果强调了人工智能安全性的一个根本限制：随着模型获得高级推理能力，它们的对齐性质可能会无意中允许更多隐蔽的安全漏洞被利用。TRAL提出了重新评估安全一致监督策略的迫切需要，因为当前的保障措施可能不足以抵御上下文感知的对抗攻击。



## **45. Analyzing the Impact of Adversarial Examples on Explainable Machine Learning**

分析对抗性示例对可解释机器学习的影响 cs.LG

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2307.08327v2) [paper-pdf](http://arxiv.org/pdf/2307.08327v2)

**Authors**: Prathyusha Devabhakthini, Sasmita Parida, Raj Mani Shukla, Suvendu Chandan Nayak, Tapadhir Das

**Abstract**: Adversarial attacks are a type of attack on machine learning models where an attacker deliberately modifies the inputs to cause the model to make incorrect predictions. Adversarial attacks can have serious consequences, particularly in applications such as autonomous vehicles, medical diagnosis, and security systems. Work on the vulnerability of deep learning models to adversarial attacks has shown that it is very easy to make samples that make a model predict things that it doesn't want to. In this work, we analyze the impact of model interpretability due to adversarial attacks on text classification problems. We develop an ML-based classification model for text data. Then, we introduce the adversarial perturbations on the text data to understand the classification performance after the attack. Subsequently, we analyze and interpret the model's explainability before and after the attack

摘要: 对抗性攻击是对机器学习模型的一种攻击，攻击者故意修改输入，导致模型做出错误的预测。对抗性攻击可能会造成严重后果，特别是在自动驾驶汽车、医疗诊断和安全系统等应用中。关于深度学习模型对对抗性攻击的脆弱性的研究表明，很容易制作样本，使模型预测它不想预测的事情。在这项工作中，我们分析了由于对抗性攻击对文本分类问题的模型可解释性的影响。我们为文本数据开发了一个基于ML的分类模型。然后，我们引入对文本数据的对抗性扰动，以了解攻击后的分类性能。随后，我们分析和解释了该模型在攻击前后的可解释性



## **46. Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective**

LLM授权的推荐系统的隐私风险：倒置攻击的角度 cs.IR

Accepted at ACM RecSys 2025 (10 pages, 4 figures)

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2508.03703v2) [paper-pdf](http://arxiv.org/pdf/2508.03703v2)

**Authors**: Yubo Wang, Min Tang, Nuo Shen, Shujie Cui, Weiqing Wang

**Abstract**: The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems.

摘要: 大语言模型（LLM）支持的推荐范式被提出来解决传统推荐系统的局限性，传统推荐系统通常难以处理冷启动用户或具有新ID的项目。尽管有效，这项研究发现LLM授权的推荐系统很容易受到重建攻击，这些攻击可能会暴露系统和用户隐私。为了研究这种威胁，我们对针对LLM授权的推荐系统的倒置攻击进行了首次系统性研究，其中对手试图通过利用推荐模型的输出日志来重建包含个人偏好、交互历史和人口统计属性的原始提示。我们重现vec 2text框架并使用我们提出的名为相似性引导细化的方法对其进行优化，从而能够从模型生成的日志中更准确地重建文本提示。跨两个领域（电影和书籍）的广泛实验和两个代表性的基于LLM的推荐模型证明我们的方法实现了高保真重建。具体来说，我们可以恢复近65%的用户交互项目，并在87%的情况下正确推断年龄和性别。实验还表明，隐私泄露在很大程度上对受害者模型的性能不敏感，但高度依赖于域一致性和提示复杂性。这些发现暴露了LLM授权的推荐系统中的关键隐私漏洞。



## **47. AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models**

AdvI 2I：对图像到图像扩散模型的对抗图像攻击 cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2410.21471v3) [paper-pdf](http://arxiv.org/pdf/2410.21471v3)

**Authors**: Yaopei Zeng, Yuanpu Cao, Bochuan Cao, Yurui Chang, Jinghui Chen, Lu Lin

**Abstract**: Recent advances in diffusion models have significantly enhanced the quality of image synthesis, yet they have also introduced serious safety concerns, particularly the generation of Not Safe for Work (NSFW) content. Previous research has demonstrated that adversarial prompts can be used to generate NSFW content. However, such adversarial text prompts are often easily detectable by text-based filters, limiting their efficacy. In this paper, we expose a previously overlooked vulnerability: adversarial image attacks targeting Image-to-Image (I2I) diffusion models. We propose AdvI2I, a novel framework that manipulates input images to induce diffusion models to generate NSFW content. By optimizing a generator to craft adversarial images, AdvI2I circumvents existing defense mechanisms, such as Safe Latent Diffusion (SLD), without altering the text prompts. Furthermore, we introduce AdvI2I-Adaptive, an enhanced version that adapts to potential countermeasures and minimizes the resemblance between adversarial images and NSFW concept embeddings, making the attack more resilient against defenses. Through extensive experiments, we demonstrate that both AdvI2I and AdvI2I-Adaptive can effectively bypass current safeguards, highlighting the urgent need for stronger security measures to address the misuse of I2I diffusion models.

摘要: 扩散模型的最新进展显着提高了图像合成的质量，但它们也带来了严重的安全问题，特别是工作不安全（NSFW）内容的生成。之前的研究表明，对抗性提示可以用于生成NSFW内容。然而，此类对抗性文本提示通常很容易被基于文本的过滤器检测到，从而限制了其功效。在本文中，我们揭露了一个以前被忽视的漏洞：针对图像到图像（I2 I）扩散模型的对抗图像攻击。我们提出了AdvI 2 I，这是一种新颖的框架，可以操纵输入图像以诱导扩散模型来生成NSFW内容。通过优化生成器来制作对抗图像，AdvI 2 I绕过了现有的防御机制，例如安全潜伏扩散（SLD），而无需更改文本提示。此外，我们还引入了AdvI 2 I-Adaptive，这是一个增强版本，可以适应潜在的对策，并最大限度地减少对抗图像和NSFW概念嵌入之间的相似性，使攻击对防御更具弹性。通过大量实验，我们证明AdvI 2 I和AdvI 2 I-Adaptive都可以有效地绕过当前的保障措施，凸显了迫切需要采取更强有力的安全措施来解决I2 I扩散模型的滥用问题。



## **48. Target Defense Using a Turret and Mobile Defender Team**

使用塔楼和机动防守队进行目标防御 eess.SY

Submitted to IEEE L-CSS and the 2026 ACC

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09777v1) [paper-pdf](http://arxiv.org/pdf/2509.09777v1)

**Authors**: Alexander Von Moll, Dipankar Maity, Meir Pachter, Daigo Shishika, Michael Dorothy

**Abstract**: A scenario is considered wherein a stationary, turn constrained agent (Turret) and a mobile agent (Defender) cooperate to protect the former from an adversarial mobile agent (Attacker). The Attacker wishes to reach the Turret prior to getting captured by either the Defender or Turret, if possible. Meanwhile, the Defender and Turret seek to capture the Attacker as far from the Turret as possible. This scenario is formulated as a differential game and solved using a geometric approach. Necessary and sufficient conditions for the Turret-Defender team winning and the Attacker winning are given. In the case of the Turret-Defender team winning equilibrium strategies for the min max terminal distance of the Attacker to the Turret are given. Three cases arise corresponding to solo capture by the Defender, solo capture by the Turret, and capture simultaneously by both Turret and Defender.

摘要: 考虑了一种场景，其中静止的、转向受限的代理（Turret）和移动代理（Defender）合作以保护前者免受对抗性移动代理（Attacker）的侵害。如果可能的话，攻击者希望在被防守者或塔楼捕获之前到达塔楼。与此同时，防御者和塔楼寻求在距离塔楼尽可能远的地方捕获攻击者。这种情况被描述为一个差异博弈，并使用几何方法来解决。给出了塔楼防守队获胜和进攻队获胜的充分必要条件。在塔楼-防守者团队获胜的情况下，给出了攻击者到塔楼的最小最大末端距离的均衡策略。出现三种情况，对应于防御者单独捕获、塔楼单独捕获以及塔楼和防御者同时捕获。



## **49. Steering MoE LLMs via Expert (De)Activation**

通过专家（去）激活MoE LLM cs.CL

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09660v1) [paper-pdf](http://arxiv.org/pdf/2509.09660v1)

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan Rossi, Trung Bui, Hinrich Schütze, Nanyun Peng

**Abstract**: Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts.

摘要: 大型语言模型（LLM）中的专家混合（MoE）通过专门的前向网络（FFN）子集（称为专家）路由每个令牌。我们介绍了SteerMoE，这是一个通过检测和控制与行为相关的专家来引导MoE模型的框架。我们的检测方法识别出在表现出相反行为的成对输入中具有不同激活模式的专家。通过在推理过程中选择性地（去）激活此类专家，我们可以在无需重新培训或修改权重的情况下控制忠诚和安全等行为。在11个基准和6个LLM中，我们的指导将安全性提高了+20%，忠诚度提高了+27%。在对抗性攻击模式下，它单独会降低-41%，与现有越狱方法结合使用时，安全性会降低-100%，绕过所有安全护栏，暴露了隐藏在专家体内的对齐伪造的新维度。



## **50. Bitcoin under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

波动区块奖励下的比特币：Mempool统计数据如何影响比特币采矿 cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2411.11702v3) [paper-pdf](http://arxiv.org/pdf/2411.11702v3)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: The security of Bitcoin protocols is deeply dependent on the incentives provided to miners, which come from a combination of block rewards and transaction fees. As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, leads to the emergence of new security threats or intensifies existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   This paper presents a reinforcement learning-based tool to develop mining strategies under a more realistic volatile model. We employ the Asynchronous Advantage Actor-Critic (A3C) algorithm, which efficiently handles dynamic environments, such as the Bitcoin mempool, to derive near-optimal mining strategies when interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   We revisit the Bitcoin security threshold presented in the WeRLman paper and demonstrate that the implicit predictability of valuable transaction arrivals in this model leads to an underestimation of the reported threshold. Additionally, we show that, while adversarial strategies like selfish mining under the fixed reward model incur an initial loss period of at least two weeks, the transition toward a transaction-fee era incentivizes mining pools to abandon honest mining for immediate profits. This incentive is expected to become more significant as the protocol reward approaches zero in the future.

摘要: 比特币协议的安全性深深依赖于提供给矿工的激励，这些激励来自区块奖励和交易费用的组合。随着比特币经历更多减半事件，协议奖励收敛为零，使交易费成为矿工奖励的主要来源。比特币激励机制的这种转变，将波动性引入区块奖励，导致新的安全威胁出现或加剧现有的安全威胁。以前对比特币的安全分析要么考虑了固定的块奖励模型，要么考虑了高度简化的波动模型，忽略了比特币内存池行为的复杂性。   本文提出了一种基于强化学习的工具，用于在更现实的波动模型下开发挖掘策略。我们采用Asynamous Active Actor-Critic（A3 C）算法，该算法有效地处理动态环境（例如比特币内存池），以便在与模拟比特币内存池复杂性的环境交互时获得近乎最优的挖掘策略。该工具能够在难度调整之前和之后分析敌对采矿策略，例如自私采矿和底挖，从而深入了解采矿攻击的短期和长期影响。   我们重新审视了WeRL man论文中提出的比特币安全阈值，并证明该模型中有价值交易到达的隐性可预测性导致了对报告阈值的低估。此外，我们表明，虽然固定回报模型下的自私采矿等对抗策略会导致至少两周的初始损失期，但向交易费时代的过渡会激励矿池放弃诚实采矿以获取立即利润。随着未来协议奖励接近零，这种激励预计将变得更加重要。



