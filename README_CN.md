# Latest Adversarial Attack Papers
**update at 2025-09-15 14:35:19**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Attacking Attention of Foundation Models Disrupts Downstream Tasks**

攻击基础模型的注意力会扰乱下游任务 cs.CR

Paper published at CVPR 2025 Workshop Advml

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2506.05394v3) [paper-pdf](http://arxiv.org/pdf/2506.05394v3)

**Authors**: Hondamunige Prasanna Silva, Federico Becattini, Lorenzo Seidenari

**Abstract**: Foundation models represent the most prominent and recent paradigm shift in artificial intelligence. Foundation models are large models, trained on broad data that deliver high accuracy in many downstream tasks, often without fine-tuning. For this reason, models such as CLIP , DINO or Vision Transfomers (ViT), are becoming the bedrock of many industrial AI-powered applications. However, the reliance on pre-trained foundation models also introduces significant security concerns, as these models are vulnerable to adversarial attacks. Such attacks involve deliberately crafted inputs designed to deceive AI systems, jeopardizing their reliability. This paper studies the vulnerabilities of vision foundation models, focusing specifically on CLIP and ViTs, and explores the transferability of adversarial attacks to downstream tasks. We introduce a novel attack, targeting the structure of transformer-based architectures in a task-agnostic fashion. We demonstrate the effectiveness of our attack on several downstream tasks: classification, captioning, image/text retrieval, segmentation and depth estimation. Code available at:https://github.com/HondamunigePrasannaSilva/attack-attention

摘要: 基础模型代表了人工智能领域最突出、最新的范式转变。基础模型是大型模型，基于广泛的数据进行训练，可以在许多下游任务中提供高准确性，通常无需微调。因此，CLIP、DINO或Vision Transfomers（ViT）等型号正在成为许多工业人工智能应用的基石。然而，对预训练的基础模型的依赖也带来了严重的安全问题，因为这些模型容易受到对抗攻击。此类攻击涉及故意设计的输入，旨在欺骗人工智能系统，危及其可靠性。本文研究了视觉基础模型的漏洞，特别关注CLIP和ViT，并探讨了对抗性攻击到下游任务的可转移性。我们引入了一种新颖的攻击，以任务不可知的方式针对基于转换器的架构的结构。我们展示了我们对几个下游任务的攻击的有效性：分类、字幕、图像/文本检索、分割和深度估计。代码可访问：https://github.com/HondamunigePrasannaSilva/attack-attention



## **2. Immunizing Images from Text to Image Editing via Adversarial Cross-Attention**

通过对抗性交叉注意使图像从文本到图像编辑免疫 cs.CV

Accepted as Regular Paper at ACM Multimedia 2025

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10359v1) [paper-pdf](http://arxiv.org/pdf/2509.10359v1)

**Authors**: Matteo Trippodo, Federico Becattini, Lorenzo Seidenari

**Abstract**: Recent advances in text-based image editing have enabled fine-grained manipulation of visual content guided by natural language. However, such methods are susceptible to adversarial attacks. In this work, we propose a novel attack that targets the visual component of editing methods. We introduce Attention Attack, which disrupts the cross-attention between a textual prompt and the visual representation of the image by using an automatically generated caption of the source image as a proxy for the edit prompt. This breaks the alignment between the contents of the image and their textual description, without requiring knowledge of the editing method or the editing prompt. Reflecting on the reliability of existing metrics for immunization success, we propose two novel evaluation strategies: Caption Similarity, which quantifies semantic consistency between original and adversarial edits, and semantic Intersection over Union (IoU), which measures spatial layout disruption via segmentation masks. Experiments conducted on the TEDBench++ benchmark demonstrate that our attack significantly degrades editing performance while remaining imperceptible.

摘要: 基于文本的图像编辑的最新进展使得能够在自然语言指导下对视觉内容进行细粒度操纵。然而，此类方法很容易受到对抗攻击。在这项工作中，我们提出了一种针对编辑方法的视觉组件的新颖攻击。我们引入了注意力攻击，它通过使用自动生成的源图像标题作为编辑提示的代理来扰乱文本提示和图像视觉表示之间的交叉注意力。这打破了图像内容与其文本描述之间的对齐，而不需要了解编辑方法或编辑提示。考虑到现有免疫成功指标的可靠性，我们提出了两种新颖的评估策略：标题相似性，量化原始编辑和对抗编辑之间的语义一致性，以及语义联合交叉（IoU），通过分割面具测量空间布局中断。在TEDBench++基准测试上进行的实验表明，我们的攻击会显着降低编辑性能，同时保持不可察觉。



## **3. Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs**

在岩石和困难之间：利用道德推理越狱法学硕士 cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.05367v2) [paper-pdf](http://arxiv.org/pdf/2509.05367v2)

**Authors**: Shei Pern Chua, Zhen Leng Thai, Teh Kai Jun, Xiao Li, Xiaolin Hu

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.

摘要: 大型语言模型（LLM）已经进行了安全调整工作，以减轻有害输出。然而，随着LLM的推理变得更加复杂，他们的智能可能会带来新的安全风险。虽然传统的越狱攻击依赖于单步攻击，但动态适应上下文的多回合越狱策略仍然没有得到充分的研究。在这项工作中，我们引入了TRAL（交互式攻击逻辑的电车问题推理），这是一个利用LLM道德推理来绕过其保障措施的框架。TRAL将对抗目标嵌入以电车问题为模型的道德困境中。TriAL展示了开放和封闭源模型的高越狱成功率。我们的研究结果强调了人工智能安全性的一个根本限制：随着模型获得高级推理能力，它们的对齐性质可能会无意中允许更多隐蔽的安全漏洞被利用。TRAL提出了重新评估安全一致监督策略的迫切需要，因为当前的保障措施可能不足以抵御上下文感知的对抗攻击。



## **4. Analyzing the Impact of Adversarial Examples on Explainable Machine Learning**

分析对抗性示例对可解释机器学习的影响 cs.LG

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2307.08327v2) [paper-pdf](http://arxiv.org/pdf/2307.08327v2)

**Authors**: Prathyusha Devabhakthini, Sasmita Parida, Raj Mani Shukla, Suvendu Chandan Nayak, Tapadhir Das

**Abstract**: Adversarial attacks are a type of attack on machine learning models where an attacker deliberately modifies the inputs to cause the model to make incorrect predictions. Adversarial attacks can have serious consequences, particularly in applications such as autonomous vehicles, medical diagnosis, and security systems. Work on the vulnerability of deep learning models to adversarial attacks has shown that it is very easy to make samples that make a model predict things that it doesn't want to. In this work, we analyze the impact of model interpretability due to adversarial attacks on text classification problems. We develop an ML-based classification model for text data. Then, we introduce the adversarial perturbations on the text data to understand the classification performance after the attack. Subsequently, we analyze and interpret the model's explainability before and after the attack

摘要: 对抗性攻击是对机器学习模型的一种攻击，攻击者故意修改输入，导致模型做出错误的预测。对抗性攻击可能会造成严重后果，特别是在自动驾驶汽车、医疗诊断和安全系统等应用中。关于深度学习模型对对抗性攻击的脆弱性的研究表明，很容易制作样本，使模型预测它不想预测的事情。在这项工作中，我们分析了由于对抗性攻击对文本分类问题的模型可解释性的影响。我们为文本数据开发了一个基于ML的分类模型。然后，我们引入对文本数据的对抗性扰动，以了解攻击后的分类性能。随后，我们分析和解释了该模型在攻击前后的可解释性



## **5. Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective**

LLM授权的推荐系统的隐私风险：倒置攻击的角度 cs.IR

Accepted at ACM RecSys 2025 (10 pages, 4 figures)

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2508.03703v2) [paper-pdf](http://arxiv.org/pdf/2508.03703v2)

**Authors**: Yubo Wang, Min Tang, Nuo Shen, Shujie Cui, Weiqing Wang

**Abstract**: The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems.

摘要: 大语言模型（LLM）支持的推荐范式被提出来解决传统推荐系统的局限性，传统推荐系统通常难以处理冷启动用户或具有新ID的项目。尽管有效，这项研究发现LLM授权的推荐系统很容易受到重建攻击，这些攻击可能会暴露系统和用户隐私。为了研究这种威胁，我们对针对LLM授权的推荐系统的倒置攻击进行了首次系统性研究，其中对手试图通过利用推荐模型的输出日志来重建包含个人偏好、交互历史和人口统计属性的原始提示。我们重现vec 2text框架并使用我们提出的名为相似性引导细化的方法对其进行优化，从而能够从模型生成的日志中更准确地重建文本提示。跨两个领域（电影和书籍）的广泛实验和两个代表性的基于LLM的推荐模型证明我们的方法实现了高保真重建。具体来说，我们可以恢复近65%的用户交互项目，并在87%的情况下正确推断年龄和性别。实验还表明，隐私泄露在很大程度上对受害者模型的性能不敏感，但高度依赖于域一致性和提示复杂性。这些发现暴露了LLM授权的推荐系统中的关键隐私漏洞。



## **6. AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models**

AdvI 2I：对图像到图像扩散模型的对抗图像攻击 cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2410.21471v3) [paper-pdf](http://arxiv.org/pdf/2410.21471v3)

**Authors**: Yaopei Zeng, Yuanpu Cao, Bochuan Cao, Yurui Chang, Jinghui Chen, Lu Lin

**Abstract**: Recent advances in diffusion models have significantly enhanced the quality of image synthesis, yet they have also introduced serious safety concerns, particularly the generation of Not Safe for Work (NSFW) content. Previous research has demonstrated that adversarial prompts can be used to generate NSFW content. However, such adversarial text prompts are often easily detectable by text-based filters, limiting their efficacy. In this paper, we expose a previously overlooked vulnerability: adversarial image attacks targeting Image-to-Image (I2I) diffusion models. We propose AdvI2I, a novel framework that manipulates input images to induce diffusion models to generate NSFW content. By optimizing a generator to craft adversarial images, AdvI2I circumvents existing defense mechanisms, such as Safe Latent Diffusion (SLD), without altering the text prompts. Furthermore, we introduce AdvI2I-Adaptive, an enhanced version that adapts to potential countermeasures and minimizes the resemblance between adversarial images and NSFW concept embeddings, making the attack more resilient against defenses. Through extensive experiments, we demonstrate that both AdvI2I and AdvI2I-Adaptive can effectively bypass current safeguards, highlighting the urgent need for stronger security measures to address the misuse of I2I diffusion models.

摘要: 扩散模型的最新进展显着提高了图像合成的质量，但它们也带来了严重的安全问题，特别是工作不安全（NSFW）内容的生成。之前的研究表明，对抗性提示可以用于生成NSFW内容。然而，此类对抗性文本提示通常很容易被基于文本的过滤器检测到，从而限制了其功效。在本文中，我们揭露了一个以前被忽视的漏洞：针对图像到图像（I2 I）扩散模型的对抗图像攻击。我们提出了AdvI 2 I，这是一种新颖的框架，可以操纵输入图像以诱导扩散模型来生成NSFW内容。通过优化生成器来制作对抗图像，AdvI 2 I绕过了现有的防御机制，例如安全潜伏扩散（SLD），而无需更改文本提示。此外，我们还引入了AdvI 2 I-Adaptive，这是一个增强版本，可以适应潜在的对策，并最大限度地减少对抗图像和NSFW概念嵌入之间的相似性，使攻击对防御更具弹性。通过大量实验，我们证明AdvI 2 I和AdvI 2 I-Adaptive都可以有效地绕过当前的保障措施，凸显了迫切需要采取更强有力的安全措施来解决I2 I扩散模型的滥用问题。



## **7. Target Defense Using a Turret and Mobile Defender Team**

使用塔楼和机动防守队进行目标防御 eess.SY

Submitted to IEEE L-CSS and the 2026 ACC

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09777v1) [paper-pdf](http://arxiv.org/pdf/2509.09777v1)

**Authors**: Alexander Von Moll, Dipankar Maity, Meir Pachter, Daigo Shishika, Michael Dorothy

**Abstract**: A scenario is considered wherein a stationary, turn constrained agent (Turret) and a mobile agent (Defender) cooperate to protect the former from an adversarial mobile agent (Attacker). The Attacker wishes to reach the Turret prior to getting captured by either the Defender or Turret, if possible. Meanwhile, the Defender and Turret seek to capture the Attacker as far from the Turret as possible. This scenario is formulated as a differential game and solved using a geometric approach. Necessary and sufficient conditions for the Turret-Defender team winning and the Attacker winning are given. In the case of the Turret-Defender team winning equilibrium strategies for the min max terminal distance of the Attacker to the Turret are given. Three cases arise corresponding to solo capture by the Defender, solo capture by the Turret, and capture simultaneously by both Turret and Defender.

摘要: 考虑了一种场景，其中静止的、转向受限的代理（Turret）和移动代理（Defender）合作以保护前者免受对抗性移动代理（Attacker）的侵害。如果可能的话，攻击者希望在被防守者或塔楼捕获之前到达塔楼。与此同时，防御者和塔楼寻求在距离塔楼尽可能远的地方捕获攻击者。这种情况被描述为一个差异博弈，并使用几何方法来解决。给出了塔楼防守队获胜和进攻队获胜的充分必要条件。在塔楼-防守者团队获胜的情况下，给出了攻击者到塔楼的最小最大末端距离的均衡策略。出现三种情况，对应于防御者单独捕获、塔楼单独捕获以及塔楼和防御者同时捕获。



## **8. Steering MoE LLMs via Expert (De)Activation**

通过专家（去）激活MoE LLM cs.CL

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09660v1) [paper-pdf](http://arxiv.org/pdf/2509.09660v1)

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan Rossi, Trung Bui, Hinrich Schütze, Nanyun Peng

**Abstract**: Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts.

摘要: 大型语言模型（LLM）中的专家混合（MoE）通过专门的前向网络（FFN）子集（称为专家）路由每个令牌。我们介绍了SteerMoE，这是一个通过检测和控制与行为相关的专家来引导MoE模型的框架。我们的检测方法识别出在表现出相反行为的成对输入中具有不同激活模式的专家。通过在推理过程中选择性地（去）激活此类专家，我们可以在无需重新培训或修改权重的情况下控制忠诚和安全等行为。在11个基准和6个LLM中，我们的指导将安全性提高了+20%，忠诚度提高了+27%。在对抗性攻击模式下，它单独会降低-41%，与现有越狱方法结合使用时，安全性会降低-100%，绕过所有安全护栏，暴露了隐藏在专家体内的对齐伪造的新维度。



## **9. Bitcoin under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

波动区块奖励下的比特币：Mempool统计数据如何影响比特币采矿 cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2411.11702v3) [paper-pdf](http://arxiv.org/pdf/2411.11702v3)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: The security of Bitcoin protocols is deeply dependent on the incentives provided to miners, which come from a combination of block rewards and transaction fees. As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, leads to the emergence of new security threats or intensifies existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   This paper presents a reinforcement learning-based tool to develop mining strategies under a more realistic volatile model. We employ the Asynchronous Advantage Actor-Critic (A3C) algorithm, which efficiently handles dynamic environments, such as the Bitcoin mempool, to derive near-optimal mining strategies when interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   We revisit the Bitcoin security threshold presented in the WeRLman paper and demonstrate that the implicit predictability of valuable transaction arrivals in this model leads to an underestimation of the reported threshold. Additionally, we show that, while adversarial strategies like selfish mining under the fixed reward model incur an initial loss period of at least two weeks, the transition toward a transaction-fee era incentivizes mining pools to abandon honest mining for immediate profits. This incentive is expected to become more significant as the protocol reward approaches zero in the future.

摘要: 比特币协议的安全性深深依赖于提供给矿工的激励，这些激励来自区块奖励和交易费用的组合。随着比特币经历更多减半事件，协议奖励收敛为零，使交易费成为矿工奖励的主要来源。比特币激励机制的这种转变，将波动性引入区块奖励，导致新的安全威胁出现或加剧现有的安全威胁。以前对比特币的安全分析要么考虑了固定的块奖励模型，要么考虑了高度简化的波动模型，忽略了比特币内存池行为的复杂性。   本文提出了一种基于强化学习的工具，用于在更现实的波动模型下开发挖掘策略。我们采用Asynamous Active Actor-Critic（A3 C）算法，该算法有效地处理动态环境（例如比特币内存池），以便在与模拟比特币内存池复杂性的环境交互时获得近乎最优的挖掘策略。该工具能够在难度调整之前和之后分析敌对采矿策略，例如自私采矿和底挖，从而深入了解采矿攻击的短期和长期影响。   我们重新审视了WeRL man论文中提出的比特币安全阈值，并证明该模型中有价值交易到达的隐性可预测性导致了对报告阈值的低估。此外，我们表明，虽然固定回报模型下的自私采矿等对抗策略会导致至少两周的初始损失期，但向交易费时代的过渡会激励矿池放弃诚实采矿以获取立即利润。随着未来协议奖励接近零，这种激励预计将变得更加重要。



## **10. ProDiGy: Proximity- and Dissimilarity-Based Byzantine-Robust Federated Learning**

ProDiGamma：基于邻近性和异化的拜占庭稳健联邦学习 cs.LG

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09534v1) [paper-pdf](http://arxiv.org/pdf/2509.09534v1)

**Authors**: Sena Ergisi, Luis Maßny, Rawad Bitar

**Abstract**: Federated Learning (FL) emerged as a widely studied paradigm for distributed learning. Despite its many advantages, FL remains vulnerable to adversarial attacks, especially under data heterogeneity. We propose a new Byzantine-robust FL algorithm called ProDiGy. The key novelty lies in evaluating the client gradients using a joint dual scoring system based on the gradients' proximity and dissimilarity. We demonstrate through extensive numerical experiments that ProDiGy outperforms existing defenses in various scenarios. In particular, when the clients' data do not follow an IID distribution, while other defense mechanisms fail, ProDiGy maintains strong defense capabilities and model accuracy. These findings highlight the effectiveness of a dual perspective approach that promotes natural similarity among honest clients while detecting suspicious uniformity as a potential indicator of an attack.

摘要: 联邦学习（FL）是一种广泛研究的分布式学习范式。尽管FL具有许多优势，但仍然容易受到对抗攻击，尤其是在数据异类情况下。我们提出了一种新的拜占庭鲁棒FL算法，称为ProDiGy。关键的新颖之处在于使用基于梯度的接近性和不同性的联合双重评分系统来评估客户梯度。我们通过广泛的数字实验证明，ProDiGamma在各种情况下优于现有的防御。特别是，当客户的数据不遵循IID分布，而其他防御机制失败时，ProDiGamma可以保持强大的防御能力和模型准确性。这些发现凸显了双视角方法的有效性，该方法促进诚实客户之间的自然相似性，同时检测可疑的一致性作为攻击的潜在指标。



## **11. On the Relationship Between Adversarial Robustness and Decision Region in Deep Neural Networks**

深度神经网络中对抗鲁棒性与决策区域之间的关系 cs.LG

10 pages

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2207.03400v2) [paper-pdf](http://arxiv.org/pdf/2207.03400v2)

**Authors**: Seongjin Park, Haedong Jeong, Tair Djanibekov, Giyoung Jeon, Jinseok Seol, Jaesik Choi

**Abstract**: In general, Deep Neural Networks (DNNs) are evaluated by the generalization performance measured on unseen data excluded from the training phase. Along with the development of DNNs, the generalization performance converges to the state-of-the-art and it becomes difficult to evaluate DNNs solely based on this metric. The robustness against adversarial attack has been used as an additional metric to evaluate DNNs by measuring their vulnerability. However, few studies have been performed to analyze the adversarial robustness in terms of the geometry in DNNs. In this work, we perform an empirical study to analyze the internal properties of DNNs that affect model robustness under adversarial attacks. In particular, we propose the novel concept of the Populated Region Set (PRS), where training samples are populated more frequently, to represent the internal properties of DNNs in a practical setting. From systematic experiments with the proposed concept, we provide empirical evidence to validate that a low PRS ratio has a strong relationship with the adversarial robustness of DNNs. We also devise PRS regularizer leveraging the characteristics of PRS to improve the adversarial robustness without adversarial training.

摘要: 一般来说，深度神经网络（DNN）是通过在训练阶段排除的未见数据上测量的概括性能来评估的。随着DNN的发展，概括性能趋于最新水平，仅根据这一指标评估DNN变得困难。对抗攻击的稳健性已被用作通过测量DNN的脆弱性来评估DNN的额外指标。然而，很少有研究来分析DNN中的几何形状的对抗鲁棒性。在这项工作中，我们进行了一项实证研究，以分析DNN在对抗攻击下影响模型稳健性的内部属性。特别是，我们提出了填充区域集（PRI）的新颖概念，其中训练样本被更频繁地填充，以表示实际环境中DNN的内部属性。通过对所提出概念的系统实验，我们提供了经验证据来验证低的PPA比与DNN的对抗鲁棒性有很强的关系。我们还设计了利用PRS的特征的PRS正规化器，以在无需对抗训练的情况下提高对抗鲁棒性。



## **12. Over-the-Air Adversarial Attack Detection: from Datasets to Defenses**

空中对抗攻击检测：从数据集到防御 eess.AS

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09296v1) [paper-pdf](http://arxiv.org/pdf/2509.09296v1)

**Authors**: Li Wang, Xiaoyan Lei, Haorui He, Lei Wang, Jie Shi, Zhizheng Wu

**Abstract**: Automatic Speaker Verification (ASV) systems can be used for voice-enabled applications for identity verification. However, recent studies have exposed these systems' vulnerabilities to both over-the-line (OTL) and over-the-air (OTA) adversarial attacks. Although various detection methods have been proposed to counter these threats, they have not been thoroughly tested due to the lack of a comprehensive data set. To address this gap, we developed the AdvSV 2.0 dataset, which contains 628k samples with a total duration of 800 hours. This dataset incorporates classical adversarial attack algorithms, ASV systems, and encompasses both OTL and OTA scenarios. Furthermore, we introduce a novel adversarial attack method based on a Neural Replay Simulator (NRS), which enhances the potency of adversarial OTA attacks, thereby presenting a greater threat to ASV systems. To defend against these attacks, we propose CODA-OCC, a contrastive learning approach within the one-class classification framework. Experimental results show that CODA-OCC achieves an EER of 11.2% and an AUC of 0.95 on the AdvSV 2.0 dataset, outperforming several state-of-the-art detection methods.

摘要: 自动说话人验证（ASV）系统可用于支持语音的应用程序进行身份验证。然而，最近的研究暴露了这些系统对在线（OTL）和空中（OTA）对抗攻击的脆弱性。尽管人们提出了各种检测方法来应对这些威胁，但由于缺乏全面的数据集，这些方法尚未经过彻底测试。为了解决这一差距，我们开发了AdvSV 2.0数据集，其中包含628 k个样本，总持续时间为800小时。该数据集融合了经典的对抗攻击算法、ASV系统，并涵盖OTL和OTA场景。此外，我们还引入了一种基于神经回放模拟器（NRS）的新型对抗攻击方法，该方法增强了对抗OTA攻击的效力，从而对ASV系统构成更大的威胁。为了抵御这些攻击，我们提出了CODA-OSC，这是一种一类分类框架内的对比学习方法。实验结果表明，CODA-BCC在AdvSV 2.0数据集上实现了11.2%的EER和0.95的AUC-0.95，优于多种最先进的检测方法。



## **13. Byzantine-Robust Federated Learning Using Generative Adversarial Networks**

使用生成对抗网络的拜占庭鲁棒联邦学习 cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2503.20884v3) [paper-pdf](http://arxiv.org/pdf/2503.20884v3)

**Authors**: Usama Zafar, André M. H. Teixeira, Salman Toor

**Abstract**: Federated learning (FL) enables collaborative model training across distributed clients without sharing raw data, but its robustness is threatened by Byzantine behaviors such as data and model poisoning. Existing defenses face fundamental limitations: robust aggregation rules incur error lower bounds that grow with client heterogeneity, while detection-based methods often rely on heuristics (e.g., a fixed number of malicious clients) or require trusted external datasets for validation. We present a defense framework that addresses these challenges by leveraging a conditional generative adversarial network (cGAN) at the server to synthesize representative data for validating client updates. This approach eliminates reliance on external datasets, adapts to diverse attack strategies, and integrates seamlessly into standard FL workflows. Extensive experiments on benchmark datasets demonstrate that our framework accurately distinguishes malicious from benign clients while maintaining overall model accuracy. Beyond Byzantine robustness, we also examine the representativeness of synthesized data, computational costs of cGAN training, and the transparency and scalability of our approach.

摘要: 联合学习（FL）可以在不共享原始数据的情况下跨分布式客户端进行协作模型训练，但其稳健性受到数据和模型中毒等拜占庭行为的威胁。现有的防御面临根本性限制：稳健的聚合规则会导致随着客户端的多样性而增长的错误下限，而基于检测的方法通常依赖于启发式方法（例如，固定数量的恶意客户端）或需要受信任的外部数据集进行验证。我们提出了一个防御框架，通过利用服务器上的条件生成对抗网络（cGAN）来合成代表性数据以验证客户端更新来解决这些挑战。这种方法消除了对外部数据集的依赖，适应不同的攻击策略，并无缝集成到标准FL工作流程中。对基准数据集的广泛实验表明，我们的框架可以准确地区分恶意客户端和良性客户端，同时保持整体模型准确性。除了拜占庭鲁棒性之外，我们还研究了合成数据的代表性，cGAN训练的计算成本以及我们方法的透明度和可扩展性。



## **14. TESSER: Transfer-Enhancing Adversarial Attacks from Vision Transformers via Spectral and Semantic Regularization**

TESSER：通过频谱和语义正规化来自视觉变形者的传输增强对抗攻击 cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2505.19613v2) [paper-pdf](http://arxiv.org/pdf/2505.19613v2)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Adversarial transferability remains a critical challenge in evaluating the robustness of deep neural networks. In security-critical applications, transferability enables black-box attacks without access to model internals, making it a key concern for real-world adversarial threat assessment. While Vision Transformers (ViTs) have demonstrated strong adversarial performance, existing attacks often fail to transfer effectively across architectures, especially from ViTs to Convolutional Neural Networks (CNNs) or hybrid models. In this paper, we introduce \textbf{TESSER} -- a novel adversarial attack framework that enhances transferability via two key strategies: (1) \textit{Feature-Sensitive Gradient Scaling (FSGS)}, which modulates gradients based on token-wise importance derived from intermediate feature activations, and (2) \textit{Spectral Smoothness Regularization (SSR)}, which suppresses high-frequency noise in perturbations using a differentiable Gaussian prior. These components work in tandem to generate perturbations that are both semantically meaningful and spectrally smooth. Extensive experiments on ImageNet across 12 diverse architectures demonstrate that TESSER achieves +10.9\% higher attack succes rate (ASR) on CNNs and +7.2\% on ViTs compared to the state-of-the-art Adaptive Token Tuning (ATT) method. Moreover, TESSER significantly improves robustness against defended models, achieving 53.55\% ASR on adversarially trained CNNs. Qualitative analysis shows strong alignment between TESSER's perturbations and salient visual regions identified via Grad-CAM, while frequency-domain analysis reveals a 12\% reduction in high-frequency energy, confirming the effectiveness of spectral regularization.

摘要: 对抗性可移植性仍然是评估深度神经网络稳健性的一个关键挑战。在安全关键型应用程序中，可移植性可以在不访问模型内部的情况下进行黑匣子攻击，使其成为现实世界对抗威胁评估的关键问题。虽然Vision Transformers（ViT）表现出了强大的对抗性能，但现有的攻击往往无法有效地跨架构转移，特别是从ViT到卷积神经网络（CNN）或混合模型。在本文中，我们介绍了\textBF{TESSER} --一种新型的对抗攻击框架，通过两种关键策略增强可移植性：（1）\textit{条件敏感的梯度缩放（FMSG）}，它根据源自中间特征激活的标记重要性来调制梯度，和（2）\textit{光谱平滑度正规化（SSSR）}，它使用可微高斯先验来抑制扰动中的高频噪音。这些组件协同工作，以生成语义上有意义且频谱上光滑的扰动。在ImageNet上跨12种不同架构的广泛实验表明，与最先进的自适应令牌调整（ATA）方法相比，TESSER在CNN上的攻击成功率（ASB）提高了+10.9%，ViT上的攻击成功率（ASB）提高了+7.2%。此外，TESSER显着提高了对防御模型的鲁棒性，在对抗训练的CNN上实现了53.55%的ASR。定性分析表明，TESSER的扰动和通过Grad-CAM识别的显著视觉区域之间具有很强的一致性，而频域分析显示高频能量减少了12%，证实了谱正则化的有效性。



## **15. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

IDEATOR：使用自己越狱和基准大型视觉语言模型 cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2411.00827v5) [paper-pdf](http://arxiv.org/pdf/2411.00827v5)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLJailbreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.VLJailbreakBench is publicly available at https://roywang021.github.io/VLJailbreakBench.

摘要: 随着大型视觉语言模型（VLM）的日益突出，确保其安全部署变得至关重要。最近的研究探索了VLM针对越狱攻击的鲁棒性--利用模型漏洞来引发有害输出的技术。然而，多样化多模式数据的可用性有限，限制了当前的方法严重依赖于从有害文本数据集派生的对抗性或手动制作的图像，而这些图像通常缺乏跨不同背景的有效性和多样性。本文中，我们提出了IDEATOR，这是一种新型越狱方法，可以自主生成用于黑匣子越狱攻击的恶意图像-文本对。IDEATOR基于这样的见解：VLM本身可以充当强大的红队模型，用于生成多模式越狱提示。具体来说，IDEATOR利用VLM创建有针对性的越狱文本，并将其与由最先进的扩散模型生成的越狱图像配对。大量实验证明了IDEATOR的高效率和可移植性，在越狱MiniGPT-4中平均只需5.34次查询即可实现94%的攻击成功率（ASB），转移到LLaVA、INSTBLIP和Chameleon时，攻击成功率分别为82%、88%和75%。基于IDEATOR强大的可移植性和自动化流程，我们推出了VLJailbreakBench，这是一个由3，654个多模式越狱样本组成的安全基准。我们对最近发布的11个VLM的基准结果揭示了安全一致方面的显着差距。例如，我们的挑战集在GPT-4 o上实现了46.31%的ASB，在Claude-3.5-Sonnet上实现了19.65%的ASB，这凸显了对更强防御的迫切需要。VLJailbreakBench可在https://roywang021.github.io/VLJailbreakBench上公开获取。



## **16. Character-Level Perturbations Disrupt LLM Watermarks**

初级扰动扰乱LLM水印 cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09112v1) [paper-pdf](http://arxiv.org/pdf/2509.09112v1)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.   To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.

摘要: 大型语言模型（LLM）水印将可检测信号嵌入到生成的文本中，以实现版权保护、防止滥用和内容检测。虽然之前的研究使用水印去除攻击来评估稳健性，但这些方法通常不是最优的，从而产生了一种误解，即有效的去除需要大的扰动或强大的对手。   为了弥合差距，我们首先形式化LLM水印的系统模型，并描述了两个受有限访问水印检测器限制的现实威胁模型。然后，我们分析不同类型的扰动在其攻击范围内的变化，即，一次编辑可以影响的代币数量。我们观察到字符级扰动（例如，拼写错误、互换、删除、同字形）可以通过扰乱标记化过程同时影响多个标记。我们证明，在最严格的威胁模型下，字符级扰动对于水印去除来说明显更有效。我们进一步提出了基于遗传算法（GA）的引导删除攻击，使用一个参考检测器进行优化。在一个实际的威胁模型与有限的黑盒查询的水印检测器，我们的方法表现出强大的去除性能。实验证实了字符级扰动的优越性和遗传算法在现实约束条件下去除水印的有效性。此外，我们认为有一个对抗性的困境时，考虑潜在的防御：任何固定的防御可以绕过一个合适的扰动策略。受此原则的启发，我们提出了一种自适应复合字符级攻击。实验结果表明，这种方法可以有效地击败防御。我们的研究结果强调了现有LLM水印方案中的重大漏洞，并强调了开发新的稳健机制的紧迫性。



## **17. AdvReal: Physical Adversarial Patch Generation Framework for Security Evaluation of Object Detection Systems**

AdvReal：用于对象检测系统安全评估的物理对抗补丁生成框架 cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2505.16402v2) [paper-pdf](http://arxiv.org/pdf/2505.16402v2)

**Authors**: Yuanhao Huang, Yilong Ren, Jinlei Wang, Lujia Huo, Xuesong Bai, Jinchuan Zhang, Haiyan Yu

**Abstract**: Autonomous vehicles are typical complex intelligent systems with artificial intelligence at their core. However, perception methods based on deep learning are extremely vulnerable to adversarial samples, resulting in security accidents. How to generate effective adversarial examples in the physical world and evaluate object detection systems is a huge challenge. In this study, we propose a unified joint adversarial training framework for both 2D and 3D domains, which simultaneously optimizes texture maps in 2D image and 3D mesh spaces to better address intra-class diversity and real-world environmental variations. The framework includes a novel realistic enhanced adversarial module, with time-space and relighting mapping pipeline that adjusts illumination consistency between adversarial patches and target garments under varied viewpoints. Building upon this, we develop a realism enhancement mechanism that incorporates non-rigid deformation modeling and texture remapping to ensure alignment with the human body's non-rigid surfaces in 3D scenes. Extensive experiment results in digital and physical environments demonstrate that the adversarial textures generated by our method can effectively mislead the target detection model. Specifically, our method achieves an average attack success rate (ASR) of 70.13% on YOLOv12 in physical scenarios, significantly outperforming existing methods such as T-SEA (21.65%) and AdvTexture (19.70%). Moreover, the proposed method maintains stable ASR across multiple viewpoints and distances, with an average attack success rate exceeding 90% under both frontal and oblique views at a distance of 4 meters. This confirms the method's strong robustness and transferability under multi-angle attacks, varying lighting conditions, and real-world distances. The demo video and code can be obtained at https://github.com/Huangyh98/AdvReal.git.

摘要: 自动驾驶汽车是典型的复杂智能系统，以人工智能为核心。然而，基于深度学习的感知方法极易受到对抗样本的影响，从而导致安全事故。如何在物理世界中生成有效的对抗示例并评估对象检测系统是一个巨大的挑战。在这项研究中，我们提出了一个针对2D和3D领域的统一联合对抗训练框架，该框架同时优化2D图像和3D网格空间中的纹理地图，以更好地解决类内多样性和现实世界的环境变化。该框架包括一个新颖的现实增强对抗模块，具有时空和重新照明映射管道，可以在不同视角下调整对抗补丁和目标服装之间的照明一致性。在此基础上，我们开发了一种真实感增强机制，该机制结合了非刚性变形建模和纹理重新映射，以确保与3D场景中人体的非刚性表面对齐。数字和物理环境中的大量实验结果表明，我们的方法生成的对抗纹理可以有效地误导目标检测模型。具体来说，我们的方法在物理场景下在YOLOv 12上实现了70.13%的平均攻击成功率（ASB），显着优于T-SEA（21.65%）和AdvTexture（19.70%）等现有方法。此外，该方法在多个视角和距离上保持稳定的ASB，在4米距离的正面和斜视下平均攻击成功率超过90%。这证实了该方法在多角度攻击、变化的照明条件和现实世界距离下具有强大的鲁棒性和可移植性。演示视频和代码可在https://github.com/Huangyh98/AdvReal.git上获取。



## **18. QubitHammer: Remotely Inducing Qubit State Change on Superconducting Quantum Computers**

QubitHammer：在超导量子计算机上远程诱导量子位状态变化 quant-ph

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2504.07875v2) [paper-pdf](http://arxiv.org/pdf/2504.07875v2)

**Authors**: Yizhuo Tan, Navnil Choudhury, Kanad Basu, Jakub Szefer

**Abstract**: To address the rapidly growing demand for cloud-based quantum computing, various researchers are proposing shifting from the existing single-tenant model to a multi-tenant model that expands resource utilization and improves accessibility. However, while multi-tenancy enables multiple users to access the same quantum computer, it introduces potential for security and reliability vulnerabilities. It therefore becomes important to investigate these vulnerabilities, especially considering realistic attackers who operate without elevated privileges relative to ordinary users. To address this research need, this paper presents and evaluates QubitHammer, the first attack to demonstrate that an adversary can remotely induce unauthorized changes to a victim's quantum circuit's qubit's state within a multi-tenant model by using custom qubit control pulses that are generated within constraints of the public interfaces and without elevated privileges. Through extensive evaluation on real-world superconducting devices from IBM and Rigetti, this work demonstrates that QubitHammer allows an adversary to significantly change the output distribution of a victim quantum circuit. In the experimentation, variational distance is used to evaluate the magnitude of the changes, and variational distance as high as 0.938 is observed. Cross-platform analysis of QubitHammer on a number of quantum computing devices exposes a fundamental susceptibility in superconducting hardware. Further, QubitHammer was also found to evade all currently proposed defenses aimed at ensuring reliable execution in multi-tenant superconducting quantum systems.

摘要: 为了满足对基于云的量子计算快速增长的需求，多名研究人员提议从现有的单租户模型转向多租户模型，以扩大资源利用率并提高可访问性。然而，虽然多租户允许多个用户访问同一个量子计算机，但它会带来潜在的安全性和可靠性漏洞。因此，调查这些漏洞变得重要，特别是考虑到现实的攻击者相对于普通用户没有更高的特权进行操作。为了满足这一研究需求，本文提出并评估了QubitHammer，这是第一个证明对手可以通过使用自定义量子位控制脉冲在多租户模型中远程诱导对受害者量子电路的量子位状态进行未经授权的更改的攻击。公共接口的约束范围内并且没有提高的特权。通过对IBM和Rigetti的现实世界的超导体设备进行广泛评估，这项工作表明QubitHammer允许对手显着改变受害者量子电路的输出分布。实验中，使用变距来评估变化的幅度，观察到变距高达0.938。对许多量子计算设备上的QubitHammer的跨平台分析揭示了高温硬件的基本易感性。此外，QubitHammer还被发现可以规避当前提出的所有旨在确保多租户超量子系统中可靠执行的防御措施。



## **19. Combating Falsification of Speech Videos with Live Optical Signatures (Extended Version)**

利用实时光学签名打击语音视频伪造（扩展版本） cs.CV

In Proceedings of the 2025 ACM SIGSAC Conference on Computer and  Communications Security (CCS '25). October 13 - 17, 2025, Taipei, Taiwan.  ACM, New York, NY, USA. 19 pages

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2504.21846v2) [paper-pdf](http://arxiv.org/pdf/2504.21846v2)

**Authors**: Hadleigh Schwartz, Xiaofeng Yan, Charles J. Carver, Xia Zhou

**Abstract**: High-profile speech videos are prime targets for falsification, owing to their accessibility and influence. This work proposes VeriLight, a low-overhead and unobtrusive system for protecting speech videos from visual manipulations of speaker identity and lip and facial motion. Unlike the predominant purely digital falsification detection methods, VeriLight creates dynamic physical signatures at the event site and embeds them into all video recordings via imperceptible modulated light. These physical signatures encode semantically-meaningful features unique to the speech event, including the speaker's identity and facial motion, and are cryptographically-secured to prevent spoofing. The signatures can be extracted from any video downstream and validated against the portrayed speech content to check its integrity. Key elements of VeriLight include (1) a framework for generating extremely compact (i.e., 150-bit), pose-invariant speech video features, based on locality-sensitive hashing; and (2) an optical modulation scheme that embeds $>$200 bps into video while remaining imperceptible both in video and live. Experiments on extensive video datasets show VeriLight achieves AUCs $\geq$ 0.99 and a true positive rate of 100% in detecting falsified videos. Further, VeriLight is highly robust across recording conditions, video post-processing techniques, and white-box adversarial attacks on its feature extraction methods. A demonstration of VeriLight is available at https://mobilex.cs.columbia.edu/verilight.

摘要: 由于备受瞩目的演讲视频的可访问性和影响力，因此成为伪造的主要目标。这项工作提出了VeriLight，这是一种低成本且不引人注目的系统，用于保护语音视频免受说话者身份以及嘴唇和面部运动的视觉操纵。与主要的纯数字伪造检测方法不同，VeriLight在活动现场创建动态物理签名，并通过不可感知的调制光将它们嵌入到所有视频记录中。这些物理签名编码语音事件特有的具有语义意义的特征，包括说话者的身份和面部动作，并且经过加密保护以防止欺骗。可以从任何下游视频中提取签名，并针对所描绘的语音内容进行验证，以检查其完整性。VeriLight的关键元素包括（1）用于生成极其紧凑的框架（即，150位）、姿势不变的语音视频特征，基于对位置敏感的哈希;和（2）一种光调制方案，将$> 200 Mbps嵌入视频中，同时在视频和直播中保持不可感知。对大量视频数据集的实验表明，VeriLight在检测伪造视频方面实现了AUCs $\geq $0.99和100%的真阳性率。此外，VeriLight在记录条件、视频后处理技术以及对其特征提取方法的白盒对抗攻击方面具有高度稳健性。VeriLight的演示可在https://mobilex.cs.columbia.edu/verilight上获取。



## **20. Quantum Error Correction in Adversarial Regimes**

对抗机制中的量子误差修正 quant-ph

19 pages, no figure

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08943v1) [paper-pdf](http://arxiv.org/pdf/2509.08943v1)

**Authors**: Rahul Arvind, Nikhil Bansal, Dax Enshan Koh, Tobias Haug, Kishor Bharti

**Abstract**: In adversarial settings, where attackers can deliberately and strategically corrupt quantum data, standard quantum error correction reaches its limits. It can only correct up to half the code distance and must output a unique answer. Quantum list decoding offers a promising alternative. By allowing the decoder to output a short list of possible errors, it becomes possible to tolerate far more errors, even under worst-case noise. But two fundamental questions remain: which quantum codes support list decoding, and can we design decoding schemes that are secure against efficient, computationally bounded adversaries? In this work, we answer both. To identify which codes are list-decodable, we provide a generalized version of the Knill-Laflamme conditions. Then, using tools from quantum cryptography, we build an unambiguous list decoding protocol based on pseudorandom unitaries. Our scheme is secure against any quantum polynomial-time adversary, even across multiple decoding attempts, in contrast to previous schemes. Our approach connects coding theory with complexity-based quantum cryptography, paving the way for secure quantum information processing in adversarial settings.

摘要: 在对抗环境中，攻击者可以故意且战略性地破坏量子数据，标准量子错误纠正就达到了极限。它只能纠正最多一半的代码距离，并且必须输出唯一的答案。量子列表解码提供了一个有希望的替代方案。通过允许解码器输出可能错误的简短列表，即使在最坏情况的噪音下，也可以容忍更多错误。但仍然存在两个基本问题：哪些量子码支持列表解码，以及我们能否设计出安全的解码方案来对抗高效的、计算受限的对手？在这项工作中，我们回答两者。为了确定哪些代码是列表可解码的，我们提供了一个广义版本的Knill-Laflamme条件。然后，利用量子密码学的工具，我们建立了一个明确的列表解码协议的基础上伪随机酉。与以前的方案相比，我们的方案对于任何量子次数的对手来说都是安全的，即使是经过多次解码尝试。我们的方法将编码理论与基于复杂性的量子加密术联系起来，为对抗环境中的安全量子信息处理铺平了道路。



## **21. Corruption-Tolerant Asynchronous Q-Learning with Near-Optimal Rates**

具有近优速率的抗破坏异步Q学习 cs.LG

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08933v1) [paper-pdf](http://arxiv.org/pdf/2509.08933v1)

**Authors**: Sreejeet Maity, Aritra Mitra

**Abstract**: We consider the problem of learning the optimal policy in a discounted, infinite-horizon reinforcement learning (RL) setting where the reward signal is subject to adversarial corruption. Such corruption, which may arise from extreme noise, sensor faults, or malicious attacks, can severely degrade the performance of classical algorithms such as Q-learning. To address this challenge, we propose a new provably robust variant of the Q-learning algorithm that operates effectively even when a fraction of the observed rewards are arbitrarily perturbed by an adversary. Under the asynchronous sampling model with time-correlated data, we establish that despite adversarial corruption, the finite-time convergence rate of our algorithm matches that of existing results for the non-adversarial case, up to an additive term proportional to the fraction of corrupted samples. Moreover, we derive an information-theoretic lower bound revealing that the additive corruption term in our upper bounds is unavoidable.   Next, we propose a variant of our algorithm that requires no prior knowledge of the statistics of the true reward distributions. The analysis of this setting is particularly challenging and is enabled by carefully exploiting a refined Azuma-Hoeffding inequality for almost-martingales, a technical tool that might be of independent interest. Collectively, our contributions provide the first finite-time robustness guarantees for asynchronous Q-learning, bridging a significant gap in robust RL.

摘要: 我们考虑在折扣、无限地平线强化学习（RL）环境中学习最佳策略的问题，其中奖励信号会受到对抗腐败的影响。这种损坏可能由极端噪音、传感器故障或恶意攻击引起，可能会严重降低Q学习等经典算法的性能。为了应对这一挑战，我们提出了一种新的、可证明稳健的Q学习算法变体，即使当一小部分观察到的奖励被对手任意干扰时，该变体也能有效运行。在具有时间相关数据的同步采样模型下，我们确定，尽管存在对抗性破坏，但我们算法的有限时间收敛率与非对抗性情况下的现有结果相匹配，直到与破坏样本比例成比例的附加项。此外，我们推导出一个信息论下限，揭示了上限中的附加腐败项是不可避免的。   接下来，我们提出了算法的一种变体，该变体不需要真实奖励分布的统计数据的先验知识。对这种背景的分析尤其具有挑战性，并且可以通过仔细利用针对几乎马丁斯的改进Azuma-Hoeffding不等式来实现，这是一种可能具有独立兴趣的技术工具。总的来说，我们的贡献为同步Q学习提供了第一个有限时间稳健性保证，弥合了稳健RL方面的重大差距。



## **22. Adversarial Attacks Against Automated Fact-Checking: A Survey**

针对自动事实核查的敌对攻击：一项调查 cs.CL

Accepted to the Main Conference of EMNLP 2025. Resources are  available at  https://github.com/FanzhenLiu/Awesome-Automated-Fact-Checking-Attacks

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08463v1) [paper-pdf](http://arxiv.org/pdf/2509.08463v1)

**Authors**: Fanzhen Liu, Alsharif Abuadbba, Kristen Moore, Surya Nepal, Cecile Paris, Jia Wu, Jian Yang, Quan Z. Sheng

**Abstract**: In an era where misinformation spreads freely, fact-checking (FC) plays a crucial role in verifying claims and promoting reliable information. While automated fact-checking (AFC) has advanced significantly, existing systems remain vulnerable to adversarial attacks that manipulate or generate claims, evidence, or claim-evidence pairs. These attacks can distort the truth, mislead decision-makers, and ultimately undermine the reliability of FC models. Despite growing research interest in adversarial attacks against AFC systems, a comprehensive, holistic overview of key challenges remains lacking. These challenges include understanding attack strategies, assessing the resilience of current models, and identifying ways to enhance robustness. This survey provides the first in-depth review of adversarial attacks targeting FC, categorizing existing attack methodologies and evaluating their impact on AFC systems. Additionally, we examine recent advancements in adversary-aware defenses and highlight open research questions that require further exploration. Our findings underscore the urgent need for resilient FC frameworks capable of withstanding adversarial manipulations in pursuit of preserving high verification accuracy.

摘要: 在错误信息自由传播的时代，事实核查（FC）在验证主张和宣传可靠信息方面发挥着至关重要的作用。虽然自动事实核查（AFC）取得了显着进步，但现有系统仍然容易受到操纵或生成主张、证据或主张-证据对的对抗攻击。这些攻击可能会歪曲事实、误导决策者，并最终破坏FC模型的可靠性。尽管人们对针对AFC系统的对抗性攻击的研究兴趣越来越大，但仍然缺乏对关键挑战的全面、全面的概述。这些挑战包括了解攻击策略、评估当前模型的弹性以及确定增强稳健性的方法。这项调查首次深入审查了针对FC的对抗性攻击，对现有攻击方法进行了分类并评估了它们对AFC系统的影响。此外，我们还研究了对抗意识防御方面的最新进展，并强调了需要进一步探索的开放研究问题。我们的研究结果强调了迫切需要能够承受对抗性操纵以保持高验证准确性的弹性FC框架。



## **23. Dual-Stage Safe Herding Framework for Adversarial Attacker in Dynamic Environment**

动态环境中对抗性攻击者的双阶段安全羊群框架 cs.RO

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08460v1) [paper-pdf](http://arxiv.org/pdf/2509.08460v1)

**Authors**: Wenqing Wang, Ye Zhang, Haoyu Li, Jingyu Wang

**Abstract**: Recent advances in robotics have enabled the widespread deployment of autonomous robotic systems in complex operational environments, presenting both unprecedented opportunities and significant security problems. Traditional shepherding approaches based on fixed formations are often ineffective or risky in urban and obstacle-rich scenarios, especially when facing adversarial agents with unknown and adaptive behaviors. This paper addresses this challenge as an extended herding problem, where defensive robotic systems must safely guide adversarial agents with unknown strategies away from protected areas and into predetermined safe regions, while maintaining collision-free navigation in dynamic environments. We propose a hierarchical hybrid framework based on reach-avoid game theory and local motion planning, incorporating a virtual containment boundary and event-triggered pursuit mechanisms to enable scalable and robust multi-agent coordination. Simulation results demonstrate that the proposed approach achieves safe and efficient guidance of adversarial agents to designated regions.

摘要: 机器人技术的最新进展使自主机器人系统能够在复杂的操作环境中广泛部署，这既带来了前所未有的机会，也带来了重大的安全问题。传统的基于固定队形的牧羊方法在城市和障碍物丰富的场景中通常是无效或危险的，特别是当面对具有未知和自适应行为的对抗性代理时。本文将这一挑战作为一个扩展的羊群问题来解决，防御机器人系统必须安全地引导具有未知策略的对抗代理远离保护区并进入预定的安全区域，同时在动态环境中保持无碰撞导航。我们提出了一个层次化的混合框架的基础上达到避免博弈论和局部运动规划，将一个虚拟的遏制边界和事件触发的追求机制，使可扩展的和强大的多智能体协调。仿真结果表明，所提出的方法可以安全有效地将对抗代理引导到指定区域。



## **24. Robustness of Locally Differentially Private Graph Analysis Against Poisoning**

局部差异私有图分析抗中毒的鲁棒性 cs.CR

47 Pages, 7 Figures. Published in AsiaCCS 2025. Revised to include  additional lower bounds and experimental results obtained after initial  release

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2210.14376v2) [paper-pdf](http://arxiv.org/pdf/2210.14376v2)

**Authors**: Jacob Imola, Amrita Roy Chowdhury, Kamalika Chaudhuri

**Abstract**: Locally differentially private (LDP) graph analysis allows private analysis on a graph that is distributed across multiple users. However, such computations are vulnerable to data poisoning attacks where an adversary can skew the results by submitting malformed data. In this paper, we formally study the impact of poisoning attacks for graph degree estimation protocols under LDP. We make two key technical contributions. First, we observe LDP makes a protocol more vulnerable to poisoning -- the impact of poisoning is worse when the adversary can directly poison their (noisy) responses, rather than their input data. Second, we observe that graph data is naturally redundant -- every edge is shared between two users. Leveraging this data redundancy, we design robust degree estimation protocols under LDP that can significantly reduce the impact of data poisoning and compute degree estimates with high accuracy. We evaluate our proposed robust degree estimation protocols under poisoning attacks on real-world datasets to demonstrate their efficacy in practice.

摘要: 本地差异私有（SDP）图分析允许对分布在多个用户之间的图进行私有分析。然而，此类计算很容易受到数据中毒攻击，对手可以通过提交格式错误的数据来扭曲结果。本文正式研究了中毒攻击对SDP下图度估计协议的影响。我们做出了两项关键技术贡献。首先，我们观察到SDP使协议更容易受到毒害--当对手可以直接毒害他们的（有噪音的）响应而不是他们的输入数据时，毒害的影响会更严重。其次，我们观察到图形数据自然是多余的--每条边都在两个用户之间共享。利用这种数据冗余，我们在SDP下设计了稳健的程度估计协议，可以显着减少数据中毒的影响并高准确度计算程度估计。我们在对现实世界数据集的中毒攻击下评估了我们提出的稳健度估计协议，以证明它们在实践中的有效性。



## **25. CyberRAG: An Agentic RAG cyber attack classification and reporting tool**

CyberRAG：一款大型RAG网络攻击分类和报告工具 cs.CR

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2507.02424v2) [paper-pdf](http://arxiv.org/pdf/2507.02424v2)

**Authors**: Francesco Blefari, Cristian Cosentino, Francesco Aurelio Pironti, Angelo Furfaro, Fabrizio Marozzo

**Abstract**: Intrusion Detection and Prevention Systems (IDS/IPS) in large enterprises can generate hundreds of thousands of alerts per hour, overwhelming analysts with logs requiring rapidly evolving expertise. Conventional machine-learning detectors reduce alert volume but still yield many false positives, while standard Retrieval-Augmented Generation (RAG) pipelines often retrieve irrelevant context and fail to justify predictions. We present CyberRAG, a modular agent-based RAG framework that delivers real-time classification, explanation, and structured reporting for cyber-attacks. A central LLM agent orchestrates: (i) fine-tuned classifiers specialized by attack family; (ii) tool adapters for enrichment and alerting; and (iii) an iterative retrieval-and-reason loop that queries a domain-specific knowledge base until evidence is relevant and self-consistent. Unlike traditional RAG, CyberRAG adopts an agentic design that enables dynamic control flow and adaptive reasoning. This architecture autonomously refines threat labels and natural-language justifications, reducing false positives and enhancing interpretability. It is also extensible: new attack types can be supported by adding classifiers without retraining the core agent. CyberRAG was evaluated on SQL Injection, XSS, and SSTI, achieving over 94\% accuracy per class and a final classification accuracy of 94.92\% through semantic orchestration. Generated explanations reached 0.94 in BERTScore and 4.9/5 in GPT-4-based expert evaluation, with robustness preserved against adversarial and unseen payloads. These results show that agentic, specialist-oriented RAG can combine high detection accuracy with trustworthy, SOC-ready prose, offering a flexible path toward partially automated cyber-defense workflows.

摘要: 大型企业中的入侵检测和预防系统（IDS/IPS）每小时可以生成数十万个警报，需要快速发展的专业知识的日志让分析师不堪重负。传统的机器学习检测器会减少警报量，但仍然会产生许多误报，而标准的检索-增强生成（RAG）管道通常会检索不相关的上下文，并且无法证明预测的合理性。我们介绍了CyberRAG，这是一个基于模块化代理的RAG框架，可为网络攻击提供实时分类、解释和结构化报告。中央LLM代理协调：（i）由攻击家族专门化的微调分类器;（ii）用于丰富和警报的工具适配器;（iii）迭代检索和推理循环，查询特定领域的知识库，直到证据相关且自相容。与传统的RAG不同，CyberRAG采用了一种代理设计，可以实现动态控制流和自适应推理。这种架构可以自动优化威胁标签和自然语言理由，减少误报并增强可解释性。它也是可扩展的：可以通过添加分类器来支持新的攻击类型，而无需重新训练核心代理。CyberRAG在SQL注入，XSS和SSTI上进行了评估，通过语义编排，每个类的准确率超过94%，最终分类准确率为94.92%。生成的解释在BERTScore中达到0.94，在基于GPT-4的专家评估中达到4.9/5，并且在对抗性和不可见的有效负载方面保持了鲁棒性。这些结果表明，面向代理的、面向专家的RAG可以将高检测准确性与值得信赖的、适合SOC的散文结合起来，为实现部分自动化的网络防御工作流程提供灵活的途径。



## **26. Phish-Blitz: Advancing Phishing Detection with Comprehensive Webpage Resource Collection and Visual Integrity Preservation**

Phish-Blitz：通过全面的网页资源收集和视觉完整性保存来推进网络钓鱼检测 cs.CR

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08375v1) [paper-pdf](http://arxiv.org/pdf/2509.08375v1)

**Authors**: Duddu Hriday, Aditya Kulkarni, Vivek Balachandran, Tamal Das

**Abstract**: Phishing attacks are increasingly prevalent, with adversaries creating deceptive webpages to steal sensitive information. Despite advancements in machine learning and deep learning for phishing detection, attackers constantly develop new tactics to bypass detection models. As a result, phishing webpages continue to reach users, particularly those unable to recognize phishing indicators. To improve detection accuracy, models must be trained on large datasets containing both phishing and legitimate webpages, including URLs, webpage content, screenshots, and logos. However, existing tools struggle to collect the required resources, especially given the short lifespan of phishing webpages, limiting dataset comprehensiveness. In response, we introduce Phish-Blitz, a tool that downloads phishing and legitimate webpages along with their associated resources, such as screenshots. Unlike existing tools, Phish-Blitz captures live webpage screenshots and updates resource file paths to maintain the original visual integrity of the webpage. We provide a dataset containing 8,809 legitimate and 5,000 phishing webpages, including all associated resources. Our dataset and tool are publicly available on GitHub, contributing to the research community by offering a more complete dataset for phishing detection.

摘要: 网络钓鱼攻击日益普遍，对手创建欺骗性网页来窃取敏感信息。尽管网络钓鱼检测的机器学习和深度学习取得了进步，但攻击者不断开发新的策略来绕过检测模型。因此，网络钓鱼网页继续影响用户，特别是那些无法识别网络钓鱼指标的用户。为了提高检测准确性，模型必须在包含网络钓鱼和合法网页（包括URL、网页内容、屏幕截图和徽标）的大型数据集上进行训练。然而，现有工具很难收集所需的资源，特别是考虑到网络钓鱼网页的生命周期较短，限制了数据集的全面性。作为回应，我们引入了Phish-Blitz，这是一种下载网络钓鱼和合法网页及其相关资源（例如屏幕截图）的工具。与现有工具不同，Phish-Blitz捕获实时网页截图并更新资源文件路径，以保持网页原始视觉完整性。我们提供的数据集包含8，809个合法网页和5，000个网络钓鱼网页，包括所有相关资源。我们的数据集和工具在GitHub上公开提供，通过提供更完整的网络钓鱼检测数据集来为研究界做出贡献。



## **27. Empirical Security Analysis of Software-based Fault Isolation through Controlled Fault Injection**

通过受控故障注入实现基于软件的故障隔离的经验安全性分析 cs.CR

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.07757v2) [paper-pdf](http://arxiv.org/pdf/2509.07757v2)

**Authors**: Nils Bars, Lukas Bernhard, Moritz Schloegel, Thorsten Holz

**Abstract**: We use browsers daily to access all sorts of information. Because browsers routinely process scripts, media, and executable code from unknown sources, they form a critical security boundary between users and adversaries. A common attack vector is JavaScript, which exposes a large attack surface due to the sheer complexity of modern JavaScript engines. To mitigate these threats, modern engines increasingly adopt software-based fault isolation (SFI). A prominent example is Google's V8 heap sandbox, which represents the most widely deployed SFI mechanism, protecting billions of users across all Chromium-based browsers and countless applications built on Node$.$js and Electron. The heap sandbox splits the address space into two parts: one part containing trusted, security-sensitive metadata, and a sandboxed heap containing memory accessible to untrusted code. On a technical level, the sandbox enforces isolation by removing raw pointers and using translation tables to resolve references to trusted objects. Consequently, an attacker cannot corrupt trusted data even with full control of the sandboxed data, unless there is a bug in how code handles data from the sandboxed heap. Despite their widespread use, such SFI mechanisms have seen little security testing.   In this work, we propose a new testing technique that models the security boundary of modern SFI implementations. Following the SFI threat model, we assume a powerful attacker who fully controls the sandbox's memory. We implement this by instrumenting memory loads originating in the trusted domain and accessing untrusted, attacker-controlled sandbox memory. We then inject faults into the loaded data, aiming to trigger memory corruption in the trusted domain. In a comprehensive evaluation, we identify 19 security bugs in V8 that enable an attacker to bypass the sandbox.

摘要: 我们每天使用浏览器来访问各种信息。由于浏览器经常处理来自未知来源的脚本、媒体和可执行代码，因此它们在用户和对手之间形成了关键的安全边界。一个常见的攻击向量是JavaScript，由于现代JavaScript引擎的复杂性，它暴露了一个巨大的攻击面。为了减轻这些威胁，现代发动机越来越多地采用基于软件的故障隔离（SFI）。一个突出的例子是Google的V8堆沙盒，它代表了部署最广泛的SFI机制，保护所有基于Chromium的浏览器和无数在节点$上构建的应用程序的数十亿用户。$ js和Electron。堆沙箱将地址空间拆分为两部分：一部分包含受信任的、安全敏感的元数据，另一部分包含不受信任代码访问的内存的沙箱堆。在技术层面上，沙箱通过删除原始指针并使用转换表来解析对可信对象的引用来强制隔离。因此，攻击者即使完全控制沙盒数据也无法破坏受信任的数据，除非代码处理沙盒堆中数据的方式存在错误。尽管它们被广泛使用，但这种SFI机制几乎没有进行过安全测试。   在这项工作中，我们提出了一种新的测试技术，该技术对现代SFI实现的安全边界进行建模。遵循SFI威胁模型，我们假设一个强大的攻击者完全控制沙箱的内存。我们通过检测源自受信任域的内存负载并访问不受信任的、攻击者控制的沙箱内存来实现这一点。然后，我们将错误注入加载的数据中，旨在触发可信域中的内存损坏。在全面评估中，我们发现了V8中的19个安全漏洞，这些漏洞使攻击者能够绕过沙箱。



## **28. Adversarial Robustness of Link Sign Prediction in Signed Graphs**

带符号图中链接符号预测的对抗鲁棒性 cs.LG

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2401.10590v3) [paper-pdf](http://arxiv.org/pdf/2401.10590v3)

**Authors**: Jialong Zhou, Xing Ai, Yuni Lai, Tomasz Michalak, Gaolei Li, Jianhua Li, Di Tang, Xingxing Zhang, Mengpei Yang, Kai Zhou

**Abstract**: Signed graphs serve as fundamental data structures for representing positive and negative relationships in social networks, with signed graph neural networks (SGNNs) emerging as the primary tool for their analysis. Our investigation reveals that balance theory, while essential for modeling signed relationships in SGNNs, inadvertently introduces exploitable vulnerabilities to black-box attacks. To showcase this, we propose balance-attack, a novel adversarial strategy specifically designed to compromise graph balance degree, and develop an efficient heuristic algorithm to solve the associated NP-hard optimization problem. While existing approaches attempt to restore attacked graphs through balance learning techniques, they face a critical challenge we term "Irreversibility of Balance-related Information," as restored edges fail to align with original attack targets. To address this limitation, we introduce Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), an innovative framework that combines contrastive learning with balance augmentation techniques to achieve robust graph representations. By maintaining high balance degree in the latent space, BA-SGCL not only effectively circumvents the irreversibility challenge but also significantly enhances model resilience. Extensive experiments across multiple SGNN architectures and real-world datasets demonstrate both the effectiveness of our proposed balance-attack and the superior robustness of BA-SGCL, advancing the security and reliability of signed graph analysis in social networks. Datasets and codes of the proposed framework are at the github repository https://anonymous.4open.science/r/BA-SGCL-submit-DF41/.

摘要: 签名图是表示社交网络中积极和消极关系的基本数据结构，签名图神经网络（SGNN）成为其分析的主要工具。我们的调查表明，平衡理论虽然对于建模SGNN中的签名关系至关重要，但无意中为黑匣子攻击引入了可利用的漏洞。为了展示这一点，我们提出了平衡攻击，这是一种专门设计用于损害图平衡度的新型对抗策略，并开发了一种高效的启发式算法来解决相关的NP难优化问题。虽然现有方法试图通过平衡学习技术恢复受攻击的图形，但它们面临着我们称之为“平衡相关信息的不可逆性”的严峻挑战，因为恢复的边缘无法与原始攻击目标对齐。为了解决这一局限性，我们引入了平衡增强签名图对比学习（BA-SGCL），这是一个创新框架，将对比学习与平衡增强技术相结合，以实现稳健的图表示。BA-SGCL通过保持潜在空间的高度平衡度，不仅有效规避了不可逆转性挑战，而且显着增强了模型的韧性。跨多个SGNN架构和现实世界数据集的广泛实验证明了我们提出的平衡攻击的有效性和BA-SGCL的卓越稳健性，提高了社交网络中签名图分析的安全性和可靠性。拟议框架的数据集和代码位于github存储库https://anonymous.4open.science/r/BA-SGCL-submit-DF41/。



## **29. SAGE: Sample-Aware Guarding Engine for Robust Intrusion Detection Against Adversarial Attacks**

SAGE：针对对抗性攻击的稳健入侵检测的样本感知守护引擎 cs.CR

Under review at IEEE TIFS

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.08091v1) [paper-pdf](http://arxiv.org/pdf/2509.08091v1)

**Authors**: Jing Chen, Onat Gungor, Zhengli Shang, Tajana Rosing

**Abstract**: The rapid proliferation of the Internet of Things (IoT) continues to expose critical security vulnerabilities, necessitating the development of efficient and robust intrusion detection systems (IDS). Machine learning-based intrusion detection systems (ML-IDS) have significantly improved threat detection capabilities; however, they remain highly susceptible to adversarial attacks. While numerous defense mechanisms have been proposed to enhance ML-IDS resilience, a systematic approach for selecting the most effective defense against a specific adversarial attack remains absent. To address this challenge, we previously proposed DYNAMITE, a dynamic defense selection approach that identifies the most suitable defense against adversarial attacks through an ML-driven selection mechanism. Building on this foundation, we propose SAGE (Sample-Aware Guarding Engine), a substantially improved defense algorithm that integrates active learning with targeted data reduction. It employs an active learning mechanism to selectively identify the most informative input samples and their corresponding optimal defense labels, which are then used to train a second-level learner responsible for selecting the most effective defense. This targeted sampling improves computational efficiency, exposes the model to diverse adversarial strategies during training, and enhances robustness, stability, and generalizability. As a result, SAGE demonstrates strong predictive performance across multiple intrusion detection datasets, achieving an average F1-score improvement of 201% over the state-of-the-art defenses. Notably, SAGE narrows the performance gap to the Oracle to just 3.8%, while reducing computational overhead by up to 29x.

摘要: 物联网（IOT）的迅速普及继续暴露出关键的安全漏洞，因此需要开发高效且强大的入侵检测系统（IDS）。基于机器学习的入侵检测系统（ML-IDS）显着提高了威胁检测能力;然而，它们仍然极易受到对抗性攻击。虽然已经提出了多种防御机制来增强ML-IDS的弹性，但仍然缺乏一种系统性的方法来选择针对特定对抗攻击的最有效防御。为了应对这一挑战，我们之前提出了CLARITE，这是一种动态防御选择方法，通过ML驱动的选择机制来识别针对对抗性攻击的最合适的防御。在此基础上，我们提出了SAGE（样本感知守护引擎），这是一种经过大幅改进的防御算法，将主动学习与有针对性的数据简化集成在一起。它采用主动学习机制来选择性地识别信息最丰富的输入样本及其相应的最佳防御标签，然后使用这些标签来训练负责选择最有效防御的二级学习器。这种有针对性的采样提高了计算效率，使模型在训练期间暴露于不同的对抗策略，并增强了稳健性、稳定性和可概括性。因此，SAGE在多个入侵检测数据集中表现出强大的预测性能，与最先进的防御相比，F1评分平均提高了201%。值得注意的是，SAGE将与Oracle的性能差距缩小到仅3.8%，同时将计算负担减少了高达29倍。



## **30. Hammer and Anvil: A Principled Defense Against Backdoors in Federated Learning**

锤子和铁锤：联邦学习中针对后门的原则性防御 cs.LG

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.08089v1) [paper-pdf](http://arxiv.org/pdf/2509.08089v1)

**Authors**: Lucas Fenaux, Zheng Wang, Jacob Yan, Nathan Chung, Florian Kerschbaum

**Abstract**: Federated Learning is a distributed learning technique in which multiple clients cooperate to train a machine learning model. Distributed settings facilitate backdoor attacks by malicious clients, who can embed malicious behaviors into the model during their participation in the training process. These malicious behaviors are activated during inference by a specific trigger. No defense against backdoor attacks has stood the test of time, especially against adaptive attackers, a powerful but not fully explored category of attackers. In this work, we first devise a new adaptive adversary that surpasses existing adversaries in capabilities, yielding attacks that only require one or two malicious clients out of 20 to break existing state-of-the-art defenses. Then, we present Hammer and Anvil, a principled defense approach that combines two defenses orthogonal in their underlying principle to produce a combined defense that, given the right set of parameters, must succeed against any attack. We show that our best combined defense, Krum+, is successful against our new adaptive adversary and state-of-the-art attacks.

摘要: 联邦学习是一种分布式学习技术，多个客户端合作训练机器学习模型。分布式设置促进了恶意客户端的后门攻击，恶意客户端可以在参与训练过程期间将恶意行为嵌入到模型中。这些恶意行为在推理期间由特定触发器激活。没有任何针对后门攻击的防御措施经得起时间的考验，尤其是针对适应性攻击者，这是一种强大但尚未充分探索的攻击者类别。在这项工作中，我们首先设计了一种新的自适应对手，它的能力超过了现有对手，从而产生的攻击只需要20个恶意客户端中的一两个就可以破解现有的最先进防御。然后，我们介绍Hammer和Anvil，这是一种有原则的防御方法，将两种基本原则上垂直的防御结合起来，以产生一种组合防御，在给定正确的参数集的情况下，该防御必须成功抵御任何攻击。我们表明，我们最好的组合防御Krum+能够成功地对抗我们的新自适应对手和最先进的攻击。



## **31. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

大型多模式模型的鲁棒适应用于检索增强仇恨模因检测 cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2502.13061v3) [paper-pdf](http://arxiv.org/pdf/2502.13061v3)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL

摘要: 仇恨模因已成为互联网上的一个重要问题，需要强大的自动化检测系统。虽然大型多模式模型（LSYS）在仇恨模因检测方面表现出了希望，但它们面临着显着的挑战，例如次优的性能和有限的域外概括能力。最近的研究进一步揭示了在这种环境下将监督微调（SFT）和上下文学习应用于LSYS时的局限性。为了解决这些问题，我们提出了一个用于仇恨模因检测的鲁棒适应框架，该框架可以增强领域内准确性和跨领域概括性，同时保留Letts的一般视觉语言能力。分析表明，与SFT模型相比，我们的方法在对抗攻击下实现了更好的鲁棒性。对六个模因分类数据集的实验表明，我们的方法实现了最先进的性能，优于更大的代理系统。此外，与标准SFT相比，我们的方法为解释仇恨内容生成了更高质量的理由，增强了模型的可解释性。代码可访问https://github.com/JingbiaoMei/RGCL



## **32. Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems**

频谱掩蔽和内插攻击（SMIA）：针对语音认证和反欺骗系统的黑匣子对抗攻击 cs.SD

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07677v1) [paper-pdf](http://arxiv.org/pdf/2509.07677v1)

**Authors**: Kamel Kamel, Hridoy Sankar Dutta, Keshav Sood, Sunil Aryal

**Abstract**: Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape.

摘要: 语音认证系统（PAS）使用独特的声音特征进行验证。他们越来越多地融入银行和医疗保健等高安全领域。尽管它们使用深度学习进行了改进，但它们仍面临来自深度造假和对抗攻击等复杂威胁的严重漏洞。现实语音克隆的出现使检测变得复杂，因为系统很难区分真实音频和合成音频。虽然存在反欺骗对策（CM）来减轻这些风险，但许多对策依赖于静态检测模型，这些模型可以被新型对抗方法绕过，从而留下了关键的安全漏洞。为了证明这一漏洞，我们提出了频谱掩蔽和内插攻击（SMIA），这是一种新颖的方法，可以战略性地操纵人工智能生成的音频的听不见的频率区域。通过改变人耳不可感知区域的声音，SMIA创建听起来真实的对抗样本，同时欺骗CM。我们在模拟现实世界条件下对多个任务中针对最先进（SOTA）模型的攻击进行了全面评估。SMIA在对抗组合式增值服务器/CM系统时，至少达到82%的攻击成功率（ASB），对抗独立说话者验证系统至少达到97.5%，对抗措施至少达到100%。这些发现最终证明，当前的安全姿态不足以抵御适应性对抗攻击。这项工作凸显了向下一代防御范式转变的迫切需要，这些防御采用能够随着威胁格局而演变的动态、上下文感知框架。



## **33. Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling**

通过激活引导MCMC采样的可转移直接即时注射 cs.AI

Accepted to EMNLP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07617v1) [paper-pdf](http://arxiv.org/pdf/2509.07617v1)

**Authors**: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang

**Abstract**: Direct Prompt Injection (DPI) attacks pose a critical security threat to Large Language Models (LLMs) due to their low barrier of execution and high potential damage. To address the impracticality of existing white-box/gray-box methods and the poor transferability of black-box methods, we propose an activations-guided prompt injection attack framework. We first construct an Energy-based Model (EBM) using activations from a surrogate model to evaluate the quality of adversarial prompts. Guided by the trained EBM, we employ the token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize adversarial prompts, thereby enabling gradient-free black-box attacks. Experimental results demonstrate our superior cross-model transferability, achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6% improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen task scenarios. Interpretability analysis reveals a correlation between activations and attack effectiveness, highlighting the critical role of semantic patterns in transferable vulnerability exploitation.

摘要: 直接提示注入（DPI）攻击由于其低执行门槛和高潜在危害性，对大型语言模型（LLM）构成了严重的安全威胁。针对现有白盒/灰盒方法的不实用性和黑盒方法的可移植性差的问题，提出了一种激活引导的提示注入攻击框架.首先，我们构建了一个基于能量的模型（EBM）使用代理模型的激活来评估对抗性提示的质量。在经过训练的EBM的指导下，我们采用令牌级马尔可夫链蒙特卡罗（MCMC）采样来自适应地优化对抗性提示，从而实现无梯度的黑盒攻击。实验结果证明了我们卓越的跨模型可移植性，在五种主流LLM中实现了49.6%的攻击成功率（ASB），比人工制作的提示提高了34.6%，并在未见的任务场景中保持了36.6%的ASB。可解释性分析揭示了激活和攻击有效性之间的相关性，凸显了语义模式在可转移漏洞利用中的关键作用。



## **34. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

语音认证和反欺骗系统威胁调查 cs.CR

This paper is submitted to the Computer Science Review

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2508.16843v3) [paper-pdf](http://arxiv.org/pdf/2508.16843v3)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.

摘要: 语音认证发生了重大变化，从依赖手工声学特征的传统系统到可以提取稳健的说话者嵌入的深度学习模型。这一进步扩大了其在金融、智能设备、执法等领域的应用。然而，随着采用率的增加，威胁也随之增加。本调查全面回顾了针对语音认证系统（PAS）和反欺骗对策（CM）的现代威胁格局，包括数据中毒、对抗性、深度伪造和对抗性欺骗攻击。我们按时间顺序追踪语音认证的发展，并研究漏洞如何随着技术进步而演变。对于每种类型的攻击，我们总结了方法论，强调常用的数据集，比较性能和局限性，并使用广泛接受的分类法组织现有文献。通过强调新出现的风险和公开挑战，本调查旨在支持开发更安全、更有弹性的语音认证系统。



## **35. Generating Transferrable Adversarial Examples via Local Mixing and Logits Optimization for Remote Sensing Object Recognition**

通过局部混合和Logits优化生成可传递的对抗示例用于遥感目标识别 cs.CV

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07495v1) [paper-pdf](http://arxiv.org/pdf/2509.07495v1)

**Authors**: Chun Liu, Hailong Wang, Bingqian Zhu, Panpan Ding, Zheng Zheng, Tao Xu, Zhigang Han, Jiayao Wang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, posing significant security threats to their deployment in remote sensing applications. Research on adversarial attacks not only reveals model vulnerabilities but also provides critical insights for enhancing robustness. Although current mixing-based strategies have been proposed to increase the transferability of adversarial examples, they either perform global blending or directly exchange a region in the images, which may destroy global semantic features and mislead the optimization of adversarial examples. Furthermore, their reliance on cross-entropy loss for perturbation optimization leads to gradient diminishing during iterative updates, compromising adversarial example quality. To address these limitations, we focus on non-targeted attacks and propose a novel framework via local mixing and logits optimization. First, we present a local mixing strategy to generate diverse yet semantically consistent inputs. Different from MixUp, which globally blends two images, and MixCut, which stitches images together, our method merely blends local regions to preserve global semantic information. Second, we adapt the logit loss from targeted attacks to non-targeted scenarios, mitigating the gradient vanishing problem of cross-entropy loss. Third, a perturbation smoothing loss is applied to suppress high-frequency noise and enhance transferability. Extensive experiments on FGSCR-42 and MTARSI datasets demonstrate superior performance over 12 state-of-the-art methods across 6 surrogate models. Notably, with ResNet as the surrogate on MTARSI, our method achieves a 17.28% average improvement in black-box attack success rate.

摘要: 深度神经网络（DNN）容易受到对抗攻击，对其在遥感应用中的部署构成重大安全威胁。对对抗攻击的研究不仅揭示了模型的漏洞，而且还为增强稳健性提供了重要见解。尽管目前提出了基于混合的策略来增加对抗性示例的可移植性，但它们要么进行全局混合，要么直接交换图像中的区域，这可能会破坏全局语义特征并误导对抗性示例的优化。此外，它们对交叉熵损失的依赖导致迭代更新期间的梯度减小，从而损害对抗性示例质量。为了解决这些限制，我们专注于非目标攻击，并通过本地混合和逻辑优化提出了一种新颖的框架。首先，我们提出了一种本地混合策略来生成多样化但语义一致的输入。与全局混合两个图像的MixUp和将图像缝合在一起的MixCut不同，我们的方法只是混合局部区域以保留全局语义信息。其次，我们将有针对性攻击的logit损失调整到非有针对性的场景，减轻了交叉熵损失的梯度消失问题。第三，应用扰动平滑损失来抑制高频噪音并增强可移植性。对FGCR-42和MTARSI数据集的广泛实验表明，在6个替代模型中，其性能优于12种最先进方法。值得注意的是，通过ResNet作为MTARSI的替代品，我们的方法使黑匣子攻击成功率平均提高了17.28%。



## **36. Texture- and Shape-based Adversarial Attacks for Overhead Image Vehicle Detection**

基于纹理和形状的对抗性攻击用于头顶图像车辆检测 cs.CV

This version corresponds to the paper accepted for presentation at  ICIP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2412.16358v2) [paper-pdf](http://arxiv.org/pdf/2412.16358v2)

**Authors**: Mikael Yeghiazaryan, Sai Abhishek Siddhartha Namburu, Emily Kim, Stanislav Panev, Celso de Melo, Fernando De la Torre, Jessica K. Hodgins

**Abstract**: Detecting vehicles in aerial images is difficult due to complex backgrounds, small object sizes, shadows, and occlusions. Although recent deep learning advancements have improved object detection, these models remain susceptible to adversarial attacks (AAs), challenging their reliability. Traditional AA strategies often ignore practical implementation constraints. Our work proposes realistic and practical constraints on texture (lowering resolution, limiting modified areas, and color ranges) and analyzes the impact of shape modifications on attack performance. We conducted extensive experiments with three object detector architectures, demonstrating the performance-practicality trade-off: more practical modifications tend to be less effective, and vice versa. We release both code and data to support reproducibility at https://github.com/humansensinglab/texture-shape-adversarial-attacks.

摘要: 由于背景复杂、物体尺寸小、阴影和遮挡，在航空图像中检测车辆很困难。尽管最近的深度学习进步改进了对象检测，但这些模型仍然容易受到对抗攻击（AA）的影响，从而挑战其可靠性。传统的AA策略通常忽视实际实施限制。我们的工作提出了对纹理的现实和实用的限制（降低分辨率、限制修改区域和颜色范围），并分析了形状修改对攻击性能的影响。我们对三种物体检测器架构进行了广泛的实验，展示了性能与实用性的权衡：更实用的修改往往效果不佳，反之亦然。我们在https://github.com/humansensinglab/texture-shape-adversarial-attacks上发布代码和数据以支持可重复性。



## **37. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

Prepared for the Worst：A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm cs.RO

9 pages (6 content, 1 reference, 2 appendix). 7 figures, accepted to  2025 IEEE International Conference on Robotics and Automation (ICRA)

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2403.05666v3) [paper-pdf](http://arxiv.org/pdf/2403.05666v3)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method for assessing the resilience of the ICP algorithm via learning-based, worst-case attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms before deployments is crucial. The ICP algorithm is the standard for lidar-based localization, but its accuracy can be greatly affected by corrupted measurements from various sources, including occlusions, adverse weather, or mechanical sensor issues. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP, our method focuses on finding the maximum possible ICP error that can arise from corrupted measurements at a location. We demonstrate that our perturbation-based adversarial attacks can be used pre-deployment to identify locations on a map where ICP is particularly vulnerable to corruptions in the measurements. With such information, autonomous robots can take safer paths when deployed, to mitigate against their measurements being corrupted. The proposed attack outperforms baselines more than 88% of the time across a wide range of scenarios.

摘要: 本文提出了一种新的方法，通过对激光雷达点云进行基于学习的最坏情况攻击来评估ICP算法的弹性。对于自主导航等安全关键应用，在部署前确保算法的弹性至关重要。ICP算法是基于激光雷达的定位的标准，但其准确性可能会受到来自各种来源的损坏测量结果的极大影响，包括遮挡、恶劣天气或机械传感器问题。不幸的是，国际比较方案的复杂性和反复性使得评估其对腐败的复原力具有挑战性。虽然人们一直在努力创建具有挑战性的数据集并开发模拟来评估ICP的弹性，但我们的方法重点是寻找可能因某个地点的测量结果损坏而产生的最大可能的ICP误差。我们证明，我们的基于扰动的对抗攻击可以在部署前使用来识别地图上ISP特别容易受到测量结果损坏的位置。有了这些信息，自主机器人在部署时可以采取更安全的路径，以防止其测量结果被破坏。在各种场景中，拟议的攻击在88%以上的时间内优于基线。



## **38. When Fine-Tuning is Not Enough: Lessons from HSAD on Hybrid and Adversarial Audio Spoof Detection**

当微调还不够时：HSAD关于混合和对抗性音频欺骗检测的教训 cs.SD

13 pages, 11 figures.This work has been submitted to the IEEE for  possible publication

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07323v1) [paper-pdf](http://arxiv.org/pdf/2509.07323v1)

**Authors**: Bin Hu, Kunyang Huang, Daehan Kwak, Meng Xu, Kuan Huang

**Abstract**: The rapid advancement of AI has enabled highly realistic speech synthesis and voice cloning, posing serious risks to voice authentication, smart assistants, and telecom security. While most prior work frames spoof detection as a binary task, real-world attacks often involve hybrid utterances that mix genuine and synthetic speech, making detection substantially more challenging. To address this gap, we introduce the Hybrid Spoofed Audio Dataset (HSAD), a benchmark containing 1,248 clean and 41,044 degraded utterances across four classes: human, cloned, zero-shot AI-generated, and hybrid audio. Each sample is annotated with spoofing method, speaker identity, and degradation metadata to enable fine-grained analysis. We evaluate six transformer-based models, including spectrogram encoders (MIT-AST, MattyB95-AST) and self-supervised waveform models (Wav2Vec2, HuBERT). Results reveal critical lessons: pretrained models overgeneralize and collapse under hybrid conditions; spoof-specific fine-tuning improves separability but struggles with unseen compositions; and dataset-specific adaptation on HSAD yields large performance gains (AST greater than 97 percent and F1 score is approximately 99 percent), though residual errors persist for complex hybrids. These findings demonstrate that fine-tuning alone is not sufficient-robust hybrid-aware benchmarks like HSAD are essential to expose calibration failures, model biases, and factors affecting spoof detection in adversarial environments. HSAD thus provides both a dataset and an analytic framework for building resilient and trustworthy voice authentication systems.

摘要: 人工智能的快速发展使高度真实的语音合成和语音克隆成为可能，这对语音认证、智能助理和电信安全构成了严重风险。虽然大多数以前的工作框架将欺骗检测作为一项二元任务，但现实世界的攻击通常涉及混合真实语音和合成语音的混合话语，这使得检测更具挑战性。为了解决这一差距，我们引入了混合欺骗音频数据集（HSAD），这是一个基准测试，包含四个类别的1，248个干净话语和41，044个降级话语：人类、克隆、零镜头人工智能生成和混合音频。每个样本都用欺骗方法、说话者身份和降级元数据进行注释，以实现细粒度分析。我们评估了六种基于变压器的模型，包括频谱编码器（MIT-AST、MattyB 95-AST）和自我监督的波形模型（Wave 2 Vec 2、HuBERT）。结果揭示了重要的教训：预训练的模型在混合条件下过度概括和崩溃;针对欺骗的微调提高了可分离性，但在处理看不见的成分时会遇到困难; HSAD上的针对厕所的特定调整产生了很大的性能提升（AST大于97%，F1评分约为99%），尽管复杂的混合体仍然存在残余误差。这些发现表明，仅靠微调并不有效--HSAD等稳健的混合感知基准对于暴露对抗环境中的校准失败、模型偏差和影响欺骗检测的因素至关重要。因此，HSAD提供了数据集和分析框架，用于构建弹性且值得信赖的语音认证系统。



## **39. GRADA: Graph-based Reranking against Adversarial Documents Attack**

GRADA：基于图的重新排名对抗文档攻击 cs.IR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.07546v2) [paper-pdf](http://arxiv.org/pdf/2505.07546v2)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.

摘要: 检索增强生成（RAG）框架通过集成来自检索文档的外部知识来提高大型语言模型（LLM）的准确性，从而克服模型静态内在知识的局限性。然而，这些系统很容易受到对抗性攻击，这些攻击通过引入对抗性但在语义上与查询相似的文档来操纵检索过程。值得注意的是，虽然这些对抗性文档类似于查询，但它们与检索集中的良性文档表现出弱的相似性。因此，我们提出了一个简单而有效的基于图形的对抗性文档攻击重新排名（GRADA）框架，旨在保留检索质量，同时显着降低对手的成功。我们的研究通过在五个LLM上进行的实验来评估我们的方法的有效性：GPT-3.5-Turbo，GPT-4 o，Llama3.1-8b，Llama3.1- 70 b和Qwen2.5- 7 b。我们使用三个数据集来评估性能，来自Natural Questions数据集的结果表明攻击成功率降低了80%，同时保持了最小的准确性损失。



## **40. Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection**

多轮会话中的社会工程个性化攻击：LLM Agent仿真与检测 cs.CR

Accepted as a paper at COLM 2025 Workshop on AI Agents: Capabilities  and Safety

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.15552v2) [paper-pdf](http://arxiv.org/pdf/2503.15552v2)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.

摘要: 会话代理的快速发展，特别是由大型语言模型（LLM）驱动的聊天机器人，对社交媒体平台构成了社会工程（SE）攻击的重大风险。由于这些会话的动态性质，多回合、基于聊天的交互中的SE检测比单实例检测复杂得多。减轻这种威胁的一个关键因素是了解SE攻击的机制，特别是攻击者如何利用漏洞以及受害者的个性特征如何影响他们的易感性。在这项工作中，我们提出了一个LLM-agentic框架，SE-VSim，模拟SE攻击机制，通过生成多轮对话。我们对具有不同性格特征的受害者特工进行建模，以评估心理特征如何影响操纵的易感性。我们使用包含1000多个模拟对话的数据集，检查了冒充招聘人员、资助机构和记者的对手试图提取敏感信息的攻击场景。基于此分析，我们提出了一个概念验证SE-OmniGuard，通过利用受害者个性的先验知识、评估攻击策略以及监控对话中的信息交换以识别潜在的SE尝试，为用户提供个性化保护。



## **41. Adversarial Attacks on Audio Deepfake Detection: A Benchmark and Comparative Study**

音频Deepfake检测的对抗攻击：基准和比较研究 cs.SD

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.07132v1) [paper-pdf](http://arxiv.org/pdf/2509.07132v1)

**Authors**: Kutub Uddin, Muhammad Umar Farooq, Awais Khan, Khalid Mahmood Malik

**Abstract**: The widespread use of generative AI has shown remarkable success in producing highly realistic deepfakes, posing a serious threat to various voice biometric applications, including speaker verification, voice biometrics, audio conferencing, and criminal investigations. To counteract this, several state-of-the-art (SoTA) audio deepfake detection (ADD) methods have been proposed to identify generative AI signatures to distinguish between real and deepfake audio. However, the effectiveness of these methods is severely undermined by anti-forensic (AF) attacks that conceal generative signatures. These AF attacks span a wide range of techniques, including statistical modifications (e.g., pitch shifting, filtering, noise addition, and quantization) and optimization-based attacks (e.g., FGSM, PGD, C \& W, and DeepFool). In this paper, we investigate the SoTA ADD methods and provide a comparative analysis to highlight their effectiveness in exposing deepfake signatures, as well as their vulnerabilities under adversarial conditions. We conducted an extensive evaluation of ADD methods on five deepfake benchmark datasets using two categories: raw and spectrogram-based approaches. This comparative analysis enables a deeper understanding of the strengths and limitations of SoTA ADD methods against diverse AF attacks. It does not only highlight vulnerabilities of ADD methods, but also informs the design of more robust and generalized detectors for real-world voice biometrics. It will further guide future research in developing adaptive defense strategies that can effectively counter evolving AF techniques.

摘要: 生成式人工智能的广泛使用在制作高度逼真的deepfake方面取得了显着的成功，对各种语音生物识别应用构成了严重威胁，包括说话人验证，语音生物识别，音频会议和刑事调查。为了解决这个问题，已经提出了几种最先进的（SoTA）音频深度伪造检测（ADD）方法来识别生成AI签名，以区分真实和深度伪造音频。然而，这些方法的有效性被隐藏生成签名的反取证（AF）攻击严重破坏。这些AF攻击涵盖了广泛的技术，包括统计修改（例如，音调移动、过滤、噪音添加和量化）和基于优化的攻击（例如，FGSM、PVD、C \& W和DeepFool）。在本文中，我们研究了SoTA ADD方法，并提供了比较分析，以强调它们在暴露Deepfake签名方面的有效性以及它们在对抗条件下的漏洞。我们使用两类：原始方法和基于谱图的方法，对五个Deepfake基准数据集进行了广泛的评估。这种比较分析使我们能够更深入地了解SoTA ADD方法针对各种AF发作的优势和局限性。它不仅强调了ADD方法的漏洞，而且还为现实世界语音生物识别技术的更稳健和更通用的检测器的设计提供了信息。它将进一步指导未来开发可有效对抗不断发展的AF技术的自适应防御策略的研究。



## **42. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

攻击LLM和AI代理：针对大型语言模型的广告嵌入攻击 cs.CR

6 pages, 2 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2508.17674v2) [paper-pdf](http://arxiv.org/pdf/2508.17674v2)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.

摘要: 我们引入了广告嵌入攻击（AEA），这是一种新型LLM安全威胁，可以悄悄地将促销或恶意内容注入模型输出和AI代理中。AEA通过两种低成本载体运作：（1）劫持第三方服务分发平台以预先设置对抗提示，以及（2）发布经过攻击者数据微调的后门开源检查点。与降低准确性的传统攻击不同，AEA破坏了信息完整性，导致模型在看起来正常的情况下返回秘密广告、宣传或仇恨言论。我们详细介绍了攻击管道，绘制了五个利益相关者受害者群体，并提出了一种初步的基于预算的自我检查防御，该防御可以减轻这些注入，而无需额外的模型再培训。我们的调查结果揭示了LLM安全方面存在一个紧迫且未充分解决的差距，并呼吁人工智能安全界协调检测、审计和政策响应。



## **43. From Noise to Narrative: Tracing the Origins of Hallucinations in Transformers**

从噪音到叙事：追踪《变形金刚》中幻觉的起源 cs.LG

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06938v1) [paper-pdf](http://arxiv.org/pdf/2509.06938v1)

**Authors**: Praneet Suresh, Jack Stanley, Sonia Joseph, Luca Scimeca, Danilo Bzdok

**Abstract**: As generative AI systems become competent and democratized in science, business, and government, deeper insight into their failure modes now poses an acute need. The occasional volatility in their behavior, such as the propensity of transformer models to hallucinate, impedes trust and adoption of emerging AI solutions in high-stakes areas. In the present work, we establish how and when hallucinations arise in pre-trained transformer models through concept representations captured by sparse autoencoders, under scenarios with experimentally controlled uncertainty in the input space. Our systematic experiments reveal that the number of semantic concepts used by the transformer model grows as the input information becomes increasingly unstructured. In the face of growing uncertainty in the input space, the transformer model becomes prone to activate coherent yet input-insensitive semantic features, leading to hallucinated output. At its extreme, for pure-noise inputs, we identify a wide variety of robustly triggered and meaningful concepts in the intermediate activations of pre-trained transformer models, whose functional integrity we confirm through targeted steering. We also show that hallucinations in the output of a transformer model can be reliably predicted from the concept patterns embedded in transformer layer activations. This collection of insights on transformer internal processing mechanics has immediate consequences for aligning AI models with human values, AI safety, opening the attack surface for potential adversarial attacks, and providing a basis for automatic quantification of a model's hallucination risk.

摘要: 随着生成性人工智能系统在科学、商业和政府中变得称职和民主化，现在迫切需要更深入地了解其失败模式。他们的行为偶尔会出现波动，例如Transformer模型产生幻觉的倾向，阻碍了信任和在高风险领域对新兴人工智能解决方案的采用。在目前的工作中，我们通过稀疏自编码器捕获的概念表示，在输入空间中具有实验控制的不确定性的情况下，建立了如何以及何时在预训练的Transformer模型中出现幻觉。我们的系统实验表明，随着输入信息变得越来越非结构化，Transformer模型使用的语义概念数量也会增加。面对输入空间日益增长的不确定性，Transformer模型变得容易激活连贯但对输入不敏感的语义特征，从而导致幻觉输出。在极端情况下，对于纯噪音输入，我们在预训练的Transformer模型的中间激活中识别了各种鲁棒触发且有意义的概念，我们通过有针对性的引导来确认其功能完整性。我们还表明，可以根据嵌入在Transformer层激活中的概念模式可靠地预测Transformer模型输出中的幻觉。这一系列关于Transformer内部处理机制的见解对调整AI模型与人类价值观、AI安全、打开潜在对抗性攻击的攻击面以及为模型幻觉风险的自动量化提供基础具有直接影响。



## **44. Evaluating the Impact of Adversarial Attacks on Traffic Sign Classification using the LISA Dataset**

使用LISA数据集评估对抗性攻击对交通标志分类的影响 cs.CV

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06835v1) [paper-pdf](http://arxiv.org/pdf/2509.06835v1)

**Authors**: Nabeyou Tadessa, Balaji Iyangar, Mashrur Chowdhury

**Abstract**: Adversarial attacks pose significant threats to machine learning models by introducing carefully crafted perturbations that cause misclassification. While prior work has primarily focused on MNIST and similar datasets, this paper investigates the vulnerability of traffic sign classifiers using the LISA Traffic Sign dataset. We train a convolutional neural network to classify 47 different traffic signs and evaluate its robustness against Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. Our results show a sharp decline in classification accuracy as the perturbation magnitude increases, highlighting the models susceptibility to adversarial examples. This study lays the groundwork for future exploration into defense mechanisms tailored for real-world traffic sign recognition systems.

摘要: 对抗性攻击通过引入精心设计的导致错误分类的扰动，对机器学习模型构成重大威胁。虽然之前的工作主要集中在MNIST和类似数据集，但本文使用LISA Traffic Sign数据集研究了交通标志分类器的漏洞。我们训练一个卷积神经网络来对47个不同的交通标志进行分类，并评估其对快速梯度标志法（FGSM）和投影梯度下降（PVD）攻击的稳健性。我们的结果显示，随着扰动幅度的增加，分类准确性急剧下降，凸显了模型对对抗性示例的敏感性。这项研究为未来探索为现实世界的交通标志识别系统量身定制的防御机制奠定了基础。



## **45. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2501.01872v4) [paper-pdf](http://arxiv.org/pdf/2501.01872v4)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 尽管大型语言模型与人类价值观和道德原则广泛一致，但仍然容易受到利用其推理能力的复杂越狱攻击。现有的安全措施通常检测到明显的恶意意图，但无法解决微妙的、推理驱动的漏洞。在这项工作中，我们引入了POATE（极反相查询生成、对抗模板构建和搜索），这是一种新颖的越狱技术，利用对比推理来引发不道德的反应。POATE精心设计了语义上相反的意图，并将它们与对抗模板集成，以非凡的微妙性引导模型走向有害的输出。我们对参数大小不同的六个不同语言模型家族进行了广泛的评估，以证明攻击的稳健性，与现有方法相比，实现了显着更高的攻击成功率（~44%）。为了解决这个问题，我们提出了意图感知CoT和反向思维CoT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的稳健性并加强了模型对对抗性利用的防御。



## **46. On Hyperparameters and Backdoor-Resistance in Horizontal Federated Learning**

水平联邦学习中的超参数和后门抵抗 cs.CR

To appear in the Proceedings of the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.05192v2) [paper-pdf](http://arxiv.org/pdf/2509.05192v2)

**Authors**: Simon Lachnit, Ghassan Karame

**Abstract**: Horizontal Federated Learning (HFL) is particularly vulnerable to backdoor attacks as adversaries can easily manipulate both the training data and processes to execute sophisticated attacks. In this work, we study the impact of training hyperparameters on the effectiveness of backdoor attacks and defenses in HFL. More specifically, we show both analytically and by means of measurements that the choice of hyperparameters by benign clients does not only influence model accuracy but also significantly impacts backdoor attack success. This stands in sharp contrast with the multitude of contributions in the area of HFL security, which often rely on custom ad-hoc hyperparameter choices for benign clients$\unicode{x2013}$leading to more pronounced backdoor attack strength and diminished impact of defenses. Our results indicate that properly tuning benign clients' hyperparameters$\unicode{x2013}$such as learning rate, batch size, and number of local epochs$\unicode{x2013}$can significantly curb the effectiveness of backdoor attacks, regardless of the malicious clients' settings. We support this claim with an extensive robustness evaluation of state-of-the-art attack-defense combinations, showing that carefully chosen hyperparameters yield across-the-board improvements in robustness without sacrificing main task accuracy. For example, we show that the 50%-lifespan of the strong A3FL attack can be reduced by 98.6%, respectively$\unicode{x2013}$all without using any defense and while incurring only a 2.9 percentage points drop in clean task accuracy.

摘要: 水平联邦学习（HFL）特别容易受到后门攻击，因为对手可以轻松操纵训练数据和流程来执行复杂的攻击。在这项工作中，我们研究了训练超参数对HFL中后门攻击和防御有效性的影响。更具体地说，我们通过分析和测量表明，良性客户端对超参数的选择不仅会影响模型准确性，还会显着影响后门攻击的成功。这与HFL安全领域的众多贡献形成鲜明对比，HFL安全领域通常依赖于良性客户端$\unicode{x2013}$的自定义临时超参数选择，导致后门攻击强度更明显，防御影响减弱。我们的结果表明，正确调整良性客户端的超参数$\unicode{x2013}$，例如学习率、批量大小和本地纪元数量$\unicode{x2013}$可以显着抑制后门攻击的有效性，无论恶意客户端的设置如何。我们通过对最先进的攻击-防御组合的广泛鲁棒性评估来支持这一说法，表明精心选择的超参数可以在不牺牲主要任务准确性的情况下全面提高鲁棒性。例如，我们表明，在不使用任何防御的情况下，强A3 FL攻击的50%寿命可以分别缩短98.6%$\unicode{x2013}$all，同时仅导致干净任务准确性下降2.9个百分点。



## **47. Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem**

注意您的服务器：对CP生态系统的寄生工具链攻击的系统研究 cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06572v1) [paper-pdf](http://arxiv.org/pdf/2509.06572v1)

**Authors**: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

**Abstract**: Large language models (LLMs) are increasingly integrated with external systems through the Model Context Protocol (MCP), which standardizes tool invocation and has rapidly become a backbone for LLM-powered applications. While this paradigm enhances functionality, it also introduces a fundamental security shift: LLMs transition from passive information processors to autonomous orchestrators of task-oriented toolchains, expanding the attack surface, elevating adversarial goals from manipulating single outputs to hijacking entire execution flows. In this paper, we reveal a new class of attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy Disclosure (MCP-UPD). These attacks require no direct victim interaction; instead, adversaries embed malicious instructions into external data sources that LLMs access during legitimate tasks. The malicious logic infiltrates the toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection, and Privacy Disclosure, culminating in stealthy exfiltration of private data. Our root cause analysis reveals that MCP lacks both context-tool isolation and least-privilege enforcement, enabling adversarial instructions to propagate unchecked into sensitive tool invocations. To assess the severity, we design MCP-SEC and conduct the first large-scale security census of the MCP ecosystem, analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP ecosystem is rife with exploitable gadgets and diverse attack methods, underscoring systemic risks in MCP platforms and the urgent need for defense mechanisms in LLM-integrated environments.

摘要: 大型语言模型（LLM）通过模型上下文协议（HCP）越来越多地与外部系统集成，该协议使工具调用同步化，并已迅速成为LLM支持的应用程序的支柱。虽然这种范式增强了功能，但它也引入了根本性的安全转变：LLM从被动信息处理器过渡到面向任务的工具链的自主编排，扩大了攻击面，将对抗目标从操纵单个输出提升到劫持整个执行流。在本文中，我们揭示了一类新的攻击，即寄生工具链攻击，实例化为LCP无意隐私泄露（MCP-UPD）。这些攻击不需要受害者直接互动;相反，对手会将恶意指令嵌入到LLM在合法任务期间访问的外部数据源中。恶意逻辑渗透到工具链中，并分三个阶段展开：寄生摄入、隐私收集和隐私披露，最终导致私人数据的秘密泄露。我们的根本原因分析表明，LCP缺乏上下文工具隔离和最低特权强制执行，使得对抗指令能够不受限制地传播到敏感工具调用中。为了评估严重性，我们设计了MCP-SEC，并对LCP生态系统进行了首次大规模安全普查，分析了1，360台服务器上的12，230个工具。我们的研究结果表明，LCP生态系统中充斥着可利用的小工具和多样化的攻击方法，凸显了LCP平台的系统性风险以及LLM集成环境中对防御机制的迫切需求。



## **48. Robustness and accuracy of mean opinion scores with hard and soft outlier detection**

具有硬异常值和软异常值检测的平均意见分数的稳健性和准确性 eess.IV

Accepted for 17th International Conference on Quality of Multimedia  Experience (QoMEX'25), September 2025, Madrid, Spain

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06554v1) [paper-pdf](http://arxiv.org/pdf/2509.06554v1)

**Authors**: Dietmar Saupe, Tim Bleile

**Abstract**: In subjective assessment of image and video quality, observers rate or compare selected stimuli. Before calculating the mean opinion scores (MOS) for these stimuli from the ratings, it is recommended to identify and deal with outliers that may have given unreliable ratings. Several methods are available for this purpose, some of which have been standardized. These methods are typically based on statistics and sometimes tested by introducing synthetic ratings from artificial outliers, such as random clickers. However, a reliable and comprehensive approach is lacking for comparative performance analysis of outlier detection methods. To fill this gap, this work proposes and applies an empirical worst-case analysis as a general solution. Our method involves evolutionary optimization of an adversarial black-box attack on outlier detection algorithms, where the adversary maximizes the distortion of scale values with respect to ground truth. We apply our analysis to several hard and soft outlier detection methods for absolute category ratings and show their differing performance in this stress test. In addition, we propose two new outlier detection methods with low complexity and excellent worst-case performance. Software for adversarial attacks and data analysis is available.

摘要: 在图像和视频质量的主观评估中，观察者对选定的刺激进行评级或比较。在根据评级计算这些刺激的平均意见分数（MOS）之前，建议识别并处理可能给出不可靠评级的离群值。为此目的，有多种方法可用，其中一些已被标准化。这些方法通常基于统计数据，有时通过引入来自人工异常值（例如随机点击器）的合成评级来进行测试。然而，缺乏可靠且全面的方法来对异常值检测方法进行比较性能分析。为了填补这一空白，这项工作提出并应用了实证最坏情况分析作为一般解决方案。我们的方法涉及对异常值检测算法的对抗性黑匣子攻击的进化优化，其中对手最大化尺度值相对于地面真相的失真。我们将我们的分析应用于绝对类别评级的几种硬异常值和软异常值检测方法，并展示它们在此压力测试中的不同表现。此外，我们还提出了两种新的异常值检测方法，具有低复杂度和优异的最坏情况性能。提供对抗攻击和数据分析软件。



## **49. IGAff: Benchmarking Adversarial Iterative and Genetic Affine Algorithms on Deep Neural Networks**

IGAff：深度神经网络上的对抗迭代和遗传仿射算法基准 cs.CV

10 pages, 7 figures, Accepted at ECAI 2025 (28th European Conference  on Artificial Intelligence)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06459v1) [paper-pdf](http://arxiv.org/pdf/2509.06459v1)

**Authors**: Sebastian-Vasile Echim, Andrei-Alexandru Preda, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: Deep neural networks currently dominate many fields of the artificial intelligence landscape, achieving state-of-the-art results on numerous tasks while remaining hard to understand and exhibiting surprising weaknesses. An active area of research focuses on adversarial attacks, which aim to generate inputs that uncover these weaknesses. However, this proves challenging, especially in the black-box scenario where model details are inaccessible. This paper explores in detail the impact of such adversarial algorithms on ResNet-18, DenseNet-121, Swin Transformer V2, and Vision Transformer network architectures. Leveraging the Tiny ImageNet, Caltech-256, and Food-101 datasets, we benchmark two novel black-box iterative adversarial algorithms based on affine transformations and genetic algorithms: 1) Affine Transformation Attack (ATA), an iterative algorithm maximizing our attack score function using random affine transformations, and 2) Affine Genetic Attack (AGA), a genetic algorithm that involves random noise and affine transformations. We evaluate the performance of the models in the algorithm parameter variation, data augmentation, and global and targeted attack configurations. We also compare our algorithms with two black-box adversarial algorithms, Pixle and Square Attack. Our experiments yield better results on the image classification task than similar methods in the literature, achieving an accuracy improvement of up to 8.82%. We provide noteworthy insights into successful adversarial defenses and attacks at both global and targeted levels, and demonstrate adversarial robustness through algorithm parameter variation.

摘要: 深度神经网络目前在人工智能领域的许多领域占据主导地位，在众多任务上实现了最先进的结果，同时仍然难以理解并表现出令人惊讶的弱点。一个活跃的研究领域专注于对抗攻击，旨在生成揭露这些弱点的输入。然而，事实证明这具有挑战性，尤其是在无法访问模型详细信息的黑匣子场景中。本文详细探讨了这种对抗算法对ResNet-18、DenseNet-121、Swin Transformer V2和Vision Transformer网络架构的影响。利用Tiny ImageNet、Caltech-256和Food-101数据集，我们对两种基于仿射变换和遗传算法的新型黑匣子迭代对抗算法进行基准测试：1）仿射变换攻击（ATA），一种使用随机仿射变换最大化我们的攻击分数函数的迭代算法，2）仿射遗传攻击（AGA），一种涉及随机噪音和仿射变换的遗传算法。我们评估模型在算法参数变化、数据增强以及全局和有针对性的攻击配置方面的性能。我们还将我们的算法与两种黑匣子对抗算法Pixle和Square Attack进行了比较。我们的实验在图像分类任务上产生了比文献中的类似方法更好的结果，实现了高达8.82%的准确性提高。我们对全球和目标层面上的成功对抗防御和攻击提供了值得注意的见解，并通过算法参数变化展示了对抗鲁棒性。



## **50. Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?**

Mask-GCG：敌对后缀中的所有代币都是越狱攻击所必需的吗？ cs.CL

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06350v1) [paper-pdf](http://arxiv.org/pdf/2509.06350v1)

**Authors**: Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, Yaoyuan Zhang, Zhengmin Yu, Wenxin Zhang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks.

摘要: 对大型语言模型（LLM）的越狱攻击已经展示了各种成功的方法，攻击者通过这些方法操纵模型来生成他们旨在避免的有害响应。其中，贪婪坐标梯度（GCG）已成为一种通用且有效的方法，可以优化后缀中的标记以生成可越狱的提示。虽然已经提出了GCG的几种改进变体，但它们都依赖于固定长度的后缀。然而，这些后缀中的潜在冗余仍有待探索。在这项工作中，我们提出了Mask-GCG，这是一种即插即用方法，它采用可学习的标记掩蔽来识别后缀内有影响力的标记。我们的方法增加了高影响力位置上代币的更新概率，同时修剪低影响力位置上的代币。与GCG相比，这种修剪不仅减少了冗余，还减少了梯度空间的大小，从而降低了计算负担并缩短了实现成功攻击所需的时间。我们通过将Mask-GCG应用于原始GCG和几种改进的变体来评估Mask-GCG。实验结果表明，后缀中的大多数令牌对攻击成功做出了显着贡献，并且修剪少数低影响令牌不会影响损失值或损害攻击成功率（ASB），从而揭示了LLM提示中的令牌冗余。我们的研究结果为从越狱攻击的角度开发高效且可解释的LLM提供了见解。



