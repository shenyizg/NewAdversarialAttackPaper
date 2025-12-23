# Latest Adversarial Attack Papers
**update at 2025-12-23 09:37:34**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Robustness of Vision in Open Foundation Models**

开放基金会模型中的对抗性稳健性 cs.CV

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17902v1) [paper-pdf](https://arxiv.org/pdf/2512.17902v1)

**Authors**: Jonathon Fox, William J Buchanan, Pavlos Papadopoulos

**Abstract**: With the increase in deep learning, it becomes increasingly difficult to understand the model in which AI systems can identify objects. Thus, an adversary could aim to modify an image by adding unseen elements, which will confuse the AI in its recognition of an entity. This paper thus investigates the adversarial robustness of LLaVA-1.5-13B and Meta's Llama 3.2 Vision-8B-2. These are tested for untargeted PGD (Projected Gradient Descent) against the visual input modality, and empirically evaluated on the Visual Question Answering (VQA) v2 dataset subset. The results of these adversarial attacks are then quantified using the standard VQA accuracy metric. This evaluation is then compared with the accuracy degradation (accuracy drop) of LLaVA and Llama 3.2 Vision. A key finding is that Llama 3.2 Vision, despite a lower baseline accuracy in this setup, exhibited a smaller drop in performance under attack compared to LLaVA, particularly at higher perturbation levels. Overall, the findings confirm that the vision modality represents a viable attack vector for degrading the performance of contemporary open-weight VLMs, including Meta's Llama 3.2 Vision. Furthermore, they highlight that adversarial robustness does not necessarily correlate directly with standard benchmark performance and may be influenced by underlying architectural and training factors.

摘要: 随着深度学习的增加，理解人工智能系统识别对象的模型变得越来越困难。因此，对手可能会通过添加不可见的元素来修改图像，这会混淆人工智能对实体的识别。因此，本文研究了LLaVA-1.5- 13 B和Meta的Llama 3.2 Vision-8B-2的对抗鲁棒性。针对视觉输入模式对这些进行了非目标PVD（投影梯度下降）测试，并在视觉问题回答（VQA）v2数据集子集上进行了经验评估。然后使用标准VQA准确性度量来量化这些对抗性攻击的结果。然后将该评估与LLaVA和Llama 3.2 Vision的精度下降（精度下降）进行比较。一个关键的发现是，尽管Llama 3.2 Vision在这种设置中的基线精度较低，但与LLaVA相比，在攻击下的性能下降较小，特别是在较高的扰动水平下。总的来说，研究结果证实，视觉模态代表了一个可行的攻击载体，用于降低当代开放重量VLMs的性能，包括Meta的Llama 3.2 Vision。此外，他们强调，对抗稳健性不一定与标准基准性能直接相关，并且可能受到基础架构和训练因素的影响。



## **2. STAR: Semantic-Traffic Alignment and Retrieval for Zero-Shot HTTPS Website Fingerprinting**

STAR：零镜头HTTPS网站指纹识别的语义流量对齐和检索 cs.CR

Accepted by IEEE INFOCOM 2026. Camera-ready version

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17667v1) [paper-pdf](https://arxiv.org/pdf/2512.17667v1)

**Authors**: Yifei Cheng, Yujia Zhu, Baiyang Li, Xinhao Deng, Yitong Cai, Yaochen Ren, Qingyun Liu

**Abstract**: Modern HTTPS mechanisms such as Encrypted Client Hello (ECH) and encrypted DNS improve privacy but remain vulnerable to website fingerprinting (WF) attacks, where adversaries infer visited sites from encrypted traffic patterns. Existing WF methods rely on supervised learning with site-specific labeled traces, which limits scalability and fails to handle previously unseen websites. We address these limitations by reformulating WF as a zero-shot cross-modal retrieval problem and introducing STAR. STAR learns a joint embedding space for encrypted traffic traces and crawl-time logic profiles using a dual-encoder architecture. Trained on 150K automatically collected traffic-logic pairs with contrastive and consistency objectives and structure-aware augmentation, STAR retrieves the most semantically aligned profile for a trace without requiring target-side traffic during training. Experiments on 1,600 unseen websites show that STAR achieves 87.9 percent top-1 accuracy and 0.963 AUC in open-world detection, outperforming supervised and few-shot baselines. Adding an adapter with only four labeled traces per site further boosts top-5 accuracy to 98.8 percent. Our analysis reveals intrinsic semantic-traffic alignment in modern web protocols, identifying semantic leakage as the dominant privacy risk in encrypted HTTPS traffic. We release STAR's datasets and code to support reproducibility and future research.

摘要: 加密客户端Hello（ECT）和加密DNS等现代HTTPS机制可以改善隐私，但仍然容易受到网站指纹识别（WF）攻击，对手可以从加密流量模式中推断访问的网站。现有的WF方法依赖于具有特定站点标记痕迹的监督学习，这限制了可扩展性并且无法处理以前未见过的网站。我们通过将WF重新定义为零触发跨模式检索问题并引入STAR来解决这些限制。STAR使用双编码器架构学习加密流量轨迹和爬行时间逻辑配置文件的联合嵌入空间。STAR在具有对比性和一致性目标和结构感知增强的150 K自动收集的流量逻辑对上进行训练，可以检索跟踪的语义最一致的配置文件，而无需在训练期间获取目标端流量。在1，600个未看过的网站上进行的实验表明，STAR在开放世界检测中实现了87.9%的前一准确率和0.963的AUC，优于监督和少镜头基线。添加每个站点仅带有四个标记轨迹的适配器，可以进一步将前5名的准确率提高到98.8%。我们的分析揭示了现代Web协议中固有的语义流量一致性，将语义泄露确定为加密HTTPS流量中的主要隐私风险。我们发布STAR的数据集和代码以支持重现性和未来的研究。



## **3. SCAR: Semantic Cardiac Adversarial Representation via Spatiotemporal Manifold Optimization in ECG**

SWR：心电图中通过时空Maniform优化的语义心脏对抗表示 eess.SP

13 pages, 5 figures

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17423v1) [paper-pdf](https://arxiv.org/pdf/2512.17423v1)

**Authors**: Shunbo Jia, Caizhi Liao

**Abstract**: Deep learning models for Electrocardiogram (ECG) analysis have achieved expert-level performance but remain vulnerable to adversarial attacks. However, applying Universal Adversarial Perturbations (UAP) to ECG signals presents a unique challenge: standard imperceptible noise constraints (e.g., 10 uV) fail to generate effective universal attacks due to the high inter-subject variability of cardiac waveforms. Furthermore, traditional "invisible" attacks are easily dismissed by clinicians as technical artifacts, failing to compromise the human-in-the-loop diagnostic pipeline. In this study, we propose SCAR (Semantic Cardiac Adversarial Representation), a novel UAP framework tailored to bypass the clinical "Human Firewall." Unlike traditional approaches, SCAR integrates spatiotemporal smoothing (W=25, approx. 50ms), spectral consistency (<15 Hz), and anatomical amplitude constraints (<0.2 mV) directly into the gradient optimization manifold.   Results: We benchmarked SCAR against a rigorous baseline (Standard Universal DeepFool with post-hoc physiological filtering). While the baseline suffers a performance collapse (~16% success rate on transfer tasks), SCAR maintains robust transferability (58.09% on ResNet) and achieves 82.46% success on the source model. Crucially, clinical analysis reveals an emergent targeted behavior: SCAR specifically converges to forging Myocardial Infarction features (90.2% misdiagnosis) by mathematically reconstructing pathological ST-segment elevations. Finally, we demonstrate that SCAR serves a dual purpose: it not only functions as a robust data augmentation strategy for Hybrid Adversarial Training, offering optimal clinical defense, but also provides effective educational samples for training clinicians to recognize low-cost, AI-targeted semantic forgeries.

摘要: 用于心电图（心电图）分析的深度学习模型已实现专家级性能，但仍然容易受到对抗攻击。然而，将通用对抗微扰（UAP）应用于心电图信号提出了一个独特的挑战：标准不可感知的噪音约束（例如，10 uV）由于心脏波的受试者间变异性高，无法产生有效的普遍发作。此外，传统的“隐形”攻击很容易被临床医生视为技术产物，无法损害人在环诊断管道。在这项研究中，我们提出了SVR（语义心脏对抗表示），这是一种新型UAP框架，旨在绕过临床“人肉防火墙”。“与传统方法不同，SWR集成了时空平滑（W=25，约为1）50 ms）、频谱一致性（<15 Hz）和解剖幅度约束（<0.2 mA）直接输入到梯度优化总管中。   结果：我们根据严格的基线（标准Universal DeepFool，带事后生理过滤）对SWR进行了基准测试。虽然基线性能崩溃（传输任务的成功率~16%），但SWR保持了强大的可传输性（ResNet上为58.09%），并在源模型上实现了82.46%的成功率。至关重要的是，临床分析揭示了一种紧急的目标行为：SCA特别收敛于通过数学重建病理性ST段抬高来伪造心肌梗塞特征（90.2%误诊）。最后，我们证明了SWR具有双重目的：它不仅充当混合对抗训练的强大数据增强策略，提供最佳临床防御，而且还为培训临床医生识别低成本、针对人工智能的语义伪造提供有效的教育样本。



## **4. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

有害在线内容的对抗稳健检测：计算设计科学方法 cs.LG

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17367v1) [paper-pdf](https://arxiv.org/pdf/2512.17367v1)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.

摘要: 社交媒体平台受到仇恨言论、错误信息和极端主义言论等有害内容的困扰。机器学习（ML）模型被广泛采用来检测此类内容;然而，它们仍然极易受到对抗攻击，其中恶意用户会巧妙地修改文本以逃避检测。因此，增强对抗鲁棒性至关重要，需要检测器能够抵御各种攻击（可概括性），同时保持高的总体准确性。然而，同时实现最佳概括性和准确性是一项挑战。遵循计算设计科学范式，本研究采用顺序方法，首先提出了一种新颖的框架（基于大语言模型的样本生成和聚合，LLM-LGA），通过识别文本对抗攻击的关键不变性并利用它们来确保框架内实例化的检测器具有很强的概括性。其次，我们实例化我们的检测器（对抗鲁棒有害在线内容检测器，ARHOCD）具有三个新颖的设计组件来提高检测准确性：（1）利用其互补优势的多个基本检测器的集成;（2）一种新颖的权重分配方法，其基于每个样本的可预测性和每个碱基检测器的能力动态调整权重，权重使用领域知识初始化并通过Bayesian推理更新;以及（3）一种新颖的对抗训练策略，迭代优化基本检测器和权重分配器。我们解决了现有对抗鲁棒性增强研究的几个局限性，并在跨越仇恨言论、谣言和极端主义内容的三个数据集中对ARHOCD进行了实证评估。结果表明，ARHOCD具有很强的概括性，并提高了对抗条件下的检测准确性。



## **5. Practical Framework for Privacy-Preserving and Byzantine-robust Federated Learning**

隐私保护和拜占庭稳健联邦学习的实用框架 cs.CR

Accepted for publication in IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17254v1) [paper-pdf](https://arxiv.org/pdf/2512.17254v1)

**Authors**: Baolei Zhang, Minghong Fang, Zhuqing Liu, Biao Yi, Peizhao Zhou, Yuan Wang, Tong Li, Zheli Liu

**Abstract**: Federated Learning (FL) allows multiple clients to collaboratively train a model without sharing their private data. However, FL is vulnerable to Byzantine attacks, where adversaries manipulate client models to compromise the federated model, and privacy inference attacks, where adversaries exploit client models to infer private data. Existing defenses against both backdoor and privacy inference attacks introduce significant computational and communication overhead, creating a gap between theory and practice. To address this, we propose ABBR, a practical framework for Byzantine-robust and privacy-preserving FL. We are the first to utilize dimensionality reduction to speed up the private computation of complex filtering rules in privacy-preserving FL. Additionally, we analyze the accuracy loss of vector-wise filtering in low-dimensional space and introduce an adaptive tuning strategy to minimize the impact of malicious models that bypass filtering on the global model. We implement ABBR with state-of-the-art Byzantine-robust aggregation rules and evaluate it on public datasets, showing that it runs significantly faster, has minimal communication overhead, and maintains nearly the same Byzantine-resilience as the baselines.

摘要: 联合学习（FL）允许多个客户协作训练模型，而无需共享其私人数据。然而，FL很容易受到拜占庭攻击（对手操纵客户端模型以损害联邦模型）和隐私推断攻击（对手利用客户端模型来推断私人数据）。针对后门和隐私推断攻击的现有防御引入了大量的计算和通信负担，在理论和实践之间造成了差距。为了解决这个问题，我们提出了ABBR，这是拜占庭稳健且隐私保护FL的实用框架。我们是第一个利用降维来加速隐私保护FL中复杂过滤规则的私人计算的人。此外，我们分析了低密度下逐向过滤的准确性损失维度空间并引入自适应调整策略，以最大限度地减少绕过过滤的恶意模型对全局模型的影响。我们使用最先进的拜占庭稳健聚合规则实施ABBR，并在公共数据集上对其进行评估，结果表明它运行速度明显更快，通信负担最小，并且保持与基线几乎相同的拜占庭弹性。



## **6. Biosecurity-Aware AI: Agentic Risk Auditing of Soft Prompt Attacks on ESM-Based Variant Predictors**

具有生物安全意识的人工智能：对基于ESM的变体预测器的软提示攻击的量化风险审计 cs.CR

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17146v1) [paper-pdf](https://arxiv.org/pdf/2512.17146v1)

**Authors**: Huixin Zhan

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated remarkable success in variant effect prediction. However, their security and robustness under adversarial manipulation remain largely unexplored. To address this gap, we introduce the Secure Agentic Genomic Evaluator (SAGE), an agentic framework for auditing the adversarial vulnerabilities of GFMs. SAGE functions through an interpretable and automated risk auditing loop. It injects soft prompt perturbations, monitors model behavior across training checkpoints, computes risk metrics such as AUROC and AUPR, and generates structured reports with large language model-based narrative explanations. This agentic process enables continuous evaluation of embedding-space robustness without modifying the underlying model. Using SAGE, we find that even state-of-the-art GFMs like ESM2 are sensitive to targeted soft prompt attacks, resulting in measurable performance degradation. These findings reveal critical and previously hidden vulnerabilities in genomic foundation models, showing the importance of agentic risk auditing in securing biomedical applications such as clinical variant interpretation.

摘要: 基因组基础模型（GFM），例如进化规模建模（ESM），在变异效应预测方面取得了显着的成功。然而，它们在对抗操纵下的安全性和稳健性在很大程度上仍未得到探索。为了解决这一差距，我们引入了安全统计基因组评估器（SAGE），这是一个用于审计GFM对抗性漏洞的代理框架。SAGE通过可解释和自动化的风险审计循环发挥作用。它注入软提示扰动，监控训练检查点的模型行为，计算AUROC和AUPR等风险指标，并生成具有基于大型语言模型的叙述性解释的结构化报告。这个代理过程能够连续评估嵌入空间稳健性，而无需修改基础模型。使用SAGE，我们发现即使是ESM 2等最先进的GFM也对有针对性的软提示攻击敏感，从而导致可衡量的性能下降。这些发现揭示了基因组基础模型中关键且先前隐藏的漏洞，表明代理风险审计在确保临床变体解释等生物医学应用方面的重要性。



## **7. Adversarial VR: An Open-Source Testbed for Evaluating Adversarial Robustness of VR Cybersickness Detection and Mitigation**

对抗VR：评估VR网络病检测和缓解对抗鲁棒性的开源测试平台 cs.CR

Published in the 2025 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct)

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.17029v1) [paper-pdf](https://arxiv.org/pdf/2512.17029v1)

**Authors**: Istiak Ahmed, Ripan Kumar Kundu, Khaza Anuarul Hoque

**Abstract**: Deep learning (DL)-based automated cybersickness detection methods, along with adaptive mitigation techniques, can enhance user comfort and interaction. However, recent studies show that these DL-based systems are susceptible to adversarial attacks; small perturbations to sensor inputs can degrade model performance, trigger incorrect mitigation, and disrupt the user's immersive experience (UIX). Additionally, there is a lack of dedicated open-source testbeds that evaluate the robustness of these systems under adversarial conditions, limiting the ability to assess their real-world effectiveness. To address this gap, this paper introduces Adversarial-VR, a novel real-time VR testbed for evaluating DL-based cybersickness detection and mitigation strategies under adversarial conditions. Developed in Unity, the testbed integrates two state-of-the-art (SOTA) DL models: DeepTCN and Transformer, which are trained on the open-source MazeSick dataset, for real-time cybersickness severity detection and applies a dynamic visual tunneling mechanism that adjusts the field-of-view based on model outputs. To assess robustness, we incorporate three SOTA adversarial attacks: MI-FGSM, PGD, and C&W, which successfully prevent cybersickness mitigation by fooling DL-based cybersickness models' outcomes. We implement these attacks using a testbed with a custom-built VR Maze simulation and an HTC Vive Pro Eye headset, and we open-source our implementation for widespread adoption by VR developers and researchers. Results show that these adversarial attacks are capable of successfully fooling the system. For instance, the C&W attack results in a $5.94x decrease in accuracy for the Transformer-based cybersickness model compared to the accuracy without the attack.

摘要: 基于深度学习（DL）的自动网络病检测方法以及自适应缓解技术可以增强用户的舒适度和交互性。然而，最近的研究表明，这些基于DL的系统很容易受到对抗性攻击;对传感器输入的微小扰动可能会降低模型性能、触发错误的缓解并扰乱用户的沉浸式体验（UIX）。此外，缺乏专门的开源测试平台来评估这些系统在对抗条件下的稳健性，从而限制了评估其现实世界有效性的能力。为了解决这一差距，本文引入了Adversarial-VR，这是一个新型的实时VR测试平台，用于评估对抗条件下基于DL的网络病检测和缓解策略。该测试平台在Unity中开发，集成了两个最先进的（SOTA）DL模型：DeepTCN和Transformer，它们在开源的MazeSick数据集上进行训练，用于实时晕机严重程度检测，并应用动态视觉隧道机制，根据模型输出调整视野。为了评估稳健性，我们结合了三种SOTA对抗性攻击：MI-FGSM、PGD和C & W，它们通过欺骗基于DL的网络病模型的结果，成功预防了网络病缓解。我们使用带有定制VR Maze模拟和HTC Vive Pro Eye耳机的测试台来实施这些攻击，并且我们开源了我们的实施，供VR开发人员和研究人员广泛采用。结果表明，这些对抗性攻击能够成功欺骗系统。例如，与没有攻击的准确性相比，C & W攻击导致基于Transformer的网络病模型的准确性下降了5.94倍。



## **8. PrivateXR: Defending Privacy Attacks in Extended Reality Through Explainable AI-Guided Differential Privacy**

PrivateXR：通过可解释的人工智能引导的差异隐私在延展实境中防御隐私攻击 cs.CR

Published in the IEEE ISMAR 2025 conference

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16851v1) [paper-pdf](https://arxiv.org/pdf/2512.16851v1)

**Authors**: Ripan Kumar Kundu, Istiak Ahmed, Khaza Anuarul Hoque

**Abstract**: The convergence of artificial AI and XR technologies (AI XR) promises innovative applications across many domains. However, the sensitive nature of data (e.g., eye-tracking) used in these systems raises significant privacy concerns, as adversaries can exploit these data and models to infer and leak personal information through membership inference attacks (MIA) and re-identification (RDA) with a high success rate. Researchers have proposed various techniques to mitigate such privacy attacks, including differential privacy (DP). However, AI XR datasets often contain numerous features, and applying DP uniformly can introduce unnecessary noise to less relevant features, degrade model accuracy, and increase inference time, limiting real-time XR deployment. Motivated by this, we propose a novel framework combining explainable AI (XAI) and DP-enabled privacy-preserving mechanisms to defend against privacy attacks. Specifically, we leverage post-hoc explanations to identify the most influential features in AI XR models and selectively apply DP to those features during inference. We evaluate our XAI-guided DP approach on three state-of-the-art AI XR models and three datasets: cybersickness, emotion, and activity classification. Our results show that the proposed method reduces MIA and RDA success rates by up to 43% and 39%, respectively, for cybersickness tasks while preserving model utility with up to 97% accuracy using Transformer models. Furthermore, it improves inference time by up to ~2x compared to traditional DP approaches. To demonstrate practicality, we deploy the XAI-guided DP AI XR models on an HTC VIVE Pro headset and develop a user interface (UI), namely PrivateXR, allowing users to adjust privacy levels (e.g., low, medium, high) while receiving real-time task predictions, protecting user privacy during XR gameplay.

摘要: 人工AI和XR技术（AI XR）的融合有望在许多领域实现创新应用。然而，数据的敏感性（例如，这些系统中使用的眼睛跟踪）引发了严重的隐私问题，因为对手可以利用这些数据和模型通过成员资格推断攻击（MIA）和重新识别（RDA）来推断和泄露个人信息，成功率很高。研究人员提出了各种技术来缓解此类隐私攻击，包括差异隐私（DP）。然而，AI XR数据集通常包含大量特征，统一应用DP可能会给不太相关的特征带来不必要的噪音，降低模型准确性，并增加推理时间，从而限制实时XR部署。出于此动机，我们提出了一种新颖的框架，将可解释人工智能（XAI）和DP支持的隐私保护机制相结合，以抵御隐私攻击。具体来说，我们利用事后解释来识别AI XR模型中最有影响力的特征，并在推理期间选择性地将DP应用于这些特征。我们在三个最先进的AI XR模型和三个数据集上评估了我们的XAI引导的DP方法：网络疾病、情绪和活动分类。我们的结果表明，对于网瘾任务，所提出的方法将MIA和RDA成功率分别降低了高达43%和39%，同时使用Transformer模型保留了高达97%的准确性的模型效用。此外，与传统DP方法相比，它将推理时间缩短了约2倍。为了展示实用性，我们在HTC VIVE Pro耳机上部署了XAI引导的DP AI XR模型，并开发了用户界面（UI），即PrivateXR，允许用户调整隐私级别（例如，低、中、高），同时接收实时任务预测，在XR游戏过程中保护用户隐私。



## **9. Misspecified Crame-Rao Bound for AoA Estimation at a ULA under a Spoofing Attack**

欺骗攻击下的预设AoA估计的Crame-Rao界被错误指定 eess.SP

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16735v1) [paper-pdf](https://arxiv.org/pdf/2512.16735v1)

**Authors**: Sotiris Skaperas, Arsenia Chorti

**Abstract**: A framework is presented for analyzing the impact of active attacks to location-based physical layer authentication (PLA) using the machinery of misspecified Cramér--Rao bound (MCRB). In this work, we focus on the MCRB in the angle-of-arrival (AoA) based authentication of a single antenna user when the verifier posseses an $M$ antenna element uniform linear array (ULA), assuming deterministic pilot signals; in our system model the presence of a spoofing adversary with an arbitrary number $L$ of antenna elements is assumed. We obtain a closed-form expression for the MCRB and demonstrate that the attack introduces in it a penalty term compared to the classic CRB, which does not depend on the signal-to-noise ratio (SNR) but on the adversary's location, the array geometry and the attacker precoding vector.

摘要: 提出了一个框架，用于分析主动攻击对使用错误指定的Cramér-Rao界（MCRB）机制的基于位置的物理层认证（PLA）的影响。在这项工作中，我们重点关注当验证者拥有$M$天线元件均匀线性阵列（RST）时，单天线用户基于到达角（AoA）的认证中的MCRB，假设导频信号确定性;在我们的系统模型中，假设存在具有任意数量$L$的欺骗对手。我们获得了MCRB的封闭形式表达，并证明与经典CRB相比，攻击在其中引入了一个罚项，该罚项不取决于信噪比（SNR），而是取决于对手的位置、阵列几何形状和攻击者预编码载体。



## **10. Dual-View Inference Attack: Machine Unlearning Amplifies Privacy Exposure**

双视图推理攻击：机器遗忘放大隐私暴露 cs.LG

Accepeted by AAAI2026

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16126v1) [paper-pdf](https://arxiv.org/pdf/2512.16126v1)

**Authors**: Lulu Xue, Shengshan Hu, Linqiang Qian, Peijin Guo, Yechao Zhang, Minghui Li, Yanjun Zhang, Dayong Ye, Leo Yu Zhang

**Abstract**: Machine unlearning is a newly popularized technique for removing specific training data from a trained model, enabling it to comply with data deletion requests. While it protects the rights of users requesting unlearning, it also introduces new privacy risks. Prior works have primarily focused on the privacy of data that has been unlearned, while the risks to retained data remain largely unexplored. To address this gap, we focus on the privacy risks of retained data and, for the first time, reveal the vulnerabilities introduced by machine unlearning under the dual-view setting, where an adversary can query both the original and the unlearned models. From an information-theoretic perspective, we introduce the concept of {privacy knowledge gain} and demonstrate that the dual-view setting allows adversaries to obtain more information than querying either model alone, thereby amplifying privacy leakage. To effectively demonstrate this threat, we propose DVIA, a Dual-View Inference Attack, which extracts membership information on retained data using black-box queries to both models. DVIA eliminates the need to train an attack model and employs a lightweight likelihood ratio inference module for efficient inference. Experiments across different datasets and model architectures validate the effectiveness of DVIA and highlight the privacy risks inherent in the dual-view setting.

摘要: 机器取消学习是一种新流行的技术，用于从训练后的模型中删除特定的训练数据，使其能够遵守数据删除请求。虽然它保护了请求取消学习的用户的权利，但它也带来了新的隐私风险。之前的作品主要关注未了解的数据的隐私，而保留数据的风险在很大程度上仍未被探索。为了解决这一差距，我们重点关注保留数据的隐私风险，并首次揭示了双视图环境下机器取消学习引入的漏洞，其中对手可以查询原始和未学习的模型。从信息论的角度，我们引入了{隐私知识获得}的概念，并证明双视角设置允许对手获得比单独查询任何一个模型更多的信息，从而放大了隐私泄露。为了有效地证明这种威胁，我们提出了DVIA，这是一种双视图推理攻击，它使用对两个模型的黑匣子查询来提取保留数据的成员信息。DVIA消除了训练攻击模型的需要，并采用轻量级似然比推理模块进行高效推理。跨不同数据集和模型架构的实验验证了DVIA的有效性，并强调了双视图设置中固有的隐私风险。



## **11. Autoencoder-based Denoising Defense against Adversarial Attacks on Object Detection**

基于自动编码器的目标检测对抗攻击去噪防御 cs.CR

7 pages, 2 figures

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16123v1) [paper-pdf](https://arxiv.org/pdf/2512.16123v1)

**Authors**: Min Geun Song, Gang Min Kim, Woonmin Kim, Yongsik Kim, Jeonghyun Sim, Sangbeom Park, Huy Kang Kim

**Abstract**: Deep learning-based object detection models play a critical role in real-world applications such as autonomous driving and security surveillance systems, yet they remain vulnerable to adversarial examples. In this work, we propose an autoencoder-based denoising defense to recover object detection performance degraded by adversarial perturbations. We conduct adversarial attacks using Perlin noise on vehicle-related images from the COCO dataset, apply a single-layer convolutional autoencoder to remove the perturbations, and evaluate detection performance using YOLOv5. Our experiments demonstrate that adversarial attacks reduce bbox mAP from 0.2890 to 0.1640, representing a 43.3% performance degradation. After applying the proposed autoencoder defense, bbox mAP improves to 0.1700 (3.7% recovery) and bbox mAP@50 increases from 0.2780 to 0.3080 (10.8% improvement). These results indicate that autoencoder-based denoising can provide partial defense against adversarial attacks without requiring model retraining.

摘要: 基于深度学习的对象检测模型在自动驾驶和安全监控系统等现实世界应用中发挥着至关重要的作用，但它们仍然容易受到对抗示例的影响。在这项工作中，我们提出了一种基于自动编码器的去噪防御，以恢复因对抗性扰动而降低的对象检测性能。我们使用Perlin噪音对COCO数据集的车辆相关图像进行对抗攻击，应用单层卷积自动编码器来消除扰动，并使用YOLOv 5评估检测性能。我们的实验表明，对抗性攻击将bbox mAP从0.2890降低到0.1640，代表43.3%的性能下降。应用拟议的自动编码器防御后，bbox mAP提高到0.1700（恢复率3.7%），bbox mAP@50从0.2780提高到0.3080（改善10.8%）。这些结果表明，基于自动编码器的去噪可以在不需要模型重新训练的情况下提供部分防御对抗攻击。



## **12. From Risk to Resilience: Towards Assessing and Mitigating the Risk of Data Reconstruction Attacks in Federated Learning**

从风险到韧性：评估和缓解联邦学习中数据重建攻击的风险 cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15460v1) [paper-pdf](https://arxiv.org/pdf/2512.15460v1)

**Authors**: Xiangrui Xu, Zhize Li, Yufei Han, Bin Wang, Jiqiang Liu, Wei Wang

**Abstract**: Data Reconstruction Attacks (DRA) pose a significant threat to Federated Learning (FL) systems by enabling adversaries to infer sensitive training data from local clients. Despite extensive research, the question of how to characterize and assess the risk of DRAs in FL systems remains unresolved due to the lack of a theoretically-grounded risk quantification framework. In this work, we address this gap by introducing Invertibility Loss (InvLoss) to quantify the maximum achievable effectiveness of DRAs for a given data instance and FL model. We derive a tight and computable upper bound for InvLoss and explore its implications from three perspectives. First, we show that DRA risk is governed by the spectral properties of the Jacobian matrix of exchanged model updates or feature embeddings, providing a unified explanation for the effectiveness of defense methods. Second, we develop InvRE, an InvLoss-based DRA risk estimator that offers attack method-agnostic, comprehensive risk evaluation across data instances and model architectures. Third, we propose two adaptive noise perturbation defenses that enhance FL privacy without harming classification accuracy. Extensive experiments on real-world datasets validate our framework, demonstrating its potential for systematic DRA risk evaluation and mitigation in FL systems.

摘要: 数据重建攻击（NPS）使对手能够从本地客户端推断敏感训练数据，从而对联邦学习（FL）系统构成重大威胁。尽管进行了广泛的研究，但由于缺乏基于理论的风险量化框架，如何描述和评估FL系统中DSA风险的问题仍未得到解决。在这项工作中，我们通过引入可逆损失（InvLoss）来量化给定数据实例和FL模型的DSA的最大可实现有效性来解决这一差距。我们推导出InvLoss的紧密且可计算的上限，并从三个角度探讨其含义。首先，我们表明，数据库风险由交换模型更新或特征嵌入的雅可比矩阵的谱属性决定，为防御方法的有效性提供了统一的解释。其次，我们开发InvRE，这是一种基于InvLoss的Inver-risk估计器，它提供跨数据实例和模型架构的攻击方法不可知的全面风险评估。第三，我们提出了两种自适应噪音扰动防御措施，可以在不损害分类准确性的情况下增强FL隐私。对现实世界数据集的广泛实验验证了我们的框架，展示了其在FL系统中进行系统性NPS风险评估和缓解的潜力。



## **13. Talking to the Airgap: Exploiting Radio-Less Embedded Devices as Radio Receivers**

与Airgap交谈：利用无线电嵌入式设备作为无线电接收器 cs.CR

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15387v1) [paper-pdf](https://arxiv.org/pdf/2512.15387v1)

**Authors**: Paul Staat, Daniel Davidovich, Christof Paar

**Abstract**: Intelligent electronics are deeply embedded in critical infrastructures and must remain reliable, particularly against deliberate attacks. To minimize risks and impede remote compromise, sensitive systems can be physically isolated from external networks, forming an airgap. Yet, airgaps can still be infiltrated by capable adversaries gaining code execution. Prior research has shown that attackers can then attempt to wirelessly exfiltrate data across the airgap by exploiting unintended radio emissions. In this work, we demonstrate reversal of this link: malicious code execution on embedded devices can enable wireless infiltration of airgapped systems without any hardware modification. In contrast to previous infiltration methods that depend on dedicated sensors (e.g., microphones, LEDs, or temperature sensors) or require strict line-of-sight, we show that unmodified, sensor-less embedded devices can inadvertently act as radio receivers. This phenomenon stems from parasitic RF sensitivity in PCB traces and on-chip analog-to-digital converters (ADCs), allowing external transmissions to be received and decoded entirely in software.   Across twelve commercially available embedded devices and two custom prototypes, we observe repeatable reception in the 300-1000 MHz range, with detectable signal power as low as 1 mW. To this end, we propose a systematic methodology to identify device configurations that foster such radio sensitivities and comprehensively evaluate their feasibility for wireless data reception. Exploiting these sensitivities, we demonstrate successful data reception over tens of meters, even in non-line-of-sight conditions and show that the reception sensitivities accommodate data rates of up to 100 kbps. Our findings reveal a previously unexplored command-and-control vector for air-gapped systems while challenging assumptions about their inherent isolation. [shortened]

摘要: 智能电子产品深深嵌入关键基础设施中，必须保持可靠性，特别是针对故意攻击。为了最大限度地减少风险并阻止远程破坏，敏感系统可以与外部网络物理隔离，形成气间隙。然而，漏洞仍然可能被有能力的对手渗透，以获得代码执行。之前的研究表明，攻击者可以尝试通过利用无意的无线电发射来通过空气间隙无线传输数据。在这项工作中，我们展示了这个链接的逆转：嵌入式设备上的恶意代码执行可以在不修改任何硬件的情况下实现对空间隙系统的无线渗透。与之前依赖于专用传感器的渗透方法相反（例如，麦克风、LED或温度传感器）或需要严格的视线，我们表明未经修改的无传感器嵌入式设备可能会无意中充当无线电接收器。这种现象源于PCB走线和片内模拟数字转换器（ADC）中的寄生RF灵敏度，允许外部传输完全通过软件接收和解码。   在十二种市售嵌入式设备和两种定制原型中，我们观察到300-1000 MHz范围内的可重复接收，可检测信号功率低至1 MW。为此，我们提出了一种系统性的方法来识别能够促进此类无线电敏感性的设备配置，并全面评估其无线数据接收的可行性。利用这些灵敏度，我们展示了即使在非视线条件下也能在数十米范围内成功接收数据，并表明接收灵敏度可适应高达100 kMbps的数据速率。我们的研究结果揭示了一种以前未探索的气间隙系统的命令和控制载体，同时挑战了有关其固有隔离性的假设。[缩短]



## **14. Bounty Hunter: Autonomous, Comprehensive Emulation of Multi-Faceted Adversaries**

赏金猎人：多面对手的自主、全面模拟 cs.CR

15 pages, 9 figures

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15275v1) [paper-pdf](https://arxiv.org/pdf/2512.15275v1)

**Authors**: Louis Hackländer-Jansen, Rafael Uetz, Martin Henze

**Abstract**: Adversary emulation is an essential procedure for cybersecurity assessments such as evaluating an organization's security posture or facilitating structured training and research in dedicated environments. To allow for systematic and time-efficient assessments, several approaches from academia and industry have worked towards the automation of adversarial actions. However, they exhibit significant limitations regarding autonomy, tactics coverage, and real-world applicability. Consequently, adversary emulation remains a predominantly manual task requiring substantial human effort and security expertise - even amidst the rise of Large Language Models. In this paper, we present Bounty Hunter, an automated adversary emulation method, designed and implemented as an open-source plugin for the popular adversary emulation platform Caldera, that enables autonomous emulation of adversaries with multi-faceted behavior while providing a wide coverage of tactics. To this end, it realizes diverse adversarial behavior, such as different levels of detectability and varying attack paths across repeated emulations. By autonomously compromising a simulated enterprise network, Bounty Hunter showcases its ability to achieve given objectives without prior knowledge of its target, including pre-compromise, initial compromise, and post-compromise attack tactics. Overall, Bounty Hunter facilitates autonomous, comprehensive, and multi-faceted adversary emulation to help researchers and practitioners in performing realistic and time-efficient security assessments, training exercises, and intrusion detection research.

摘要: Adobile仿真是网络安全评估的重要程序，例如评估组织的安全态势或促进专用环境中的结构化培训和研究。为了进行系统性且高效的评估，学术界和工业界的多种方法致力于对抗行动的自动化。然而，它们在自主性、战术覆盖范围和现实世界的适用性方面表现出显着的局限性。因此，对手模拟仍然是一项主要的手动任务，需要大量的人力和安全专业知识--即使在大型语言模型的兴起下也是如此。在本文中，我们介绍了Bounty Hunter，这是一种自动化的对手模拟方法，作为流行的对手模拟平台Caldera的开源插件设计和实现，它能够自主模拟具有多方面行为的对手，同时提供广泛的战术覆盖范围。为此，它实现了多样化的对抗行为，例如不同级别的可检测性和重复模拟中的不同攻击路径。通过自主破坏模拟企业网络，Bounty Hunter展示了其在不了解其目标的情况下实现给定目标的能力，包括破坏前、初始破坏和破坏后攻击策略。总体而言，Bounty Hunter促进了自主、全面和多方面的对手模拟，以帮助研究人员和从业者执行现实且省时的安全评估、培训练习和入侵检测研究。



## **15. An Efficient Gradient-Based Inference Attack for Federated Learning**

一种有效的联邦学习基于对象的推理攻击 cs.LG

This paper was supported by the TRUMPET project, funded by the European Union under Grant Agreement No. 101070038

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15143v1) [paper-pdf](https://arxiv.org/pdf/2512.15143v1)

**Authors**: Pablo Montaña-Fernández, Ines Ortega-Fernandez

**Abstract**: Federated Learning is a machine learning setting that reduces direct data exposure, improving the privacy guarantees of machine learning models. Yet, the exchange of model updates between the participants and the aggregator can still leak sensitive information. In this work, we present a new gradient-based membership inference attack for federated learning scenarios that exploits the temporal evolution of last-layer gradients across multiple federated rounds. Our method uses the shadow technique to learn round-wise gradient patterns of the training records, requiring no access to the private dataset, and is designed to consider both semi-honest and malicious adversaries (aggregators or data owners). Beyond membership inference, we also provide a natural extension of the proposed attack to discrete attribute inference by contrasting gradient responses under alternative attribute hypotheses. The proposed attacks are model-agnostic, and therefore applicable to any gradient-based model and can be applied to both classification and regression settings. We evaluate the attack on CIFAR-100 and Purchase100 datasets for membership inference and on Breast Cancer Wisconsin for attribute inference. Our findings reveal strong attack performance and comparable computational and memory overhead in membership inference when compared to another attack from the literature. The obtained results emphasize that multi-round federated learning can increase the vulnerability to inference attacks, that aggregators pose a more substantial threat than data owners, and that attack performance is strongly influenced by the nature of the training dataset, with richer, high-dimensional data leading to stronger leakage than simpler tabular data.

摘要: 联合学习是一种机器学习设置，可以减少直接数据暴露，改善机器学习模型的隐私保证。然而，参与者和聚合器之间的模型更新交换仍然可能泄露敏感信息。在这项工作中，我们针对联邦学习场景提出了一种新的基于梯度的成员资格推断攻击，该攻击利用了多个联邦回合中最后一层梯度的时间演变。我们的方法使用阴影技术来学习训练记录的全方位梯度模式，不需要访问私人数据集，并且旨在考虑半诚实和恶意对手（聚合器或数据所有者）。除了成员资格推断，我们还提供了一个自然的扩展，提出的攻击离散属性推理对比梯度响应下的替代属性假设。所提出的攻击是与模型无关的，因此适用于任何基于梯度的模型，并且可以应用于分类和回归设置。我们评估了对CIFAR-100和Purchase 100数据集的攻击，用于成员推断和对乳腺癌威斯康星州的属性推断。我们的研究结果揭示了强大的攻击性能和可比的计算和内存开销的成员推断相比，从文献中的另一种攻击。所获得的结果强调，多轮联邦学习可能会增加推理攻击的脆弱性，聚合器比数据所有者构成更大的威胁，攻击性能受到训练数据集性质的强烈影响，更丰富、多维的数据会导致比更简单的表格数据更强的泄漏。



## **16. Quantifying Return on Security Controls in LLM Systems**

量化LLM系统中安全控制的回报 cs.CR

13 pages, 9 figures, 3 tables

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15081v1) [paper-pdf](https://arxiv.org/pdf/2512.15081v1)

**Authors**: Richard Helder Moulton, Austin O'Brien, John D. Hastings

**Abstract**: Although large language models (LLMs) are increasingly used in security-critical workflows, practitioners lack quantitative guidance on which safeguards are worth deploying. This paper introduces a decision-oriented framework and reproducible methodology that together quantify residual risk, convert adversarial probe outcomes into financial risk estimates and return-on-control (RoC) metrics, and enable monetary comparison of layered defenses for LLM-based systems. A retrieval-augmented generation (RAG) service is instantiated using the DeepSeek-R1 model over a corpus containing synthetic personally identifiable information (PII), and subjected to automated attacks with Garak across five vulnerability classes: PII leakage, latent context injection, prompt injection, adversarial attack generation, and divergence. For each (vulnerability, control) pair, attack success probabilities are estimated via Laplace's Rule of Succession and combined with loss triangle distributions, calibrated from public breach-cost data, in 10,000-run Monte Carlo simulations to produce loss exceedance curves and expected losses. Three widely used mitigations, attribute-based access control (ABAC); named entity recognition (NER) redaction using Microsoft Presidio; and NeMo Guardrails, are then compared to a baseline RAG configuration. The baseline system exhibits very high attack success rates (>= 0.98 for PII, latent injection, and prompt injection), yielding a total simulated expected loss of $313k per attack scenario. ABAC collapses success probabilities for PII and prompt-related attacks to near zero and reduces the total expected loss by ~94%, achieving an RoC of 9.83. NER redaction likewise eliminates PII leakage and attains an RoC of 5.97, while NeMo Guardrails provides only marginal benefit (RoC of 0.05).

摘要: 尽管大型语言模型（LLM）越来越多地用于安全关键工作流程，但从业者缺乏关于哪些保障措施值得部署的量化指导。本文介绍了一个面向决策的框架和可重复的方法论，它们共同量化剩余风险，将对抗性调查结果转化为财务风险估计和控制回报（RoC）指标，并实现基于LLM的系统的分层防御的货币比较。检索增强生成（RAG）服务使用DeepSeek-R1模型在包含合成个人可识别信息（PRI）的数据库上实例化，并在五个漏洞类别上受到Garak的自动攻击：PIP泄露、潜在上下文注入、提示注入、对抗攻击生成和分歧。对于每个（漏洞、控制）对，攻击成功概率是通过拉普拉斯继承规则估计的，并结合根据公共违规成本数据校准的损失三角分布，在10，000次运行的蒙特卡洛模拟中生成损失延迟曲线和预期损失。然后将三种广泛使用的缓解措施：基于属性的访问控制（ABAC）;使用Microsoft Presidio的命名实体识别（NER）编辑;和NeMo Guardrails与基线RAG配置进行比较。基线系统表现出非常高的攻击成功率（PRI、潜伏注射和即时注射>= 0.98），每个攻击场景的模拟预期损失总额为31.3万美元。ABAC将PRI和预算相关攻击的成功概率降至接近零，并将总预期损失降低约94%，实现了9.83的RoC。NER编辑同样消除了PRI泄漏，并获得了5.97的RoC，而NeMo Guardrails仅提供了边际效益（RoC为0.05）。



## **17. Cloud Security Leveraging AI: A Fusion-Based AISOC for Malware and Log Behaviour Detection**

利用人工智能的云安全：基于融合的AISOC，用于恶意软件和日志行为检测 cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14935v1) [paper-pdf](https://arxiv.org/pdf/2512.14935v1)

**Authors**: Nnamdi Philip Okonkwo, Lubna Luxmi Dhirani

**Abstract**: Cloud Security Operations Center (SOC) enable cloud governance, risk and compliance by providing insights visibility and control. Cloud SOC triages high-volume, heterogeneous telemetry from elastic, short-lived resources while staying within tight budgets. In this research, we implement an AI-Augmented Security Operations Center (AISOC) on AWS that combines cloud-native instrumentation with ML-based detection. The architecture uses three Amazon EC2 instances: Attacker, Defender, and Monitoring. We simulate a reverse-shell intrusion with Metasploit, and Filebeat forwards Defender logs to an Elasticsearch and Kibana stack for analysis. We train two classifiers, a malware detector built on a public dataset and a log-anomaly detector trained on synthetically augmented logs that include adversarial variants. We calibrate and fuse the scores to produce multi-modal threat intelligence and triage activity into NORMAL, SUSPICIOUS, and HIGH\_CONFIDENCE\_ATTACK. On held-out tests the fusion achieves strong macro-F1 (up to 1.00) under controlled conditions, though performance will vary in noisier and more diverse environments. These results indicate that simple, calibrated fusion can enhance cloud SOC capabilities in constrained, cost-sensitive setups.

摘要: 云安全运营中心（SOC）通过提供洞察、可见性和控制来实现云治理、风险和合规性。云SOC从弹性、短暂的资源中对大容量、异类遥感进行分类，同时保持在紧张的预算范围内。在这项研究中，我们在AWS上实现了人工智能增强安全运营中心（AISOC），该中心将云原生工具与基于ML的检测相结合。该架构使用三个Amazon EC 2实例：Attacker、Defender和Monitoring。我们使用Metasploit模拟反向Shell入侵，Filebat将Defender日志转发到Elasticsearch和Kibana堆栈进行分析。我们训练两个分类器，一个是基于公共数据集构建的恶意软件检测器，另一个是基于包括对抗性变体的合成增强日志训练的日志异常检测器。我们校准和融合分数，以产生多模式威胁情报，并将活动分类为正常、可疑和高\_信心\_攻击。在进行测试中，融合在受控条件下实现了强大的宏F1（高达1.00），尽管性能在噪音更大和更多样化的环境中会有所不同。这些结果表明，简单的校准融合可以在受约束、成本敏感的设置中增强云SOC能力。



## **18. PerProb: Indirectly Evaluating Memorization in Large Language Models**

PerProb：间接评估大型语言模型中的精简化 cs.CR

Accepted at APSEC 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14600v1) [paper-pdf](https://arxiv.org/pdf/2512.14600v1)

**Authors**: Yihan Liao, Jacky Keung, Xiaoxue Ma, Jingyu Zhang, Yicheng Sun

**Abstract**: The rapid advancement of Large Language Models (LLMs) has been driven by extensive datasets that may contain sensitive information, raising serious privacy concerns. One notable threat is the Membership Inference Attack (MIA), where adversaries infer whether a specific sample was used in model training. However, the true impact of MIA on LLMs remains unclear due to inconsistent findings and the lack of standardized evaluation methods, further complicated by the undisclosed nature of many LLM training sets. To address these limitations, we propose PerProb, a unified, label-free framework for indirectly assessing LLM memorization vulnerabilities. PerProb evaluates changes in perplexity and average log probability between data generated by victim and adversary models, enabling an indirect estimation of training-induced memory. Compared with prior MIA methods that rely on member/non-member labels or internal access, PerProb is independent of model and task, and applicable in both black-box and white-box settings. Through a systematic classification of MIA into four attack patterns, we evaluate PerProb's effectiveness across five datasets, revealing varying memory behaviors and privacy risks among LLMs. Additionally, we assess mitigation strategies, including knowledge distillation, early stopping, and differential privacy, demonstrating their effectiveness in reducing data leakage. Our findings offer a practical and generalizable framework for evaluating and improving LLM privacy.

摘要: 大型语言模型（LLM）的快速发展是由可能包含敏感信息的大量数据集推动的，从而引发了严重的隐私问题。一个值得注意的威胁是会员推断攻击（MIA），对手推断特定样本是否用于模型训练。然而，由于调查结果不一致和缺乏标准化的评估方法，MIA对LLM的真正影响仍然不清楚，而且许多LLM培训集的未公开性质使其更加复杂。为了解决这些限制，我们提出了PerProb，这是一个统一的、无标签的框架，用于间接评估LLM记忆漏洞。PerProb评估受害者和对手模型生成的数据之间的困惑度和平均日志概率的变化，从而能够间接估计训练诱导的记忆。与之前依赖成员/非成员标签或内部访问的MIA方法相比，PerProb独立于模型和任务，适用于黑盒和白盒设置。通过将MIA系统地分类为四种攻击模式，我们评估了PerProb在五个数据集中的有效性，揭示了LLM之间不同的记忆行为和隐私风险。此外，我们还评估了缓解策略，包括知识提炼、提前停止和差异隐私，证明它们在减少数据泄露方面的有效性。我们的研究结果为评估和改善LLM隐私提供了一个实用和可推广的框架。



## **19. Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space**

通过隐形方式转移对LLM代理的推理方式中毒：RSV空间中的过程级攻击和收件箱监控 cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14448v1) [paper-pdf](https://arxiv.org/pdf/2512.14448v1)

**Authors**: Xingfu Zhou, Pengfei Wang

**Abstract**: Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis.

摘要: 依赖外部检索的大型语言模型（LLM）代理越来越多地部署在高风险环境中。虽然现有的对抗性攻击主要集中在内容伪造或指令注入上，但我们发现了一种新颖的、面向过程的攻击表面：代理的推理风格。我们提出了推理式中毒（RSP），这是一种操纵代理如何处理信息而不是处理内容的范式。我们引入了生成风格注入（GSI），这是一种攻击方法，将检索到的文档改写为病理性语气--特别是“分析瘫痪”或“认知仓促”--而无需改变基本事实或使用明确的触发器。为了量化这些转变，我们开发了推理风格载体（RSV），这是一种跟踪验证深度、自信和注意力焦点的指标。使用ReAct、ReReReflection和Tree of Thoughts（ToT）架构对HotpotQA和FEVER进行的实验表明，GSI会显着降低性能。它将推理步骤增加多达4.4倍，否则会导致过早错误，从而成功绕过最先进的内容过滤器。最后，我们提出了RSP-M，这是一种轻量级的运行时监视器，可实时计算RSV指标，并在值超过安全阈值时触发警报。我们的工作表明，推理风格是一个独特的、可利用的漏洞，需要静态内容分析之外的流程级防御。



## **20. Mimicking Human Visual Development for Learning Robust Image Representations**

模仿人类视觉发展以学习稳健的图像表示 cs.CV

Accepted to ICVGIP 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14360v1) [paper-pdf](https://arxiv.org/pdf/2512.14360v1)

**Authors**: Ankita Raj, Kaashika Prajaapat, Tapan Kumar Gandhi, Chetan Arora

**Abstract**: The human visual system is remarkably adept at adapting to changes in the input distribution; a capability modern convolutional neural networks (CNNs) still struggle to match. Drawing inspiration from the developmental trajectory of human vision, we propose a progressive blurring curriculum to improve the generalization and robustness of CNNs. Human infants are born with poor visual acuity, gradually refining their ability to perceive fine details. Mimicking this process, we begin training CNNs on highly blurred images during the initial epochs and progressively reduce the blur as training advances. This approach encourages the network to prioritize global structures over high-frequency artifacts, improving robustness against distribution shifts and noisy inputs. Challenging prior claims that blurring in the initial training epochs imposes a stimulus deficit and irreversibly harms model performance, we reveal that early-stage blurring enhances generalization with minimal impact on in-domain accuracy. Our experiments demonstrate that the proposed curriculum reduces mean corruption error (mCE) by up to 8.30% on CIFAR-10-C and 4.43% on ImageNet-100-C datasets, compared to standard training without blurring. Unlike static blur-based augmentation, which applies blurred images randomly throughout training, our method follows a structured progression, yielding consistent gains across various datasets. Furthermore, our approach complements other augmentation techniques, such as CutMix and MixUp, and enhances both natural and adversarial robustness against common attack methods. Code is available at https://github.com/rajankita/Visual_Acuity_Curriculum.

摘要: 人类视觉系统非常善于适应输入分布的变化;现代卷积神经网络（CNN）仍然难以匹敌的能力。我们从人类视觉的发展轨迹中汲取灵感，提出了一种渐进式模糊课程，以提高CNN的概括性和稳健性。人类婴儿生来视力就差，逐渐提高他们感知细微细节的能力。模仿这个过程，我们开始在初始时期对高度模糊的图像进行CNN训练，并随着训练的进展逐渐减少模糊。这种方法鼓励网络优先考虑全球结构而不是高频伪影，从而提高针对分布变化和有噪输入的鲁棒性。我们反驳了之前的说法，即初始训练时期的模糊会造成刺激赤字并不可逆转地损害模型性能，我们发现早期模糊会增强概括性，对域内准确性的影响最小。我们的实验表明，与没有模糊的标准训练相比，拟议的课程在CIFAR-10-C数据集上将平均腐败错误（mCE）减少了高达8.30%，在ImageNet-100-C数据集上将平均腐败错误（mCE）减少了4.43%。与基于模糊的静态增强（在整个训练过程中随机应用模糊图像）不同，我们的方法遵循结构化进程，在各种数据集中产生一致的收益。此外，我们的方法还补充了CutMix和MixUp等其他增强技术，并增强了针对常见攻击方法的自然和对抗鲁棒性。代码可在https://github.com/rajankita/Visual_Acuity_Curriculum上获取。



## **21. Optimizing the Adversarial Perturbation with a Momentum-based Adaptive Matrix**

利用基于动量的自适应矩阵优化对抗扰动 cs.LG

IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14188v1) [paper-pdf](https://arxiv.org/pdf/2512.14188v1)

**Authors**: Wei Tao, Sheng Long, Xin Liu, Wei Li, Qing Tao

**Abstract**: Generating adversarial examples (AEs) can be formulated as an optimization problem. Among various optimization-based attacks, the gradient-based PGD and the momentum-based MI-FGSM have garnered considerable interest. However, all these attacks use the sign function to scale their perturbations, which raises several theoretical concerns from the point of view of optimization. In this paper, we first reveal that PGD is actually a specific reformulation of the projected gradient method using only the current gradient to determine its step-size. Further, we show that when we utilize a conventional adaptive matrix with the accumulated gradients to scale the perturbation, PGD becomes AdaGrad. Motivated by this analysis, we present a novel momentum-based attack AdaMI, in which the perturbation is optimized with an interesting momentum-based adaptive matrix. AdaMI is proved to attain optimal convergence for convex problems, indicating that it addresses the non-convergence issue of MI-FGSM, thereby ensuring stability of the optimization process. The experiments demonstrate that the proposed momentum-based adaptive matrix can serve as a general and effective technique to boost adversarial transferability over the state-of-the-art methods across different networks while maintaining better stability and imperceptibility.

摘要: 生成对抗性示例（AE）可以被公式化为一个优化问题。在各种基于优化的攻击中，基于梯度的PGD和基于动量的MI-FGSM已经引起了相当大的兴趣。然而，所有这些攻击都使用符号函数来缩放其扰动，这从优化的角度提出了一些理论问题。在本文中，我们首先揭示，PGD实际上是一个特定的投影梯度方法的改造，只使用当前梯度来确定其步长。此外，我们表明，当我们利用具有累积梯度的传统自适应矩阵来缩放扰动时，PVD就会变成AdaGrad。受此分析的启发，我们提出了一种新型的基于动量的攻击AdaMI，其中使用一个有趣的基于动量的自适应矩阵对扰动进行了优化。AdaMI被证明能够实现凸问题的最优收敛，表明它解决了MI-FGSM的非收敛问题，从而确保了优化过程的稳定性。实验表明，所提出的基于动量的自适应矩阵可以作为一种通用且有效的技术，与最先进的方法相比，可以提高不同网络之间的对抗可转移性，同时保持更好的稳定性和不可感知性。



## **22. On Improving Deep Active Learning with Formal Verification**

利用形式验证改进深度主动学习 cs.LG

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14170v1) [paper-pdf](https://arxiv.org/pdf/2512.14170v1)

**Authors**: Jonathan Spiegelman, Guy Amir, Guy Katz

**Abstract**: Deep Active Learning (DAL) aims to reduce labeling costs in neural-network training by prioritizing the most informative unlabeled samples for annotation. Beyond selecting which samples to label, several DAL approaches further enhance data efficiency by augmenting the training set with synthetic inputs that do not require additional manual labeling. In this work, we investigate how augmenting the training data with adversarial inputs that violate robustness constraints can improve DAL performance. We show that adversarial examples generated via formal verification contribute substantially more than those produced by standard, gradient-based attacks. We apply this extension to multiple modern DAL techniques, as well as to a new technique that we propose, and show that it yields significant improvements in model generalization across standard benchmarks.

摘要: 深度主动学习（DAL）旨在通过优先考虑信息最丰富的未标记样本进行注释来降低神经网络训练中的标记成本。除了选择要标记的样本之外，几种DAL方法还通过使用不需要额外手动标记的合成输入来增强训练集，进一步提高了数据效率。在这项工作中，我们研究了如何用违反稳健性约束的对抗输入来增强训练数据可以提高DAL性能。我们表明，通过形式验证生成的对抗性示例的贡献远高于通过标准的基于梯度的攻击生成的对抗性示例。我们将此扩展应用于多种现代DAL技术以及我们提出的一种新技术，并表明它在标准基准测试中的模型概括方面产生了显着改进。



## **23. MURIM: Multidimensional Reputation-based Incentive Mechanism for Federated Learning**

MURIM：基于声誉的多维联邦学习激励机制 cs.AI

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13955v1) [paper-pdf](https://arxiv.org/pdf/2512.13955v1)

**Authors**: Sindhuja Madabushi, Dawood Wasif, Jin-Hee Cho

**Abstract**: Federated Learning (FL) has emerged as a leading privacy-preserving machine learning paradigm, enabling participants to share model updates instead of raw data. However, FL continues to face key challenges, including weak client incentives, privacy risks, and resource constraints. Assessing client reliability is essential for fair incentive allocation and ensuring that each client's data contributes meaningfully to the global model. To this end, we propose MURIM, a MUlti-dimensional Reputation-based Incentive Mechanism that jointly considers client reliability, privacy, resource capacity, and fairness while preventing malicious or unreliable clients from earning undeserved rewards. MURIM allocates incentives based on client contribution, latency, and reputation, supported by a reliability verification module. Extensive experiments on MNIST, FMNIST, and ADULT Income datasets demonstrate that MURIM achieves up to 18% improvement in fairness metrics, reduces privacy attack success rates by 5-9%, and improves robustness against poisoning and noisy-gradient attacks by up to 85% compared to state-of-the-art baselines. Overall, MURIM effectively mitigates adversarial threats, promotes fair and truthful participation, and preserves stable model convergence across heterogeneous and dynamic federated settings.

摘要: 联合学习（FL）已成为一种领先的隐私保护机器学习范式，使参与者能够共享模型更新而不是原始数据。然而，FL继续面临关键挑战，包括客户激励薄弱、隐私风险和资源限制。评估客户可靠性对于公平的激励分配和确保每个客户的数据对全球模型做出有意义的贡献至关重要。为此，我们提出了MURIM，这是一种多维基于声誉的激励机制，它联合考虑客户可靠性、隐私、资源容量和公平性，同时防止恶意或不可靠的客户获得不应有的回报。MURIM根据客户贡献、延迟和声誉分配激励，并由可靠性验证模块支持。对MNIST、FMNIST和ADSYS Income数据集的广泛实验表明，与最先进的基线相比，MURIM在公平性指标方面实现了高达18%的提高，将隐私攻击成功率降低5- 9%，并将针对中毒和噪音梯度攻击的鲁棒性提高了高达85%。总体而言，MURIM有效地缓解了对抗威胁，促进了公平和真实的参与，并在异类和动态联邦环境中保持稳定的模型融合。



## **24. Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures (XAMT)**

异类多代理体系结构（XAMT）中隐蔽内存篡改的双层优化 cs.CR

10 pages, 5 figures, 4 tables. Conference-style paper (IEEEtran). Proposes unified bilevel optimization framework for covert memory poisoning attacks in heterogeneous multi-agent systems (MARL + RAG)

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.15790v1) [paper-pdf](https://arxiv.org/pdf/2512.15790v1)

**Authors**: Akhil Sharma, Shaikh Yaser Arafat, Jai Kumar Sharma, Ken Huang

**Abstract**: The increasing operational reliance on complex Multi-Agent Systems (MAS) across safety-critical domains necessitates rigorous adversarial robustness assessment. Modern MAS are inherently heterogeneous, integrating conventional Multi-Agent Reinforcement Learning (MARL) with emerging Large Language Model (LLM) agent architectures utilizing Retrieval-Augmented Generation (RAG). A critical shared vulnerability is reliance on centralized memory components: the shared Experience Replay (ER) buffer in MARL and the external Knowledge Base (K) in RAG agents. This paper proposes XAMT (Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures), a novel framework that formalizes attack generation as a bilevel optimization problem. The Upper Level minimizes perturbation magnitude (delta) to enforce covertness while maximizing system behavior divergence toward an adversary-defined target (Lower Level). We provide rigorous mathematical instantiations for CTDE MARL algorithms and RAG-based LLM agents, demonstrating that bilevel optimization uniquely crafts stealthy, minimal-perturbation poisons evading detection heuristics. Comprehensive experimental protocols utilize SMAC and SafeRAG benchmarks to quantify effectiveness at sub-percent poison rates (less than or equal to 1 percent in MARL, less than or equal to 0.1 percent in RAG). XAMT defines a new unified class of training-time threats essential for developing intrinsically secure MAS, with implications for trust, formal verification, and defensive strategies prioritizing intrinsic safety over perimeter-based detection.

摘要: 安全关键领域对复杂多智能体系统（MAS）的运营依赖日益增加，需要进行严格的对抗稳健性评估。现代MAS本质上是异类的，将传统的多智能体强化学习（MARL）与利用检索增强生成（RAG）的新兴大型语言模型（LLM）智能体架构集成在一起。一个关键的共享漏洞是对集中式内存组件的依赖：MARL中的共享体验重播（ER）缓冲区和RAG代理中的外部知识库（K）。本文提出了XAPT（在异类多代理体系结构中针对隐蔽内存篡改的双层优化），这是一个新颖的框架，将攻击生成形式化为双层优化问题。上级最小化扰动幅度（增量）以加强隐蔽性，同时最大化系统行为向敌对定义的目标（下级）的分歧。我们为CTDE MARL算法和基于RAG的LLM代理提供了严格的数学实例，证明两层优化独特地处理了逃避检测启发的隐形、最小扰动毒药。全面的实验方案利用SMAC和SafeRAG基准来量化次百分中毒率（MARL中小于或等于1%，RAG中小于或等于0.1%）的有效性。XAMT定义了一种新的统一训练时威胁，这对于开发本质安全的MAS至关重要，对信任、形式验证和防御策略产生了影响，这些策略优先于本质安全而不是基于边界的检测。



## **25. Evaluating Adversarial Attacks on Federated Learning for Temperature Forecasting**

评估温度预测联邦学习的对抗攻击 cs.LG

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.13207v2) [paper-pdf](https://arxiv.org/pdf/2512.13207v2)

**Authors**: Karina Chichifoi, Fabio Merizzi, Michele Colajanni

**Abstract**: Deep learning and federated learning (FL) are becoming powerful partners for next-generation weather forecasting. Deep learning enables high-resolution spatiotemporal forecasts that can surpass traditional numerical models, while FL allows institutions in different locations to collaboratively train models without sharing raw data, addressing efficiency and security concerns. While FL has shown promise across heterogeneous regions, its distributed nature introduces new vulnerabilities. In particular, data poisoning attacks, in which compromised clients inject manipulated training data, can degrade performance or introduce systematic biases. These threats are amplified by spatial dependencies in meteorological data, allowing localized perturbations to influence broader regions through global model aggregation. In this study, we investigate how adversarial clients distort federated surface temperature forecasts trained on the Copernicus European Regional ReAnalysis (CERRA) dataset. We simulate geographically distributed clients and evaluate patch-based and global biasing attacks on regional temperature forecasts. Our results show that even a small fraction of poisoned clients can mislead predictions across large, spatially connected areas. A global temperature bias attack from a single compromised client shifts predictions by up to -1.7 K, while coordinated patch attacks more than triple the mean squared error and produce persistent regional anomalies exceeding +3.5 K. Finally, we assess trimmed mean aggregation as a defense mechanism, showing that it successfully defends against global bias attacks (2-13% degradation) but fails against patch attacks (281-603% amplification), exposing limitations of outlier-based defenses for spatially correlated data.

摘要: 深度学习和联合学习（FL）正在成为下一代天气预报的强大合作伙伴。深度学习可以实现超越传统数值模型的高分辨率时空预测，而FL则允许不同地点的机构在无需共享原始数据的情况下协作训练模型，从而解决效率和安全问题。虽然FL在不同地区表现出了希望，但其分布式性质带来了新的漏洞。特别是，数据中毒攻击（即受影响的客户端注入操纵的训练数据）可能会降低性能或引入系统性偏差。气象数据的空间依赖性放大了这些威胁，使局部扰动能够通过全球模型聚合影响更广泛的区域。在这项研究中，我们调查了对抗性客户端如何扭曲在哥白尼欧洲区域再分析（CERRA）数据集上训练的联合表面温度预测。我们模拟地理上分布的客户端和评估补丁为基础的和全球偏见的攻击区域温度预报。我们的研究结果表明，即使是一小部分中毒的客户端也会误导大型空间连接区域的预测。来自单个受损客户端的全球温度偏差攻击将预测值改变高达-1.7 K，而协调补丁攻击则是均方误差的三倍多，并产生超过+3.5 K的持续区域异常。最后，我们评估了修剪平均聚集作为防御机制，表明它成功防御全局偏差攻击（2-13%降级），但未能防御补丁攻击（281-603%放大），暴露了基于离群值的防御空间相关数据的局限性。



## **26. The Eminence in Shadow: Exploiting Feature Boundary Ambiguity for Robust Backdoor Attacks**

阴影中的杰出：利用特征边界模糊性进行稳健的后门攻击 cs.LG

Accepted by KDD2026 Cycle 1 Research Track

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.10402v2) [paper-pdf](https://arxiv.org/pdf/2512.10402v2)

**Authors**: Zhou Feng, Jiahao Chen, Chunyi Zhou, Yuwen Pu, Tianyu Du, Jinbao Li, Jianhai Chen, Shouling Ji

**Abstract**: Deep neural networks (DNNs) underpin critical applications yet remain vulnerable to backdoor attacks, typically reliant on heuristic brute-force methods. Despite significant empirical advancements in backdoor research, the lack of rigorous theoretical analysis limits understanding of underlying mechanisms, constraining attack predictability and adaptability. Therefore, we provide a theoretical analysis targeting backdoor attacks, focusing on how sparse decision boundaries enable disproportionate model manipulation. Based on this finding, we derive a closed-form, ambiguous boundary region, wherein negligible relabeled samples induce substantial misclassification. Influence function analysis further quantifies significant parameter shifts caused by these margin samples, with minimal impact on clean accuracy, formally grounding why such low poison rates suffice for efficacious attacks. Leveraging these insights, we propose Eminence, an explainable and robust black-box backdoor framework with provable theoretical guarantees and inherent stealth properties. Eminence optimizes a universal, visually subtle trigger that strategically exploits vulnerable decision boundaries and effectively achieves robust misclassification with exceptionally low poison rates (< 0.1%, compared to SOTA methods typically requiring > 1%). Comprehensive experiments validate our theoretical discussions and demonstrate the effectiveness of Eminence, confirming an exponential relationship between margin poisoning and adversarial boundary manipulation. Eminence maintains > 90% attack success rate, exhibits negligible clean-accuracy loss, and demonstrates high transferability across diverse models, datasets and scenarios.

摘要: 深度神经网络（DNN）支撑关键应用程序，但仍然容易受到后门攻击，通常依赖于启发式暴力方法。尽管后门研究取得了重大的经验进展，但缺乏严格的理论分析限制了对潜在机制的理解，限制了攻击的可预测性和适应性。因此，我们提供了针对后门攻击的理论分析，重点关注稀疏决策边界如何导致不成比例的模型操纵。基于这一发现，我们得到了一个封闭形式的、模糊的边界区域，其中可忽略的重新标记样本会导致严重的误分类。影响函数分析进一步量化了这些边缘样本引起的显着参数变化，对清洁准确性的影响最小，正式证明了为什么如此低的中毒率足以进行有效的攻击。利用这些见解，我们提出了Eminence，这是一个可解释且强大的黑匣子后门框架，具有可证明的理论保证和固有的隐形属性。Eminence优化了一种通用的、视觉上微妙的触发器，该触发器战略性地利用脆弱的决策边界，并以极低的中毒率（<0.1%，而SOTA方法通常需要> 1%）有效地实现稳健的错误分类。全面的实验验证了我们的理论讨论，并证明了Eminence的有效性，证实了边缘中毒和对抗性边界操纵之间的指数关系。Eminence保持> 90%的攻击成功率，表现出可忽略不计的干净准确性损失，并表现出跨不同模型、数据集和场景的高可移植性。



## **27. CNFinBench: A Benchmark for Safety and Compliance of Large Language Models in Finance**

CNFinBench：金融领域大型语言模型安全和合规性的基准 cs.CE

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.09506v2) [paper-pdf](https://arxiv.org/pdf/2512.09506v2)

**Authors**: Jinru Ding, Chao Ding, Wenrao Pang, Boyi Xiao, Zhiqiang Liu, Pengcheng Chen, Jiayuan Chen, Tiantian Yuan, Junming Guan, Yidong Jiang, Dawei Cheng, Jie Xu

**Abstract**: Large language models (LLMs) are increasingly deployed across the financial sector for tasks like investment research and algorithmic trading. Their high-stakes nature demands rigorous evaluation of models' safety and regulatory alignment. However, there is a significant gap between evaluation capabilities and safety requirements. Current financial benchmarks mainly focus on textbook-style question answering and numerical problem-solving, failing to simulate the open-ended scenarios where safety risks typically manifest. To close these gaps, we introduce CNFinBench, a benchmark structured around a Capability-Compliance-Safety triad encompassing 15 subtasks. For Capability Q&As, we introduce a novel business-vertical taxonomy aligned with core financial domains like banking operations, which allows institutions to assess model readiness for deployment in operational scenarios. For Compliance and Risk Control Q&As, we embed regulatory requirements within realistic business scenarios to ensure models are evaluated under practical, scenario-driven conditions. For Safety Q&As, we uniquely incorporate structured bias and fairness auditing, a dimension overlooked by other holistic financial benchmarks, and introduce the first multi-turn adversarial dialogue task to systematically expose compliance decay under sustained, context-aware attacks. Accordingly, we propose the Harmful Instruction Compliance Score (HICS) to quantify models' consistency in resisting harmful instructions across multi-turn dialogues. Experiments on 21 models across all subtasks reveal a persistent gap between capability and compliance: models achieve an average score of 61.0 on capability tasks but drop to 34.2 on compliance and risk-control evaluations. In multi-turn adversarial dialogue tests, most LLMs attain only partial resistance, demonstrating that refusal alone is insufficient without cited, verifiable reasoning.

摘要: 大型语言模型（LLM）越来越多地被部署在金融领域，用于投资研究和算法交易等任务。它们的高风险性质要求对模型的安全性和监管一致性进行严格评估。然而，评估能力与安全要求之间存在明显差距。当前的财务基准主要集中在教科书式的问答和数字问题解决上，未能模拟安全风险通常显现的开放式场景。为了缩小这些差距，我们引入了CNFinBench，这是一个围绕能力-合规-安全三位一体构建的基准，包含15个子任务。对于能力问答，我们引入了一种与银行运营等核心金融领域保持一致的新型业务垂直分类法，使机构能够评估模型在运营场景中部署的准备情况。对于合规和风险控制问答，我们将监管要求嵌入现实的业务场景中，以确保模型在实际的、业务驱动的条件下进行评估。对于安全问答，我们独特地结合了结构性偏见和公平性审计（这是其他整体财务基准所忽视的一个维度），并引入了第一个多回合对抗性对话任务，以系统性地揭露持续、上下文感知攻击下的合规性衰退。因此，我们提出了有害指令合规评分（HICS）来量化模型在多轮对话中抵抗有害指令的一致性。对所有子任务的21个模型进行的实验揭示了能力与合规性之间持续存在的差距：模型在能力任务上的平均得分为61.0，但在合规和风险控制评估上的平均得分下降至34.2。在多轮对抗性对话测试中，大多数LLM仅获得部分抵抗，这表明如果没有引用的、可验证的推理，仅靠拒绝是不够的。



## **28. Developing Distance-Aware, and Evident Uncertainty Quantification in Dynamic Physics-Constrained Neural Networks for Robust Bearing Degradation Estimation**

在动态物理约束神经网络中开发距离感知和明显不确定性量化，用于鲁棒轴承退化估计 cs.LG

Under review at Structural health Monitoring - SAGE

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.08499v2) [paper-pdf](https://arxiv.org/pdf/2512.08499v2)

**Authors**: Waleed Razzaq, Yun-Bo Zhao

**Abstract**: Accurate and uncertainty-aware degradation estimation is essential for predictive maintenance in safety-critical systems like rotating machinery with rolling-element bearings. Many existing uncertainty methods lack confidence calibration, are costly to run, are not distance-aware, and fail to generalize under out-of-distribution data. We introduce two distance-aware uncertainty methods for deterministic physics-guided neural networks: PG-SNGP, based on Spectral Normalization Gaussian Process, and PG-SNER, based on Deep Evidential Regression. We apply spectral normalization to the hidden layers so the network preserves distances from input to latent space. PG-SNGP replaces the final dense layer with a Gaussian Process layer for distance-sensitive uncertainty, while PG-SNER outputs Normal Inverse Gamma parameters to model uncertainty in a coherent probabilistic form. We assess performance using standard accuracy metrics and a new distance-aware metric based on the Pearson Correlation Coefficient, which measures how well predicted uncertainty tracks the distance between test and training samples. We also design a dynamic weighting scheme in the loss to balance data fidelity and physical consistency. We test our methods on rolling-element bearing degradation using the PRONOSTIA, XJTU-SY and HUST datasets and compare them with Monte Carlo and Deep Ensemble PGNNs. Results show that PG-SNGP and PG-SNER improve prediction accuracy, generalize reliably under OOD conditions, and remain robust to adversarial attacks and noise.

摘要: 准确且具有不确定性的退化估计对于具有滚动元件轴承的旋转机械等安全关键系统的预测性维护至关重要。许多现有的不确定性方法缺乏置信度校准、运行成本高、不具有距离意识，并且无法在非分布数据下进行概括。我们为确定性物理引导神经网络引入了两种距离感知不确定性方法：基于谱正规化高斯过程的PG-SNGP和基于深度证据回归的PG-SNER。我们对隐藏层应用光谱正规化，以便网络保留从输入到潜在空间的距离。PG-SNGP用高斯过程层取代最终的密集层，以实现距离敏感的不确定性，而PG-SNER输出正态逆伽玛参数，以连贯的概率形式对不确定性进行建模。我们使用标准准确性指标和基于皮尔逊相关系数的新距离感知指标来评估性能，皮尔逊相关系数衡量预测的不确定性跟踪测试和训练样本之间距离的程度。我们还设计了一个动态加权方案，以平衡数据保真度和物理一致性的损失。我们使用PRONOSTIA、XJTU-SY和HUST数据集测试我们关于滚动元件轴承退化的方法，并将其与Monte Carlo和Deep Ensemble PGNN进行比较。结果表明，PG-SNGP和PG-SNER提高了预测精度，在OOD条件下可靠地推广，并对对抗性攻击和噪声保持鲁棒性。



## **29. SEA: Spectral Edge Attack**

SEA：光谱边缘攻击 cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.08964v2) [paper-pdf](https://arxiv.org/pdf/2512.08964v2)

**Authors**: Yongyu Wang

**Abstract**: Graph based machine learning algorithms occupy an important position in today AI landscape. The ability of graph topology to represent complex data structures is both the key strength of graph algorithms and a source of their vulnerability. In other words, attacking or perturbing a graph can severely degrade the performance of graph-based methods. For the attack methods, the greatest challenge is achieving strong attack effectiveness while remaining undetected. To address this problem, this paper proposes a new attack model that employs spectral adversarial robustness evaluation to quantitatively analyze the vulnerability of each edge in a graph under attack. By precisely targeting the weakest links, the proposed approach achieves the maximum attack impact with minimal perturbation. Experimental results demonstrate the effectiveness of the proposed method.

摘要: 基于图的机器学习算法在当今人工智能领域占据重要地位。图布局表示复杂数据结构的能力既是图算法的关键优势，也是其脆弱性的根源。换句话说，攻击或干扰图可能会严重降低基于图的方法的性能。对于攻击方法来说，最大的挑战是在不被发现的情况下实现强大的攻击有效性。为了解决这个问题，本文提出了一种新的攻击模型，该模型采用谱对抗鲁棒性评估来定量分析受攻击图中每条边的脆弱性。通过精确地针对最弱的环节，提出的方法以最小的干扰实现了最大的攻击影响。实验结果证明了该方法的有效性。



## **30. Tuning for Two Adversaries: Enhancing the Robustness Against Transfer and Query-Based Attacks using Hyperparameter Tuning**

针对两个对手进行调优：使用超参数调优增强针对传输和基于查询的攻击的鲁棒性 cs.LG

To appear in the Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) 2026

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2511.13654v2) [paper-pdf](https://arxiv.org/pdf/2511.13654v2)

**Authors**: Pascal Zimmer, Ghassan Karame

**Abstract**: In this paper, we present the first detailed analysis of how training hyperparameters -- such as learning rate, weight decay, momentum, and batch size -- influence robustness against both transfer-based and query-based attacks. Supported by theory and experiments, our study spans a variety of practical deployment settings, including centralized training, ensemble learning, and distributed training. We uncover a striking dichotomy: for transfer-based attacks, decreasing the learning rate significantly enhances robustness by up to $64\%$. In contrast, for query-based attacks, increasing the learning rate consistently leads to improved robustness by up to $28\%$ across various settings and data distributions. Leveraging these findings, we explore -- for the first time -- the training hyperparameter space to jointly enhance robustness against both transfer-based and query-based attacks. Our results reveal that distributed models benefit the most from hyperparameter tuning, achieving a remarkable tradeoff by simultaneously mitigating both attack types more effectively than other training setups.

摘要: 在本文中，我们首次详细分析了训练超参数（如学习率、权重衰减、动量和批量大小）如何影响对基于传输和基于查询的攻击的鲁棒性。在理论和实验的支持下，我们的研究涵盖了各种实际部署环境，包括集中式培训、集成学习和分布式培训。我们发现了一个引人注目的二分法：对于基于传输的攻击，降低学习率可以显着增强鲁棒性，提高高达64美元。相比之下，对于基于查询的攻击，在各种设置和数据分布中持续提高学习率可使稳健性提高高达28美元。利用这些发现，我们首次探索了训练超参数空间，以共同增强针对基于传输和基于查询的攻击的鲁棒性。我们的结果表明，分布式模型从超参数调整中受益最多，通过比其他训练设置更有效地同时减轻两种攻击类型，实现了显着的权衡。



## **31. Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks**

幻影威胁：探索和增强VLA模型对抗物理传感器攻击的鲁棒性 cs.RO

Accepted by AAAI 2026 main track

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2511.10008v2) [paper-pdf](https://arxiv.org/pdf/2511.10008v2)

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored. To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel "Real-Sim-Real" framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.

摘要: 视觉-语言-动作（VLA）模型通过实现端到端的感知到动作管道，彻底改变了机器人系统，该管道集成了多种感官模式，例如由摄像机处理的视觉信号和由麦克风捕获的听觉信号。这种多模式集成使VLA模型能够使用不同的传感器数据流来解释复杂的现实世界环境。鉴于基于VLA的系统严重依赖感官输入，VLA模型对抗物理世界传感器攻击的安全性仍然严重不足。为了弥补这一差距，我们首次对针对VLA的物理传感器攻击进行了系统研究，量化了传感器攻击的影响并调查VLA模型的防御。我们引入了一个新颖的“Real-Sim-Real”框架，该框架自动模拟基于物理的传感器攻击载体，包括六次针对摄像头和两个针对麦克风的攻击，并在真实的机器人系统上对其进行验证。通过在不同攻击参数下对各种VLA架构和任务进行大规模评估，我们展示了显着的漏洞，其易感性模式揭示了对任务类型和模型设计的关键依赖性。我们进一步开发了一种基于对抗训练的防御，可以增强VLA对传感器攻击引起的分布外物理扰动的鲁棒性，同时保持模型性能。我们的研究结果揭示了迫切需要标准化的稳健性基准和缓解策略，以确保VLA在安全关键环境中的部署。



## **32. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

对生成性基因组模型的生物知情混合成员推断攻击 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2511.07503v3) [paper-pdf](https://arxiv.org/pdf/2511.07503v3)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.

摘要: 遗传数据可用性的增加改变了基因组学研究，但由于其敏感性，对其处理提出了许多隐私问题。这项工作探索了使用语言模型（LM）来生成合成基因突变谱，利用差异隐私（DP）来保护敏感遗传数据。我们通过引入一种新型的生物知情混合成员推断攻击（biHMIA）来经验性地评估DP模式的隐私保证，该攻击将传统的黑匣子MIA与上下文基因组学指标相结合，以增强攻击能力。我们的实验表明，小型和大型Transformer GPT类模型都是小规模基因组学的可行合成变体生成器，并且与传统的基于度量的MIA相比，我们的混合攻击平均会导致更高的对抗成功。



## **33. Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems**

频谱掩蔽和内插攻击（SMIA）：针对语音认证和反欺骗系统的黑匣子对抗攻击 cs.SD

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2509.07677v3) [paper-pdf](https://arxiv.org/pdf/2509.07677v3)

**Authors**: Kamel Kamel, Hridoy Sankar Dutta, Keshav Sood, Sunil Aryal

**Abstract**: Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape.

摘要: 语音认证系统（PAS）使用独特的声音特征进行验证。他们越来越多地融入银行和医疗保健等高安全领域。尽管它们使用深度学习进行了改进，但它们仍面临来自深度造假和对抗攻击等复杂威胁的严重漏洞。现实语音克隆的出现使检测变得复杂，因为系统很难区分真实音频和合成音频。虽然存在反欺骗对策（CM）来减轻这些风险，但许多对策依赖于静态检测模型，这些模型可以被新型对抗方法绕过，从而留下了关键的安全漏洞。为了证明这一漏洞，我们提出了频谱掩蔽和内插攻击（SMIA），这是一种新颖的方法，可以战略性地操纵人工智能生成的音频的听不见的频率区域。通过改变人耳不可感知区域的声音，SMIA创建听起来真实的对抗样本，同时欺骗CM。我们在模拟现实世界条件下对多个任务中针对最先进（SOTA）模型的攻击进行了全面评估。SMIA在对抗组合式增值服务器/CM系统时，至少达到82%的攻击成功率（ASB），对抗独立说话者验证系统至少达到97.5%，对抗措施至少达到100%。这些发现最终证明，当前的安全姿态不足以抵御适应性对抗攻击。这项工作凸显了向下一代防御范式转变的迫切需要，这些防御采用能够随着威胁格局而演变的动态、上下文感知框架。



## **34. Larger Scale Offers Better Security in the Nakamoto-style Blockchain**

更大的规模在Nakamoto式区块链中提供更好的安全性 cs.CR

22 pages, 4 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2509.05708v3) [paper-pdf](https://arxiv.org/pdf/2509.05708v3)

**Authors**: Junjie Hu

**Abstract**: Traditional security models for Nakamoto-style blockchains assume instantaneous synchronization among malicious nodes, which overestimate adversarial coordination capability. We revisit these existing models and propose two more realistic security models. First, we propose the static delay model. This model first incorporates adversarial communication delay. It quantifies how the delay constrains the effective growth rate of private chains and yields a closed-form expression for the security threshold. Second, we propose the dynamic delay model that further captures the decay of adversarial corruption capability and the total adversarial delay window. Theoretical analysis shows that private attacks remain optimal under both models. Finally, we prove that large-scale Nakamoto-style blockchains offer better security. This result provided a theoretical foundation for optimizing consensus protocols and assessing the robustness of large-scale blockchains.

摘要: 中本式区块链的传统安全模型假设恶意节点之间的即时同步，这高估了对抗协调能力。我们重新审视这些现有模型并提出两个更现实的安全模型。首先，我们提出了静态延迟模型。该模型首先考虑了对抗性通信延迟。它量化了延迟如何限制私有链的有效增长率，并给出了安全阈值的封闭表达式。其次，我们提出了动态延迟模型，进一步捕捉对抗腐败能力和总对抗延迟窗口的衰减。理论分析表明，在这两种模型下，私有攻击仍然是最优的。最后，我们证明了大规模的Nakamoto式区块链提供了更好的安全性。这一结果为优化共识协议和评估大规模区块链的稳健性提供了理论基础。



## **35. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

相信我，我知道这个功能：使用偏差劫持LLM静态分析 cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2508.17361v2) [paper-pdf](https://arxiv.org/pdf/2508.17361v2)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective against basic and reasoning models, but are also transferable across model families (OpenAI, Anthropic, Google), and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.

摘要: 大型语言模型（LLM）越来越被信任大规模执行自动代码审查和静态分析，支持漏洞检测、总结和重构等任务。在本文中，我们识别并利用基于LLM的代码分析中的一个关键漏洞：一种抽象偏见，导致模型过度概括熟悉的编程模式并忽略小而有意义的错误。对手可以利用这个盲点来劫持LLM解释的控制流，只需最少的编辑，并且不会影响实际的运行时行为。我们将这种攻击称为熟悉模式攻击（FTA）。   我们开发了一种全自动的黑匣子算法，可以发现并将FPA注入目标代码。我们的评估表明，PFA不仅对基本模型和推理模型有效，而且还可以跨模型家族（OpenAI、Anthropic、Google）移植，并且跨编程语言（Python、C、Rust、Go）通用。此外，即使通过强大的系统提示明确警告模型有关攻击，FPA仍然有效。最后，我们探讨了PFA的积极、防御性用途，并讨论了它们对面向代码的LLM的可靠性和安全性的更广泛影响。



## **36. Exact Verification of Graph Neural Networks with Incremental Constraint Solving**

增量约束求解的图神经网络精确验证 cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2508.09320v2) [paper-pdf](https://arxiv.org/pdf/2508.09320v2)

**Authors**: Minghao Liu, Chia-Hsuan Lu, Marta Kwiatkowska

**Abstract**: Graph neural networks (GNNs) are increasingly employed in high-stakes applications, such as fraud detection or healthcare, but are susceptible to adversarial attacks. A number of techniques have been proposed to provide adversarial robustness guarantees, but support for commonly used aggregation functions in message-passing GNNs is lacking. In this paper, we develop an exact (sound and complete) verification method for GNNs to compute guarantees against attribute and structural perturbations that involve edge addition or deletion, subject to budget constraints. Our method employs constraint solving with bound tightening, and iteratively solves a sequence of relaxed constraint satisfaction problems while relying on incremental solving capabilities of solvers to improve efficiency. We implement GNNev, a versatile exact verifier for message-passing neural networks, which supports three aggregation functions, sum, max and mean, with the latter two considered here for the first time. Extensive experimental evaluation of GNNev on real-world fraud datasets (Amazon and Yelp) and biochemical datasets (MUTAG and ENZYMES) demonstrates its usability and effectiveness, as well as superior performance for node classification and competitiveness on graph classification compared to existing exact verification tools on sum-aggregated GNNs.

摘要: 图神经网络（GNN）越来越多地用于欺诈检测或医疗保健等高风险应用，但很容易受到对抗攻击。人们提出了多种技术来提供对抗稳健性保证，但缺乏对消息传递GNN中常用的聚合函数的支持。在本文中，我们为GNN开发了一种精确的（合理且完整的）验证方法，以计算针对涉及边添加或删除的属性和结构扰动的保证，并受预算限制。我们的方法采用带边界收紧的约束求解，并迭代地解决一系列宽松的约束满足问题，同时依靠求解器的增量求解能力来提高效率。我们实现了GNNev，这是一个用于消息传递神经网络的通用精确验证器，它支持三个聚合函数：sum、max和mean，其中后两个是本文首次考虑。GNNev在现实世界欺诈数据集（Amazon和Yelp）和生化数据集（MUTAG和ENZYMES）上进行的广泛实验评估证明了其可用性和有效性，以及与现有的精确验证工具相比，在节点分类和图形分类上具有卓越的性能。聚合GNN。



## **37. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

论LLM在对抗性攻击中言语信心的稳健性 cs.CL

Published in NeurIPS 2025

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2507.06489v3) [paper-pdf](https://arxiv.org/pdf/2507.06489v3)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.

摘要: 大型语言模型（LLM）产生的强大言语信心对于LLM的部署至关重要，以帮助确保许多应用程序（包括涉及人机交互的应用程序）的透明度、信任和安全性。在本文中，我们首次对对抗攻击下言语信心的稳健性进行了全面研究。我们通过干扰和基于越狱的方法引入了针对言语信心分数的攻击框架，并证明这些攻击会显着损害言语信心估计并导致答案频繁变化。我们检查了各种提示策略、模型大小和应用领域，揭示了当前的言语自信很脆弱，并且常用的防御技术在很大程度上无效或适得其反。我们的研究结果强调了为LLM中的信心表达设计稳健的机制的必要性，因为即使是微妙的语义保留修改也可能导致反应中的误导性信心。



## **38. Breaking the Bulkhead: Demystifying Cross-Namespace Reference Vulnerabilities in Kubernetes Operators**

打破障碍：揭开Kubernetes运营商中的跨空间引用漏洞的神秘面纱 cs.CR

18 pages. Accepted by Network and Distributed System Security (NDSS) Symposium 2026. Some information has been omitted from this preprint version due to ethical considerations. The final published version differs from this version

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2507.03387v3) [paper-pdf](https://arxiv.org/pdf/2507.03387v3)

**Authors**: Andong Chen, Ziyi Guo, Zhaoxuan Jin, Zhenyuan Li, Yan Chen

**Abstract**: Kubernetes Operators, automated tools designed to manage application lifecycles within Kubernetes clusters, extend the functionalities of Kubernetes, and reduce the operational burden on human engineers. While Operators significantly simplify DevOps workflows, they introduce new security risks. In particular, Kubernetes enforces namespace isolation to separate workloads and limit user access, ensuring that users can only interact with resources within their authorized namespaces. However, Kubernetes Operators often demand elevated privileges and may interact with resources across multiple namespaces. This introduces a new class of vulnerabilities, the Cross-Namespace Reference Vulnerability. The root cause lies in the mismatch between the declared scope of resources and the implemented scope of the Operator logic, resulting in Kubernetes being unable to properly isolate the namespace. Leveraging such vulnerability, an adversary with limited access to a single authorized namespace may exploit the Operator to perform operations affecting other unauthorized namespaces, causing Privilege Escalation and further impacts.   To the best of our knowledge, this paper is the first to systematically investigate Kubernetes Operator attacks. We present Cross-Namespace Reference Vulnerability with two strategies, demonstrating how an attacker can bypass namespace isolation. Through large-scale measurements, we found that over 14% of Operators in the wild are potentially vulnerable. Our findings have been reported to the relevant developers, resulting in 8 confirmations and 7 CVEs by the time of submission, affecting vendors including Red Hat and NVIDIA, highlighting the critical need for enhanced security practices in Kubernetes Operators. To mitigate it, we open-source the static analysis suite and propose concrete mitigation to benefit the ecosystem.

摘要: Kubernetes Operators是一种自动化工具，旨在管理Kubernetes集群内的应用程序生命周期，扩展Kubernetes的功能，并减少人类工程师的操作负担。虽然运营商大大简化了DevOps工作流程，但它们引入了新的安全风险。特别是，Kubernetes强制执行命名空间隔离，以分离工作负载并限制用户访问，确保用户只能与其授权命名空间内的资源交互。但是，Kubernetes Operator通常需要提升的权限，并且可能会跨多个命名空间与资源交互。这引入了一类新的漏洞，即跨空间引用漏洞。根本原因在于声明的资源范围与Operator逻辑的实现范围不匹配，导致Kubernetes无法正确隔离命名空间。利用此类漏洞，对单个授权命名空间的访问权限有限的对手可能会利用运营商执行影响其他未经授权命名空间的操作，从而导致特权升级和进一步影响。   据我们所知，本文是第一篇系统性研究Kubernetes Operator攻击的论文。我们通过两种策略展示跨命名空间引用漏洞，展示攻击者如何绕过命名空间隔离。通过大规模测量，我们发现超过14%的野外经营者存在潜在的脆弱性。我们的调查结果已报告给相关开发人员，截至提交时已获得8项确认和7项CVS，影响了Red Hat和NVIDIA等供应商，凸显了Kubernetes运营商对增强安全实践的迫切需求。为了缓解这种情况，我们开源了静态分析套件，并提出具体的缓解措施以造福生态系统。



## **39. Multimodal Representation Learning and Fusion**

多模式表示学习与融合 cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2506.20494v2) [paper-pdf](https://arxiv.org/pdf/2506.20494v2)

**Authors**: Qihang Jin, Enze Ge, Yuhang Xie, Hongying Luo, Junhao Song, Ziqian Bi, Chia Xin Liang, Jibin Guan, Joe Yeong, Xinyuan Song, Junfeng Hao

**Abstract**: Multi-modal learning is a fast growing area in artificial intelligence. It tries to help machines understand complex things by combining information from different sources, like images, text, and audio. By using the strengths of each modality, multi-modal learning allows AI systems to build stronger and richer internal representations. These help machines better interpretation, reasoning, and making decisions in real-life situations. This field includes core techniques such as representation learning (to get shared features from different data types), alignment methods (to match information across modalities), and fusion strategies (to combine them by deep learning models). Although there has been good progress, some major problems still remain. Like dealing with different data formats, missing or incomplete inputs, and defending against adversarial attacks. Researchers now are exploring new methods, such as unsupervised or semi-supervised learning, AutoML tools, to make models more efficient and easier to scale. And also more attention on designing better evaluation metrics or building shared benchmarks, make it easier to compare model performance across tasks and domains. As the field continues to grow, multi-modal learning is expected to improve many areas: computer vision, natural language processing, speech recognition, and healthcare. In the future, it may help to build AI systems that can understand the world in a way more like humans, flexible, context aware, and able to deal with real-world complexity.

摘要: 多模式学习是人工智能中一个快速发展的领域。它试图通过组合来自图像、文本和音频等不同来源的信息来帮助机器理解复杂的事物。通过利用每种模式的优势，多模式学习允许人工智能系统构建更强大、更丰富的内部表示。这些可以帮助机器在现实生活中更好地解释、推理和做出决策。该领域包括核心技术，例如表示学习（从不同数据类型获得共享特征）、对齐方法（匹配跨模式的信息）和融合策略（通过深度学习模型将它们组合起来）。尽管取得了良好进展，但仍存在一些重大问题。例如处理不同的数据格式、缺失或不完整的输入以及防御对抗性攻击。研究人员现在正在探索新方法，例如无监督或半监督学习、AutoML工具，以使模型更高效、更容易扩展。此外，还要更加关注设计更好的评估指标或构建共享基准，从而更容易比较跨任务和领域的模型性能。随着该领域的不断发展，多模式学习有望改善许多领域：计算机视觉、自然语言处理、语音识别和医疗保健。未来，它可能有助于构建能够以更像人类的方式理解世界、灵活、上下文感知并能够处理现实世界复杂性的人工智能系统。



## **40. Benchmarking Gaslighting Negation Attacks Against Reasoning Models**

针对推理模型的煤气灯否定攻击基准 cs.CV

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2506.09677v2) [paper-pdf](https://arxiv.org/pdf/2506.09677v2)

**Authors**: Bin Zhu, Hailong Yin, Jingjing Chen, Yu-Gang Jiang

**Abstract**: Recent advances in reasoning-centric models promise improved robustness through mechanisms such as chain-of-thought prompting and test-time scaling. However, their ability to withstand gaslighting negation attacks-adversarial prompts that confidently deny correct answers-remains underexplored. In this paper, we conduct a systematic evaluation of three state-of-the-art reasoning models, i.e., OpenAI's o4-mini, Claude-3.7-Sonnet and Gemini-2.5-Flash, across three multimodal benchmarks: MMMU, MathVista, and CharXiv. Our evaluation reveals significant accuracy drops (25-29% on average) following gaslighting negation attacks, indicating that even top-tier reasoning models struggle to preserve correct answers under manipulative user feedback. Built upon the insights of the evaluation and to further probe this vulnerability, we introduce GaslightingBench-R, a new diagnostic benchmark specifically designed to evaluate reasoning models' susceptibility to defend their belief under gaslighting negation attacks. Constructed by filtering and curating 1,025 challenging samples from the existing benchmarks, GaslightingBench-R induces even more dramatic failures, with accuracy drops exceeding 53% on average. Our findings highlight a fundamental gap between step-by-step reasoning and resistance to adversarial manipulation, calling for new robustness strategies that safeguard reasoning models against gaslighting negation attacks.

摘要: 以推理为中心的模型的最新进展承诺通过思想链提示和测试时扩展等机制来提高鲁棒性。然而，它们抵御煤气灯否定攻击（自信地否认正确答案的对抗性提示）的能力仍然没有得到充分的探索。在本文中，我们对三种最先进的推理模型进行了系统评估，即OpenAI的o 4-mini、Claude-3.7-Sonnet和Gemini-2.5-Flash，跨三种多模式基准：MMMU、MathVista和CharXiv。我们的评估显示，在煤气灯否定攻击后，准确性显着下降（平均25-29%），这表明即使是顶级推理模型也很难在操纵性用户反馈下保留正确答案。基于评估的见解并进一步探索这一漏洞，我们引入了GaslightingBench-R，这是一种新的诊断基准，专门用于评估推理模型在Gaslightning否定攻击下捍卫其信念的敏感性。GaslightingBench-R通过从现有基准中过滤和整理1，025个具有挑战性的样本而构建，导致了更严重的失败，准确性平均下降超过53%。我们的研究结果强调了逐步推理和对抗性操纵抵抗之间的根本差距，呼吁采取新的稳健性策略来保护推理模型免受煤气灯否定攻击。



## **41. MoAPT: Mixture of Adversarial Prompt Tuning for Vision-Language Models**

MoAPT：视觉语言模型的对抗性提示调优混合 cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2505.17509v2) [paper-pdf](https://arxiv.org/pdf/2505.17509v2)

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Maoxun Yuan, Jialing Tao, Jiexi Liu, Ranjie Duan, Jie Zhang, Jie Zhang, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) demonstrate excellent generalization capabilities but remain highly susceptible to adversarial examples, posing potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which ultimately results in overfitting. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts yields greater robustness improvements than simply extending the length of a single prompt. Building on this observation, we propose an adversarial tuning method named \textbf{Mixture of Adversarial Prompt Tuning (MoAPT)} to enhance the generalization against various adversarial attacks for VLMs. MoAPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the adversarial images to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific mixture text features aligning with different adversarial image features. Extensive experiments across 11 datasets under different settings show that our method can achieve better adversarial robustness than state-of-the-art approaches.

摘要: 大型预先训练的视觉语言模型（VLM）表现出出色的概括能力，但仍然极易受到对抗性示例的影响，从而构成潜在的安全风险。为了提高VLM对对抗性示例的鲁棒性，提出了对抗性提示调整方法，以在不改变模型参数的情况下将文本特征与对抗性图像特征对齐。然而，当面临各种对抗性攻击时，单个可学习文本提示的概括性不足以与所有对抗性图像特征很好地对齐，这最终会导致过度匹配。为了解决上述挑战，在本文中，我们经验发现，增加学习提示的数量比简单地延长单个提示的长度可以产生更大的鲁棒性改进。在这一观察的基础上，我们提出了一种名为\textBF{混合对抗提示调整（MoAPT）}的对抗性调整方法，以增强针对VLM的各种对抗性攻击的概括性。MoAPT旨在学习混合文本提示以获得更稳健的文本特征。为了进一步增强适应性，我们提出了一种基于对抗图像的条件权重路由器来预测多个学习提示的混合权重，这有助于获得与不同对抗图像特征对齐的样本特定混合文本特征。在不同设置下对11个数据集进行的广泛实验表明，我们的方法可以实现比最先进的方法更好的对抗鲁棒性。



## **42. Improving Graph Neural Network Training, Defense and Hypergraph Partitioning via Adversarial Robustness Evaluation**

通过对抗稳健性评估改进图神经网络训练、防御和超图划分 cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2412.14738v9) [paper-pdf](https://arxiv.org/pdf/2412.14738v9)

**Authors**: Yongyu Wang

**Abstract**: Graph Neural Networks (GNNs) are a highly effective neural network architecture for processing graph-structured data. Unlike traditional neural networks that rely solely on the features of the data as input, GNNs leverage both the graph structure, which represents the relationships between data points, and the feature matrix of the data to optimize their feature representation. This unique capability enables GNNs to achieve superior performance across various tasks. However, it also makes GNNs more susceptible to noise and adversarial attacks from both the graph structure and data features, which can significantly increase the training difficulty and degrade their performance. Similarly, a hypergraph is a highly complex structure, and partitioning a hypergraph is a challenging task. This paper leverages spectral adversarial robustness evaluation to effectively address key challenges in complex-graph algorithms. By using spectral adversarial robustness evaluation to distinguish robust nodes from non-robust ones and treating them differently, we propose a training-set construction strategy that improves the training quality of GNNs. In addition, we develop algorithms to enhance both the adversarial robustness of GNNs and the performance of hypergraph partitioning. Experimental results show that this series of methods is highly effective.

摘要: 图神经网络（GNN）是一种用于处理图结构数据的高效神经网络架构。与仅依赖数据特征作为输入的传统神经网络不同，GNN利用表示数据点之间关系的图结构和数据的特征矩阵来优化其特征表示。这种独特的功能使GNN能够在各种任务中实现卓越的性能。然而，它也使GNN更容易受到来自图结构和数据特征的噪音和对抗攻击，这可能会显着增加训练难度并降低其性能。同样，超图是一种高度复杂的结构，划分超图是一项具有挑战性的任务。本文利用谱对抗鲁棒性评估来有效解决复杂图算法中的关键挑战。通过使用谱对抗鲁棒性评估来区分鲁棒节点和非鲁棒节点并区别对待它们，我们提出了一种提高GNN训练质量的训练集构建策略。此外，我们还开发了算法来增强GNN的对抗鲁棒性和超图分区的性能。实验结果表明，该系列方法非常有效。



## **43. Scaling Laws for Black box Adversarial Attacks**

黑匣子对抗攻击的缩放定律 cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2411.16782v4) [paper-pdf](https://arxiv.org/pdf/2411.16782v4)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Jun Zhu, Yinpeng Dong

**Abstract**: Adversarial examples exhibit cross-model transferability, enabling threatening black-box attacks on commercial models. Model ensembling, which attacks multiple surrogate models, is a known strategy to improve this transferability. However, prior studies typically use small, fixed ensembles, which leaves open an intriguing question of whether scaling the number of surrogate models can further improve black-box attacks. In this work, we conduct the first large-scale empirical study of this question. We show that by resolving gradient conflict with advanced optimizers, we discover a robust and universal log-linear scaling law through both theoretical analysis and empirical evaluations: the Attack Success Rate (ASR) scales linearly with the logarithm of the ensemble size $T$. We rigorously verify this law across standard classifiers, SOTA defenses, and MLLMs, and find that scaling distills robust, semantic features of the target class. Consequently, we apply this fundamental insight to benchmark SOTA MLLMs. This reveals both the attack's devastating power and a clear robustness hierarchy: we achieve 80\%+ transfer attack success rate on proprietary models like GPT-4o, while also highlighting the exceptional resilience of Claude-3.5-Sonnet. Our findings urge a shift in focus for robustness evaluation: from designing intricate algorithms on small ensembles to understanding the principled and powerful threat of scaling.

摘要: 对抗性示例表现出跨模型的可移植性，从而能够对商业模型进行威胁性的黑匣子攻击。攻击多个代理模型的模型集成是提高这种可移植性的已知策略。然而，之前的研究通常使用小型、固定的集合，这留下了一个有趣的问题：扩大代理模型的数量是否可以进一步改善黑匣子攻击。在这项工作中，我们对这个问题进行了首次大规模的实证研究。我们表明，通过使用高级优化器解决梯度冲突，我们通过理论分析和经验评估发现了一个鲁棒且通用的日志线性缩放定律：攻击成功率（ASB）与总体大小$T$的log呈线性缩放。我们在标准分类器、SOTA防御和MLLM之间严格验证了这一定律，并发现缩放可以提炼出目标类的稳健语义特征。因此，我们将这一基本见解应用于基准SOTA MLLM。这既揭示了攻击的破坏力，又揭示了明确的鲁棒性层次结构：我们在GPT-4 o等专有模型上实现了80%以上的传输攻击成功率，同时也凸显了Claude-3.5-Sonnet的非凡弹性。我们的研究结果促使稳健性评估的重点发生转变：从在小集合上设计复杂的算法到理解扩展的原则性且强大的威胁。



## **44. Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors**

针对车辆检测器实现稳健、准确的对抗性伪装生成 cs.CV

14 pages. arXiv admin note: substantial text overlap with arXiv:2402.15853

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2411.10029v2) [paper-pdf](https://arxiv.org/pdf/2411.10029v2)

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, End-to-End Neural Renderer Plus (E2E-NRP), which can accurately optimize and project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the E2E-NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA-final outperforms existing methods in both simulation and real-world settings.

摘要: 对抗伪装因其在多视角攻击性能上的优势而被广泛使用，是针对车辆探测器的物理攻击。一种有前途的方法涉及使用可微神经渲染器通过梯度反向传播促进对抗伪装优化。然而，现有方法通常难以在渲染过程中捕捉环境特征或产生可以精确映射到目标车辆的对抗纹理。此外，这些方法忽视了不同的天气条件，降低了在不同天气场景下生成的伪装的功效。为了应对这些挑战，我们提出了一种稳健且准确的伪装生成方法，即RAUCA。RAUCA的核心是一个新型神经渲染组件端到端神经渲染Plus（E2 E-NRP），它可以准确地优化和投影车辆纹理，并渲染具有灯光和天气等环境特征的图像。此外，我们还集成了多天气数据集来生成伪装，利用E2 E-NRP来增强攻击稳健性。六种流行物体检测器的实验结果表明，RAUCA-final在模拟和现实环境中都优于现有方法。



## **45. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

人工智能生成图像检测中的漏洞：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2407.20836v5) [paper-pdf](https://arxiv.org/pdf/2407.20836v5)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g., transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we demonstrate that adversarial attacks pose a real threat to AIGI detectors. FPBA can deliver successful black-box attacks across various detectors, generators, defense methods, and even evade cross-generator and compressed image detection, which are crucial real-world detection scenarios. Our code is available at https://github.com/onotoa/fpba.

摘要: 图像合成领域的最新进展，特别是GAN和扩散模型的出现，加剧了公众对虚假信息传播的担忧。为了解决这些问题，人们提出了许多人工智能生成的图像（AIGI）检测器，并在识别虚假图像方面取得了良好的性能。然而，人们仍然缺乏对AIGI检测器对抗鲁棒性的系统了解。在本文中，我们研究了最先进的AIGI检测器在白盒和黑盒设置下对抗攻击的脆弱性，迄今为止，这很少被研究。为此，我们提出了一种攻击AIGI检测器的新方法。首先，受到频域中真实图像和假图像之间明显差异的启发，我们在频域下添加扰动，以推动图像远离其原始频率分布。其次，我们探索代理模型的完整后验分布，以进一步缩小异类AIGI检测器之间的差距，例如，跨CNN和ViT传输对抗性示例。这是通过引入一种新型的训练后Bayesian策略来实现的，该策略将单个代理变成了Bayesian代理，能够使用一个预先训练的代理来模拟不同的受害者模型，而无需重新训练。我们将我们的方法命名为基于频率的训练后Bayesian Attack（FPBA）。通过FPBA，我们证明了对抗性攻击对AIGI检测器构成了真正的威胁。FPBA可以跨各种检测器、生成器、防御方法提供成功的黑匣子攻击，甚至规避交叉生成器和压缩图像检测，这些都是现实世界中至关重要的检测场景。我们的代码可在https://github.com/onotoa/fpba上获取。



## **46. Over-parameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis**

神经网络中的过度参数化和对抗鲁棒性：概述和实证分析 cs.LG

Submitted to Discover AI

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2406.10090v2) [paper-pdf](https://arxiv.org/pdf/2406.10090v2)

**Authors**: Srishti Gupta, Zhang Chen, Luca Demetrio, Xiaoyi Feng, Zhaoqiang Xia, Antonio Emanuele Cinà, Maura Pintor, Luca Oneto, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Thanks to their extensive capacity, over-parameterized neural networks exhibit superior predictive capabilities and generalization. However, having a large parameter space is considered one of the main suspects of the neural networks' vulnerability to adversarial example -- input samples crafted ad-hoc to induce a desired misclassification. Relevant literature has claimed contradictory remarks in support of and against the robustness of over-parameterized networks. These contradictory findings might be due to the failure of the attack employed to evaluate the networks' robustness. Previous research has demonstrated that depending on the considered model, the algorithm employed to generate adversarial examples may not function properly, leading to overestimating the model's robustness. In this work, we empirically study the robustness of over-parameterized networks against adversarial examples. However, unlike the previous works, we also evaluate the considered attack's reliability to support the results' veracity. Our results show that over-parameterized networks are robust against adversarial attacks as opposed to their under-parameterized counterparts.

摘要: 由于其广泛的容量，过度参数化神经网络展现出卓越的预测能力和概括性。然而，拥有大的参数空间被认为是神经网络容易受到对抗性示例影响的主要嫌疑人之一--即专门制作的输入样本，以引发所需的错误分类。相关文献在支持和反对过度参数化网络的鲁棒性方面提出了相互矛盾的言论。这些相互矛盾的发现可能是由于用于评估网络稳健性的攻击失败所致。之前的研究表明，根据所考虑的模型，用于生成对抗性示例的算法可能无法正常运行，从而导致高估模型的稳健性。在这项工作中，我们实证研究了过度参数化网络对对抗示例的鲁棒性。然而，与之前的作品不同的是，我们还评估了所考虑的攻击的可靠性，以支持结果的准确性。我们的结果表明，与参数化不足的网络相比，过度参数化的网络对对抗攻击具有鲁棒性。



## **47. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

MIIR：基于互信息的对抗鲁棒性的掩蔽图像建模 cs.CV

Accepted by NDSS 2026

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2312.04960v5) [paper-pdf](https://arxiv.org/pdf/2312.04960v5)

**Authors**: Xiaoyun Xu, Shujian Yu, Zhuoran Liu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) have emerged as a fundamental architecture and serve as the backbone of modern vision-language models. Despite their impressive performance, ViTs exhibit notable vulnerability to evasion attacks, necessitating the development of specialized Adversarial Training (AT) strategies tailored to their unique architecture. While a direct solution might involve applying existing AT methods to ViTs, our analysis reveals significant incompatibilities, particularly with state-of-the-art (SOTA) approaches such as Generalist (CVPR 2023) and DBAT (USENIX Security 2024). This paper presents a systematic investigation of adversarial robustness in ViTs and provides a novel theoretical Mutual Information (MI) analysis in its autoencoder-based self-supervised pre-training. Specifically, we show that MI between the adversarial example and its latent representation in ViT-based autoencoders should be constrained via derived MI bounds. Building on this insight, we propose a self-supervised AT method, MIMIR, that employs an MI penalty to facilitate adversarial pre-training by masked image modeling with autoencoders. Extensive experiments on CIFAR-10, Tiny-ImageNet, and ImageNet-1K show that MIMIR can consistently provide improved natural and robust accuracy, where MIMIR outperforms SOTA AT results on ImageNet-1K. Notably, MIMIR demonstrates superior robustness against unforeseen attacks and common corruption data and can also withstand adaptive attacks where the adversary possesses full knowledge of the defense mechanism. Our code and trained models are publicly available at: https://github.com/xiaoyunxxy/MIMIR.

摘要: 视觉变形者（ViT）已成为一种基本架构，并成为现代视觉语言模型的支柱。尽管ViT的性能令人印象深刻，但其对规避攻击表现出明显的脆弱性，因此需要开发针对其独特架构定制的专门对抗训练（AT）策略。虽然直接的解决方案可能涉及将现有的AT方法应用于ViT，但我们的分析揭示了显着的不兼容性，特别是与最先进的（SOTA）方法，例如Generalist（CVPR 2023）和DBAT（USENIX Security 2024）。本文对ViT中的对抗鲁棒性进行了系统研究，并在其基于自动编码器的自我监督预训练中提供了一种新颖的理论互信息（MI）分析。具体来说，我们表明对抗性示例及其在基于ViT的自动编码器中的潜在表示之间的MI应该通过推导出的MI界限来约束。基于这一见解，我们提出了一种自我监督的AT方法MIIR，它采用MI罚分来通过使用自动编码器进行掩蔽图像建模来促进对抗性预训练。CIFAR-10、Tiny-ImageNet和ImageNet-1 K上的大量实验表明，MIIR可以始终如一地提供改进的自然和稳健的准确性，其中MIIR优于ImageNet-1 K上的SOTA AT结果。值得注意的是，MIIR表现出针对不可预见的攻击和常见腐败数据的卓越鲁棒性，并且还可以抵御对手完全了解防御机制的自适应攻击。我们的代码和训练模型可在以下网址公开获取：https://github.com/xiaoyunxxy/MIMIR。



## **48. Enigma: Application-Layer Privacy for Quantum Optimization on Untrusted Computers**

Enigma：不可信计算机上量子优化的应用层隐私 quant-ph

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2311.13546v2) [paper-pdf](https://arxiv.org/pdf/2311.13546v2)

**Authors**: Ramin Ayanzadeh, Ahmad Mousavi, Amirhossein Basareh, Narges Alavisamani, Kazem Taram

**Abstract**: The Early Fault-Tolerant (EFT) era is emerging, where modest Quantum Error Correction (QEC) can enable quantum utility before full-scale fault tolerance. Quantum optimization is a leading candidate for early applications, but protecting these workloads is critical since they will run on expensive cloud services where providers could learn sensitive problem details. Experience with classical computing systems has shown that treating security as an afterthought can lead to significant vulnerabilities. Thus, we must address the security implications of quantum computing before widespread adoption. However, current Secure Quantum Computing (SQC) approaches, although theoretically promising, are impractical in the EFT era: blind quantum computing requires large-scale quantum networks, and quantum homomorphic encryption depends on full QEC.   We propose application-specific SQC, a principle that applies obfuscation at the application layer to enable practical deployment while remaining agnostic to algorithms, computing models, and hardware architectures. We present Enigma, the first realization of this principle for quantum optimization. Enigma integrates three complementary obfuscations: ValueGuard scrambles coefficients, StructureCamouflage inserts decoys, and TopologyTrimmer prunes variables. These techniques guarantee recovery of original solutions, and their stochastic nature resists repository-matching attacks. Evaluated against seven state-of-the-art AI models across five representative graph families, even combined adversaries, under a conservatively strong attacker model, identify the correct problem within their top five guesses in only 4.4% of cases. The protections come at the cost of problem size and T-gate counts increasing by averages of 1.07x and 1.13x, respectively, with both obfuscation and decoding completing within seconds for large-scale problems.

摘要: 早期故障容忍（EFT）时代正在兴起，适度的量子错误纠正（QEC）可以在全面故障容忍之前实现量子效用。量子优化是早期应用程序的主要候选者，但保护这些工作负载至关重要，因为它们将在昂贵的云服务上运行，提供商可以了解敏感的问题详细信息。经典计算系统的经验表明，将安全性视为事后考虑可能会导致重大漏洞。因此，我们必须在广泛采用量子计算之前解决量子计算的安全影响。然而，当前的安全量子计算（SQC）方法尽管在理论上很有希望，但在EFT时代是不切实际的：盲量子计算需要大规模量子网络，而量子同质加密依赖于完整的QEC。   我们提出了特定于应用程序的SQC，这一原则在应用程序层应用模糊处理，以实现实际部署，同时对算法、计算模型和硬件架构保持不可知。我们提出Enigma，这是量子优化原理的第一次实现。Enigma集成了三种互补的混淆：ValueGuard扰乱系数、StructureCamerage插入诱饵以及TopologyTrimmer修剪变量。这些技术保证原始解决方案的恢复，并且其随机性可以抵抗存储库匹配攻击。在保守强大的攻击者模型下，针对五个代表性图族的七个最先进的人工智能模型进行评估，即使是组合的对手，也只能在4.4%的情况下在前五个猜测中识别出正确的问题。这些保护的代价是问题大小和T门计数平均分别增加1.07倍和1.13倍，对于大规模问题，混淆和解码都在几秒钟内完成。



## **49. Why Does Little Robustness Help? A Further Step Towards Understanding Adversarial Transferability**

为什么小鲁棒性有帮助？了解对抗性可转让性的又一步 cs.LG

IEEE Symposium on Security and Privacy (Oakland) 2024; Extended version; Fix an proof error of Theorem 1

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2307.07873v8) [paper-pdf](https://arxiv.org/pdf/2307.07873v8)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.

摘要: DNN的对抗示例（AE）已经被证明是可转移的：成功欺骗白盒代理模型的AE也可以欺骗其他具有不同架构的黑盒模型。尽管大量的实证研究为产生高度可转移的不良事件提供了指导，但其中许多研究结果缺乏解释，甚至导致不一致的建议。在本文中，我们采取了进一步的理解对抗性的可转让性，特别侧重于代理方面。从有趣的小鲁棒性现象开始，其中使用轻度扰动的对抗样本进行对抗训练的模型可以作为更好的替代品，我们将其归因于两个主要因素之间的权衡：模型平滑度和梯度相似性。我们的研究重点是它们的联合影响，而不是它们与可转移性的单独相关性。通过一系列理论和实证分析，我们推测对抗训练中的数据分布变化解释了梯度相似性的下降。在这些见解的基础上，我们探索了数据增强和梯度正规化对可移植性的影响，并确定各种训练机制中普遍存在权衡，从而为可移植性背后的监管机制构建了全面的蓝图。最后，我们提供了一种通用路线来构建更好的代理以提高可移植性，该路线同时优化模型平滑度和梯度相似性，例如，输入梯度正规化和清晰度感知最小化（Sam）的结合，经过大量实验验证。总而言之，我们呼吁关注这两个因素对发起有效传输攻击的联合影响，而不是优化其中一个因素而忽略另一个因素，并强调操纵代理模型的关键作用。



## **50. A First Order Meta Stackelberg Method for Robust Federated Learning (Technical Report)**

鲁棒联邦学习的一阶Meta Stackelberg方法（技术报告） cs.CR

This submission is a technical report for "A First Order Meta Stackelberg Method for Robust Federated Learning" (arXiv:2306.13800). We later submitted a full paper, "Meta Stackelberg Game: Robust Federated Learning Against Adaptive and Mixed Poisoning Attacks" (arXiv:2410.17431), which fully incorporates this report in its Appendix. To avoid duplication, we withdraw this submission

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2306.13273v3) [paper-pdf](https://arxiv.org/pdf/2306.13273v3)

**Authors**: Henger Li, Tianyi Xu, Tao Li, Yunian Pan, Quanyan Zhu, Zizhan Zheng

**Abstract**: Recent research efforts indicate that federated learning (FL) systems are vulnerable to a variety of security breaches. While numerous defense strategies have been suggested, they are mainly designed to counter specific attack patterns and lack adaptability, rendering them less effective when facing uncertain or adaptive threats. This work models adversarial FL as a Bayesian Stackelberg Markov game (BSMG) between the defender and the attacker to address the lack of adaptability to uncertain adaptive attacks. We further devise an effective meta-learning technique to solve for the Stackelberg equilibrium, leading to a resilient and adaptable defense. The experiment results suggest that our meta-Stackelberg learning approach excels in combating intense model poisoning and backdoor attacks of indeterminate types.

摘要: 最近的研究工作表明，联邦学习（FL）系统容易受到各种安全漏洞的影响。虽然已经提出了许多防御策略，但它们主要是为了对抗特定的攻击模式而设计的，并且缺乏适应性，导致它们在面临不确定或适应性威胁时效果较差。这项工作将对抗FL建模为防御者和攻击者之间的Bayesian Stackelberg Markov博弈（BSMG），以解决对不确定的适应性攻击缺乏适应性的问题。我们进一步设计了一种有效的元学习技术来解决Stackelberg均衡，从而实现弹性和适应性强的防御。实验结果表明，我们的元Stackelberg学习方法在对抗不确定类型的强烈模型中毒和后门攻击方面表现出色。



