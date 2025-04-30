# Latest Adversarial Attack Papers
**update at 2025-04-30 16:22:34**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security**

AegisLLM：扩展统计系统以实现LLM安全中的自我反思防御 cs.LG

ICLR 2025 Workshop BuildingTrust

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20965v1) [paper-pdf](http://arxiv.org/pdf/2504.20965v1)

**Authors**: Zikui Cai, Shayan Shabihi, Bang An, Zora Che, Brian R. Bartoldson, Bhavya Kailkhura, Tom Goldstein, Furong Huang

**Abstract**: We introduce AegisLLM, a cooperative multi-agent defense against adversarial attacks and information leakage. In AegisLLM, a structured workflow of autonomous agents - orchestrator, deflector, responder, and evaluator - collaborate to ensure safe and compliant LLM outputs, while self-improving over time through prompt optimization. We show that scaling agentic reasoning system at test-time - both by incorporating additional agent roles and by leveraging automated prompt optimization (such as DSPy)- substantially enhances robustness without compromising model utility. This test-time defense enables real-time adaptability to evolving attacks, without requiring model retraining. Comprehensive evaluations across key threat scenarios, including unlearning and jailbreaking, demonstrate the effectiveness of AegisLLM. On the WMDP unlearning benchmark, AegisLLM achieves near-perfect unlearning with only 20 training examples and fewer than 300 LM calls. For jailbreaking benchmarks, we achieve 51% improvement compared to the base model on StrongReject, with false refusal rates of only 7.9% on PHTest compared to 18-55% for comparable methods. Our results highlight the advantages of adaptive, agentic reasoning over static defenses, establishing AegisLLM as a strong runtime alternative to traditional approaches based on model modifications. Code is available at https://github.com/zikuicai/aegisllm

摘要: 我们引入了AegisLLM，这是一种针对对抗攻击和信息泄露的协作多代理防御系统。在AegisLLM中，由自治代理（协调器、偏转器、响应者和评估器）组成的结构化工作流程相互协作，以确保安全合规的LLM输出，同时通过及时优化随着时间的推移进行自我改进。我们表明，在测试时扩展代理推理系统--通过合并额外的代理角色和利用自动化提示优化（例如DSPy）--可以在不损害模型效用的情况下大幅增强稳健性。这种测试时防御能够实时适应不断发展的攻击，而无需模型重新训练。对关键威胁场景（包括取消学习和越狱）的全面评估展示了AegisLLM的有效性。在WMDP取消学习基准上，AegisLLM仅使用20个训练示例和少于300个LM调用即可实现近乎完美的取消学习。对于越狱基准，与Strongestival上的基本模型相比，我们实现了51%的改进，PHTest上的错误拒绝率仅为7.9%，而类似方法的错误拒绝率为18-55%。我们的结果强调了自适应、代理推理相对于静态防御的优势，将AegisLLM确立为基于模型修改的传统方法的强大运行时替代方案。代码可在https://github.com/zikuicai/aegisllm上获得



## **2. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

NoPain：通过最佳传输奇异边界进行无箱点云攻击 cs.CV

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2503.00063v4) [paper-pdf](http://arxiv.org/pdf/2503.00063v4)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. Code and model are available at https://github.com/cognaclee/nopain

摘要: 对抗性攻击利用深度模型针对对抗性样本的脆弱性。现有的点云攻击者针对特定模型进行定制，根据白盒或黑盒设置中的梯度迭代优化扰动。尽管它们的攻击性能令人鼓舞，但由于过度匹配代理模型的特定参数，它们经常难以产生可转移的对抗样本。为了克服这个问题，我们将重点转移到数据分布本身上，并引入了一种名为NoPain的新颖方法，该方法采用最优传输（OT）来识别跨网络点云攻击的数据集的固有奇异边界。具体来说，我们首先计算从噪音到目标特征空间的OT映射，然后通过定位不可微位置来识别奇异边界。最后，我们沿着奇异边界进行采样以生成对抗点云。一旦确定奇异边界，NoPain就可以有效地生成对抗样本，而无需迭代更新或代理分类器的指导。大量实验表明，提出的端到端方法在可移植性和效率方面都优于基线方法，同时即使在防御策略方面也保持了显着的优势。代码和型号可在https://github.com/cognaclee/nopain上获得



## **3. Quantifying the Noise of Structural Perturbations on Graph Adversarial Attacks**

量化图对抗攻击的结构扰动噪音 cs.LG

Ubder Review

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20869v1) [paper-pdf](http://arxiv.org/pdf/2504.20869v1)

**Authors**: Junyuan Fang, Han Yang, Haixian Wen, Jiajing Wu, Zibin Zheng, Chi K. Tse

**Abstract**: Graph neural networks have been widely utilized to solve graph-related tasks because of their strong learning power in utilizing the local information of neighbors. However, recent studies on graph adversarial attacks have proven that current graph neural networks are not robust against malicious attacks. Yet much of the existing work has focused on the optimization objective based on attack performance to obtain (near) optimal perturbations, but paid less attention to the strength quantification of each perturbation such as the injection of a particular node/link, which makes the choice of perturbations a black-box model that lacks interpretability. In this work, we propose the concept of noise to quantify the attack strength of each adversarial link. Furthermore, we propose three attack strategies based on the defined noise and classification margins in terms of single and multiple steps optimization. Extensive experiments conducted on benchmark datasets against three representative graph neural networks demonstrate the effectiveness of the proposed attack strategies. Particularly, we also investigate the preferred patterns of effective adversarial perturbations by analyzing the corresponding properties of the selected perturbation nodes.

摘要: 图神经网络因其在利用邻居的局部信息方面具有强大的学习能力，已被广泛用于解决与图相关的任务。然而，最近关于图对抗攻击的研究证明，当前的图神经网络对恶意攻击并不强大。然而，现有的大部分工作都集中在基于攻击性能的优化目标上，以获得（接近）最佳扰动，但较少关注每个扰动的强度量化，例如特定节点/链路的注入，这使得扰动的选择成为缺乏可解释性的黑匣子模型。在这项工作中，我们提出了噪音的概念来量化每个对抗链接的攻击强度。此外，我们根据定义的噪音和分类裕度提出了三种攻击策略，从单步和多步优化角度进行。针对三个代表性图神经网络在基准数据集上进行的大量实验证明了所提出的攻击策略的有效性。特别是，我们还通过分析所选扰动节点的相应属性来研究有效对抗扰动的首选模式。



## **4. Mitigating the Structural Bias in Graph Adversarial Defenses**

缓解图对抗防御中的结构偏差 cs.LG

Under Review

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20848v1) [paper-pdf](http://arxiv.org/pdf/2504.20848v1)

**Authors**: Junyuan Fang, Huimin Liu, Han Yang, Jiajing Wu, Zibin Zheng, Chi K. Tse

**Abstract**: In recent years, graph neural networks (GNNs) have shown great potential in addressing various graph structure-related downstream tasks. However, recent studies have found that current GNNs are susceptible to malicious adversarial attacks. Given the inevitable presence of adversarial attacks in the real world, a variety of defense methods have been proposed to counter these attacks and enhance the robustness of GNNs. Despite the commendable performance of these defense methods, we have observed that they tend to exhibit a structural bias in terms of their defense capability on nodes with low degree (i.e., tail nodes), which is similar to the structural bias of traditional GNNs on nodes with low degree in the clean graph. Therefore, in this work, we propose a defense strategy by including hetero-homo augmented graph construction, $k$NN augmented graph construction, and multi-view node-wise attention modules to mitigate the structural bias of GNNs against adversarial attacks. Notably, the hetero-homo augmented graph consists of removing heterophilic links (i.e., links connecting nodes with dissimilar features) globally and adding homophilic links (i.e., links connecting nodes with similar features) for nodes with low degree. To further enhance the defense capability, an attention mechanism is adopted to adaptively combine the representations from the above two kinds of graph views. We conduct extensive experiments to demonstrate the defense and debiasing effect of the proposed strategy on benchmark datasets.

摘要: 近年来，图神经网络（GNN）在解决各种与图结构相关的下游任务方面表现出了巨大的潜力。然而，最近的研究发现，当前的GNN容易受到恶意对抗攻击。鉴于现实世界中不可避免地存在对抗攻击，人们提出了各种防御方法来对抗这些攻击并增强GNN的鲁棒性。尽管这些防御方法的性能值得赞扬，但我们观察到它们在低程度节点上的防御能力方面往往表现出结构性偏差（即，尾节点），这类似于传统GNN在干净图中度较低的节点上的结构偏差。因此，在这项工作中，我们提出了一种防御策略，包括异同增强图构造、$k$NN增强图构造和多视图节点注意力模块，以减轻GNN对对抗性攻击的结构偏见。值得注意的是，异同增强图包括去除异亲性链接（即，连接具有不同特征的节点的链接）全球并添加同同性链接（即，连接具有相似特征的节点的链接）对于程度较低的节点。为了进一步增强防御能力，采用关注机制自适应地组合上述两种图视图的表示。我们进行了广泛的实验来证明所提出的策略对基准数据集的防御和去偏置效果。



## **5. GaussTrap: Stealthy Poisoning Attacks on 3D Gaussian Splatting for Targeted Scene Confusion**

GaussTrap：对3D高斯飞溅进行隐形中毒攻击以造成目标场景混乱 cs.CV

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20829v1) [paper-pdf](http://arxiv.org/pdf/2504.20829v1)

**Authors**: Jiaxin Hong, Sixu Chen, Shuoyang Sun, Hongyao Yu, Hao Fang, Yuqi Tan, Bin Chen, Shuhan Qi, Jiawei Li

**Abstract**: As 3D Gaussian Splatting (3DGS) emerges as a breakthrough in scene representation and novel view synthesis, its rapid adoption in safety-critical domains (e.g., autonomous systems, AR/VR) urgently demands scrutiny of potential security vulnerabilities. This paper presents the first systematic study of backdoor threats in 3DGS pipelines. We identify that adversaries may implant backdoor views to induce malicious scene confusion during inference, potentially leading to environmental misperception in autonomous navigation or spatial distortion in immersive environments. To uncover this risk, we propose GuassTrap, a novel poisoning attack method targeting 3DGS models. GuassTrap injects malicious views at specific attack viewpoints while preserving high-quality rendering in non-target views, ensuring minimal detectability and maximizing potential harm. Specifically, the proposed method consists of a three-stage pipeline (attack, stabilization, and normal training) to implant stealthy, viewpoint-consistent poisoned renderings in 3DGS, jointly optimizing attack efficacy and perceptual realism to expose security risks in 3D rendering. Extensive experiments on both synthetic and real-world datasets demonstrate that GuassTrap can effectively embed imperceptible yet harmful backdoor views while maintaining high-quality rendering in normal views, validating its robustness, adaptability, and practical applicability.

摘要: 随着3D高斯飞溅（3DGS）成为场景表示和新颖视图合成领域的突破，它在安全关键领域（例如，自治系统（AR/VR）迫切需要对潜在的安全漏洞进行审查。本文首次对3DGS管道中的后门威胁进行了系统研究。我们发现对手可能会植入后门视图，以在推理过程中引发恶意场景混乱，这可能会导致自主导航中的环境误解或沉浸式环境中的空间失真。为了发现这一风险，我们提出了GuassTrap，这是一种针对3DGS模型的新型中毒攻击方法。GuassTrap在特定的攻击观点处注入恶意视图，同时在非目标视图中保留高质量渲染，确保最小的可检测性并最大化潜在危害。具体来说，所提出的方法由三阶段管道（攻击、稳定和正常训练）组成，在3DGS中植入隐形、观点一致的有毒渲染，共同优化攻击功效和感知真实感，以暴露3D渲染中的安全风险。对合成和现实世界数据集的广泛实验表明，GuassTrap可以有效地嵌入难以感知但有害的后门视图，同时在正常视图中保持高质量的渲染，验证了其稳健性、适应性和实际适用性。



## **6. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

正规的鲁棒可靠的学习者和实例有针对性的攻击 cs.LG

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2410.10572v3) [paper-pdf](http://arxiv.org/pdf/2410.10572v3)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.

摘要: 针对实例的数据中毒攻击（对手破坏训练集以在特定测试点上引发错误）引发了严重担忧。Balcan等人（2022）提出了一种解决这一挑战的方法，通过定义鲁棒可靠学习器的概念，即使存在数据中毒攻击，也可以在明确定义的假设下提供每个实例的正确性保证。然后，他们给出了一个通用的最佳（但计算效率低下）鲁棒可靠的学习器，以及一个计算高效的算法，用于线性分离器在线性分离器的情况。   在这项工作中，我们解决了Balcan等人（2022）留下的两个挑战。首先，Balcan et al（2022）中对鲁棒可靠学习者的定义对于高度灵活的假设类别来说变得空洞：如果H中有两个分类器h_0、h_1 \，两者在训练集上的误差为零，使得h_0（x）\neq h_1（x），那么鲁棒可靠学习者必须放弃x。我们通过定义一个修改的正规化鲁棒可靠学习器概念来解决这个问题，该概念允许在这种情况下的非平凡陈述。其次，Balcan等人（2022）的通用算法需要在每个测试点x上重新运行ERM Oracle（本质上是重新训练分类器），即使可以有效地实施ERM，这通常也是不切实际的。为了解决这个问题，我们表明，至少在某些有趣的情况下，我们可以通过使用动态算法设计的技术来设计可以在训练时间内产生次线性输出的算法。



## **7. A Survey on Adversarial Contention Resolution**

对抗性竞争解决方法研究 cs.DC

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2403.03876v4) [paper-pdf](http://arxiv.org/pdf/2403.03876v4)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We also highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.

摘要: 竞争解决解决了协调多个进程对共享资源（例如内存、磁盘存储或通信通道）的访问的挑战。尽管经历了几十年的技术变革，竞争解决最初是受数据库系统和公交网络挑战的推动，但它仍然作为资源共享的重要抽象而经久不衰。在这里，我们调查了有关解决最坏情况争用的文献，其中进程的数量以及每个进程可能开始寻求访问资源的时间由对手决定。我们还强调了竞争解决方案的演变，现代系统引发了新的担忧（例如安全性、服务质量和能源效率）。这些努力深入了解了随机和确定性方法的局限性，以及不同模型假设（例如全球时钟同步、处理器数量的知识、访问尝试的反馈以及对共享资源可用性的攻击）的影响。



## **8. Data Encryption Battlefield: A Deep Dive into the Dynamic Confrontations in Ransomware Attacks**

数据加密战场：深入探讨勒索软件攻击中的动态对抗 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20681v1) [paper-pdf](http://arxiv.org/pdf/2504.20681v1)

**Authors**: Arash Mahboubi, Hamed Aboutorab, Seyit Camtepe, Hang Thanh Bui, Khanh Luong, Keyvan Ansari, Shenlu Wang, Bazara Barry

**Abstract**: In the rapidly evolving landscape of cybersecurity threats, ransomware represents a significant challenge. Attackers increasingly employ sophisticated encryption methods, such as entropy reduction through Base64 encoding, and partial or intermittent encryption to evade traditional detection methods. This study explores the dynamic battle between adversaries who continuously refine encryption strategies and defenders developing advanced countermeasures to protect vulnerable data. We investigate the application of online incremental machine learning algorithms designed to predict file encryption activities despite adversaries evolving obfuscation techniques. Our analysis utilizes an extensive dataset of 32.6 GB, comprising 11,928 files across multiple formats, including Microsoft Word documents (doc), PowerPoint presentations (ppt), Excel spreadsheets (xlsx), image formats (jpg, jpeg, png, tif, gif), PDFs (pdf), audio (mp3), and video (mp4) files. These files were encrypted by 75 distinct ransomware families, facilitating a robust empirical evaluation of machine learning classifiers effectiveness against diverse encryption tactics. Results highlight the Hoeffding Tree algorithms superior incremental learning capability, particularly effective in detecting traditional and AES-Base64 encryption methods employed to lower entropy. Conversely, the Random Forest classifier with warm-start functionality excels at identifying intermittent encryption methods, demonstrating the necessity of tailored machine learning solutions to counter sophisticated ransomware strategies.

摘要: 在迅速变化的网络安全威胁格局中，勒索软件构成了一个重大挑战。攻击者越来越多地使用复杂的加密方法，例如通过Base 64编码进行的减序，以及部分或间歇性加密来逃避传统的检测方法。本研究探讨了不断完善加密策略的对手与开发高级对策来保护脆弱数据的防御者之间的动态战斗。我们研究了在线增量机器学习算法的应用，该算法旨在预测文件加密活动，尽管对手不断发展混淆技术。我们的分析利用了32.6 GB的广泛数据集，包括多种格式的11，928个文件，包括Microsoft Word文档（Doc）、PowerPoint演示文稿（GPT）、Excel电子表格（xlsx）、图像格式（jpg、jpeg、png、tif、gif）、PDF（pdf）、音频（mp3）和视频（mp4）文件。这些文件由75个不同的勒索软件家族加密，有助于对机器学习分类器针对不同加密策略的有效性进行稳健的实证评估。结果凸显了Hoeffding Tree算法优越的增量学习能力，在检测用于降低信息量的传统加密方法和AES-Base 64加密方法方面特别有效。相反，具有热启动功能的随机森林分类器擅长识别间歇性加密方法，证明了定制机器学习解决方案来对抗复杂勒索软件策略的必要性。



## **9. Learning and Generalization with Mixture Data**

使用混合数据学习和概括 stat.ML

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20651v1) [paper-pdf](http://arxiv.org/pdf/2504.20651v1)

**Authors**: Harsh Vardhan, Avishek Ghosh, Arya Mazumdar

**Abstract**: In many, if not most, machine learning applications the training data is naturally heterogeneous (e.g. federated learning, adversarial attacks and domain adaptation in neural net training). Data heterogeneity is identified as one of the major challenges in modern day large-scale learning. A classical way to represent heterogeneous data is via a mixture model. In this paper, we study generalization performance and statistical rates when data is sampled from a mixture distribution. We first characterize the heterogeneity of the mixture in terms of the pairwise total variation distance of the sub-population distributions. Thereafter, as a central theme of this paper, we characterize the range where the mixture may be treated as a single (homogeneous) distribution for learning. In particular, we study the generalization performance under the classical PAC framework and the statistical error rates for parametric (linear regression, mixture of hyperplanes) as well as non-parametric (Lipschitz, convex and H\"older-smooth) regression problems. In order to do this, we obtain Rademacher complexity and (local) Gaussian complexity bounds with mixture data, and apply them to get the generalization and convergence rates respectively. We observe that as the (regression) function classes get more complex, the requirement on the pairwise total variation distance gets stringent, which matches our intuition. We also do a finer analysis for the case of mixed linear regression and provide a tight bound on the generalization error in terms of heterogeneity.

摘要: 在许多（如果不是大多数）机器学习应用中，训练数据自然是异类的（例如联邦学习、对抗性攻击和神经网络训练中的领域适应）。数据异类被认为是现代大规模学习的主要挑战之一。表示异类数据的经典方法是通过混合模型。本文研究了从混合分布中采样数据时的概括性能和统计率。我们首先根据亚群体分布的成对总变异距离来描述混合物的均匀性。此后，作为本文的中心主题，我们描述了混合物可以被视为单一（均匀）学习分布的范围。特别是，我们研究了经典PAC框架下的推广性能以及参数（线性回归、超平面混合）以及非参数（Lipschitz、凸和H ' old smooth）回归问题的统计错误率。为了做到这一点，我们通过混合数据获得Rademacher复杂度和（局部）高斯复杂度界限，并分别应用它们来获得概括率和收敛率。我们观察到，随着（回归）函数类变得越来越复杂，对成对总变异距离的要求变得严格，这符合我们的直觉。我们还对混合线性回归的情况进行了更细致的分析，并在方差方面为概括误差提供了严格的界限。



## **10. WILD: a new in-the-Wild Image Linkage Dataset for synthetic image attribution**

WILD：一个新的用于合成图像属性的野外图像链接数据集 cs.MM

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.19595v2) [paper-pdf](http://arxiv.org/pdf/2504.19595v2)

**Authors**: Pietro Bongini, Sara Mandelli, Andrea Montibeller, Mirko Casu, Orazio Pontorno, Claudio Vittorio Ragaglia, Luca Zanchetta, Mattia Aquilina, Taiba Majid Wani, Luca Guarnera, Benedetta Tondi, Giulia Boato, Paolo Bestagini, Irene Amerini, Francesco De Natale, Sebastiano Battiato, Mauro Barni

**Abstract**: Synthetic image source attribution is an open challenge, with an increasing number of image generators being released yearly. The complexity and the sheer number of available generative techniques, as well as the scarcity of high-quality open source datasets of diverse nature for this task, make training and benchmarking synthetic image source attribution models very challenging. WILD is a new in-the-Wild Image Linkage Dataset designed to provide a powerful training and benchmarking tool for synthetic image attribution models. The dataset is built out of a closed set of 10 popular commercial generators, which constitutes the training base of attribution models, and an open set of 10 additional generators, simulating a real-world in-the-wild scenario. Each generator is represented by 1,000 images, for a total of 10,000 images in the closed set and 10,000 images in the open set. Half of the images are post-processed with a wide range of operators. WILD allows benchmarking attribution models in a wide range of tasks, including closed and open set identification and verification, and robust attribution with respect to post-processing and adversarial attacks. Models trained on WILD are expected to benefit from the challenging scenario represented by the dataset itself. Moreover, an assessment of seven baseline methodologies on closed and open set attribution is presented, including robustness tests with respect to post-processing.

摘要: 合成图像源归属是一个公开的挑战，每年发布的图像生成器数量越来越多。可用生成技术的复杂性和数量之多，以及用于该任务的多样化高质量开源数据集的稀缺性，使得训练和基准合成图像源归因模型变得非常具有挑战性。WILD是一个新的野外图像联动数据集，旨在为合成图像归因模型提供强大的训练和基准测试工具。该数据集由一组由10个流行的商业生成器组成的封闭集和一组由10个额外生成器组成的开放集构建，该生成器构成了归因模型的训练基础，模拟了现实世界的野外场景。每个生成器由1，000个图像表示，闭集中总共有10，000个图像，开集中有10，000个图像。一半的图像经过各种操作员的后处理。WILD允许在广泛的任务中对归因模型进行基准测试，包括封闭和开放集识别和验证，以及针对后处理和对抗攻击的稳健归因。在WILD上训练的模型预计将受益于数据集本身所代表的具有挑战性的场景。此外，还对封闭集和开放集归因的七种基线方法进行了评估，包括后处理方面的稳健性测试。



## **11. Enhancing Leakage Attacks on Searchable Symmetric Encryption Using LLM-Based Synthetic Data Generation**

使用基于LLM的合成数据生成增强对可搜索对称加密的泄漏攻击 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20414v1) [paper-pdf](http://arxiv.org/pdf/2504.20414v1)

**Authors**: Joshua Chiu, Partha Protim Paul, Zahin Wahab

**Abstract**: Searchable Symmetric Encryption (SSE) enables efficient search capabilities over encrypted data, allowing users to maintain privacy while utilizing cloud storage. However, SSE schemes are vulnerable to leakage attacks that exploit access patterns, search frequency, and volume information. Existing studies frequently assume that adversaries possess a substantial fraction of the encrypted dataset to mount effective inference attacks, implying there is a database leakage of such documents, thus, an assumption that may not hold in real-world scenarios. In this work, we investigate the feasibility of enhancing leakage attacks under a more realistic threat model in which adversaries have access to minimal leaked data. We propose a novel approach that leverages large language models (LLMs), specifically GPT-4 variants, to generate synthetic documents that statistically and semantically resemble the real-world dataset of Enron emails. Using the email corpus as a case study, we evaluate the effectiveness of synthetic data generated via random sampling and hierarchical clustering methods on the performance of the SAP (Search Access Pattern) keyword inference attack restricted to token volumes only. Our results demonstrate that, while the choice of LLM has limited effect, increasing dataset size and employing clustering-based generation significantly improve attack accuracy, achieving comparable performance to attacks using larger amounts of real data. We highlight the growing relevance of LLMs in adversarial contexts.

摘要: 可搜索对称加密（SSE）支持对加密数据进行高效搜索，使用户能够在利用云存储的同时维护隐私。然而，SSE方案很容易受到利用访问模式、搜索频率和量信息的泄露攻击。现有的研究经常假设对手拥有很大一部分加密数据集来发起有效的推理攻击，这意味着此类文档的数据库泄露，因此，这一假设在现实世界的场景中可能不成立。在这项工作中，我们研究了在更现实的威胁模型下增强泄露攻击的可行性，其中对手可以访问最少的泄露数据。我们提出了一种新颖的方法，利用大型语言模型（LLM），特别是GPT-4变体，来生成在统计和语义上与安然电子邮件的现实世界数据集相似的合成文档。使用电子邮件库作为案例研究，我们评估了通过随机抽样和分层集群方法生成的合成数据对仅限于令牌量的SAP（搜索访问模式）关键字推理攻击性能的有效性。我们的结果表明，虽然选择LLM的效果有限，但增加数据集大小和采用基于集群的生成可以显着提高攻击准确性，实现与使用大量真实数据的攻击相当的性能。我们强调法学硕士在对抗背景下日益增长的相关性。



## **12. Inception: Jailbreak the Memory Mechanism of Text-to-Image Generation Systems**

盗梦空间：越狱文本到图像生成系统的记忆机制 cs.CV

17 pages, 8 figures

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20376v1) [paper-pdf](http://arxiv.org/pdf/2504.20376v1)

**Authors**: Shiqian Zhao, Jiayang Liu, Yiming Li, Runyi Hu, Xiaojun Jia, Wenshu Fan, Xinfeng Li, Jie Zhang, Wei Dong, Tianwei Zhang, Luu Anh Tuan

**Abstract**: Currently, the memory mechanism has been widely and successfully exploited in online text-to-image (T2I) generation systems ($e.g.$, DALL$\cdot$E 3) for alleviating the growing tokenization burden and capturing key information in multi-turn interactions. Despite its practicality, its security analyses have fallen far behind. In this paper, we reveal that this mechanism exacerbates the risk of jailbreak attacks. Different from previous attacks that fuse the unsafe target prompt into one ultimate adversarial prompt, which can be easily detected or may generate non-unsafe images due to under- or over-optimization, we propose Inception, the first multi-turn jailbreak attack against the memory mechanism in real-world text-to-image generation systems. Inception embeds the malice at the inception of the chat session turn by turn, leveraging the mechanism that T2I generation systems retrieve key information in their memory. Specifically, Inception mainly consists of two modules. It first segments the unsafe prompt into chunks, which are subsequently fed to the system in multiple turns, serving as pseudo-gradients for directive optimization. Specifically, we develop a series of segmentation policies that ensure the images generated are semantically consistent with the target prompt. Secondly, after segmentation, to overcome the challenge of the inseparability of minimum unsafe words, we propose recursion, a strategy that makes minimum unsafe words subdivisible. Collectively, segmentation and recursion ensure that all the request prompts are benign but can lead to malicious outcomes. We conduct experiments on the real-world text-to-image generation system ($i.e.$, DALL$\cdot$E 3) to validate the effectiveness of Inception. The results indicate that Inception surpasses the state-of-the-art by a 14\% margin in attack success rate.

摘要: 目前，存储机制已在在线文本到图像（T2 I）生成系统中得到广泛且成功的利用（$e.g.$，DALL$\csot $E 3）用于减轻日益增长的代币化负担并捕获多回合交互中的关键信息。尽管它实用，但它的安全分析却远远落后。在本文中，我们揭示了这种机制加剧了越狱攻击的风险。与之前将不安全的目标提示融合为一个终极对抗提示的攻击不同，这种攻击可以很容易地检测到，或者可能会由于优化不足或过度而生成非不安全的图像，我们提出了Incept，这是第一个针对现实世界中的存储机制的多回合越狱攻击。文本到图像生成系统。Incement利用T2 I生成系统在其内存中检索关键信息的机制，在聊天会话开始时轮流嵌入恶意。具体来说，Incion主要由两个模块组成。它首先将不安全的提示分割成块，随后将这些块分多次输送到系统，作为指令优化的伪梯度。具体来说，我们开发了一系列分割策略，以确保生成的图像在语义上与目标提示一致。其次，在分段之后，为了克服最小不安全词不可分割的挑战，我们提出了回归，这是一种使最小不安全词可细分的策略。总的来说，分段和回归确保所有请求提示都是良性的，但可能会导致恶意结果。我们对现实世界的文本到图像生成系统（$i.e.$，DALL$\csot $E 3）验证Incession的有效性。结果表明，Incion的攻击成功率比最新技术水平高出14%。



## **13. A Cryptographic Perspective on Mitigation vs. Detection in Machine Learning**

机器学习中缓解与检测的密码学视角 cs.LG

29 pages

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20310v1) [paper-pdf](http://arxiv.org/pdf/2504.20310v1)

**Authors**: Greg Gluch, Shafi Goldwasser

**Abstract**: In this paper, we initiate a cryptographically inspired theoretical study of detection versus mitigation of adversarial inputs produced by attackers of Machine Learning algorithms during inference time.   We formally define defense by detection (DbD) and defense by mitigation (DbM). Our definitions come in the form of a 3-round protocol between two resource-bounded parties: a trainer/defender and an attacker. The attacker aims to produce inference-time inputs that fool the training algorithm. We define correctness, completeness, and soundness properties to capture successful defense at inference time while not degrading (too much) the performance of the algorithm on inputs from the training distribution.   We first show that achieving DbD and achieving DbM are equivalent for ML classification tasks. Surprisingly, this is not the case for ML generative learning tasks, where there are many possible correct outputs that can be generated for each input. We show a separation between DbD and DbM by exhibiting a generative learning task for which is possible to defend by mitigation but is provably impossible to defend by detection under the assumption that the Identity-Based Fully Homomorphic Encryption (IB-FHE), publicly-verifiable zero-knowledge Succinct Non-Interactive Arguments of Knowledge (zk-SNARK) and Strongly Unforgeable Signatures exist. The mitigation phase uses significantly fewer samples than the initial training algorithm.

摘要: 在本文中，我们启动了一项受密码启发的理论研究，研究机器学习算法攻击者在推理时间内产生的对抗输入的检测与缓解。   我们正式定义了检测防御（GbD）和缓解防御（GbM）。我们的定义以两个资源有限方之间的三轮协议的形式出现：训练者/防御者和攻击者。攻击者的目标是产生欺骗训练算法的推断时输入。我们定义了正确性、完整性和可靠性属性，以在推理时捕获成功的防御，同时不会降低（太多）算法在训练分布输入上的性能。   我们首先表明，实现GbD和实现GbM对于ML分类任务来说是等效的。令人惊讶的是，ML生成式学习任务的情况并非如此，其中可以为每个输入生成许多可能的正确输出。我们通过展示生成式学习任务来展示GbD和GbM之间的分离，该任务可以通过缓解来防御，但在假设基于身份的完全同质加密（IB-FHE）、可公开验证的零知识连续非交互式知识参数（zk-SNARK）和强不可伪造签名存在的假设下，该任务可以通过缓解来防御。缓解阶段使用的样本比初始训练算法少得多。



## **14. The Dark Side of Digital Twins: Adversarial Attacks on AI-Driven Water Forecasting**

数字双胞胎的阴暗面：对人工智能驱动的水预测的对抗攻击 cs.LG

7 Pages, 7 Figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20295v1) [paper-pdf](http://arxiv.org/pdf/2504.20295v1)

**Authors**: Mohammadhossein Homaei, Victor Gonzalez Morales, Oscar Mogollon-Gutierrez, Andres Caro

**Abstract**: Digital twins (DTs) are improving water distribution systems by using real-time data, analytics, and prediction models to optimize operations. This paper presents a DT platform designed for a Spanish water supply network, utilizing Long Short-Term Memory (LSTM) networks to predict water consumption. However, machine learning models are vulnerable to adversarial attacks, such as the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). These attacks manipulate critical model parameters, injecting subtle distortions that degrade forecasting accuracy. To further exploit these vulnerabilities, we introduce a Learning Automata (LA) and Random LA-based approach that dynamically adjusts perturbations, making adversarial attacks more difficult to detect. Experimental results show that this approach significantly impacts prediction reliability, causing the Mean Absolute Percentage Error (MAPE) to rise from 26% to over 35%. Moreover, adaptive attack strategies amplify this effect, highlighting cybersecurity risks in AI-driven DTs. These findings emphasize the urgent need for robust defenses, including adversarial training, anomaly detection, and secure data pipelines.

摘要: 数字双胞胎（DT）正在通过使用实时数据、分析和预测模型来优化运营来改善供水系统。本文介绍了一个为西班牙供水网络设计的DT平台，利用长短期记忆（LSTM）网络来预测用水量。然而，机器学习模型很容易受到对抗攻击，例如快速梯度符号法（FGSM）和投影梯度下降（PVD）。这些攻击操纵关键模型参数，注入微妙的扭曲，降低预测准确性。为了进一步利用这些漏洞，我们引入了学习自动机（LA）和基于随机LA的方法，该方法动态调整扰动，使对抗性攻击更难以检测。实验结果表明，这种方法显着影响预测可靠性，导致平均绝对百分比误差（MAPE）从26%上升到35%以上。此外，自适应攻击策略放大了这种影响，凸显了人工智能驱动的DT中的网络安全风险。这些发现强调了对强大防御的迫切需要，包括对抗训练、异常检测和安全数据管道。



## **15. A Case Study on the Use of Representativeness Bias as a Defense Against Adversarial Cyber Threats**

使用代表性偏见作为对抗性网络威胁防御的案例研究 cs.CR

To appear in the 4th Workshop on Active Defense and Deception (ADnD),  co-located with the 10th IEEE European Symposium on Security and Privacy  (EuroS&P 2025)

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20245v1) [paper-pdf](http://arxiv.org/pdf/2504.20245v1)

**Authors**: Briland Hitaj, Grit Denker, Laura Tinnel, Michael McAnally, Bruce DeBruhl, Nathan Bunting, Alex Fafard, Daniel Aaron, Richard D. Roberts, Joshua Lawson, Greg McCain, Dylan Starink

**Abstract**: Cyberspace is an ever-evolving battleground involving adversaries seeking to circumvent existing safeguards and defenders aiming to stay one step ahead by predicting and mitigating the next threat. Existing mitigation strategies have focused primarily on solutions that consider software or hardware aspects, often ignoring the human factor. This paper takes a first step towards psychology-informed, active defense strategies, where we target biases that human beings are susceptible to under conditions of uncertainty.   Using capture-the-flag events, we create realistic challenges that tap into a particular cognitive bias: representativeness. This study finds that this bias can be triggered to thwart hacking attempts and divert hackers into non-vulnerable attack paths. Participants were exposed to two different challenges designed to exploit representativeness biases. One of the representativeness challenges significantly thwarted attackers away from vulnerable attack vectors and onto non-vulnerable paths, signifying an effective bias-based defense mechanism. This work paves the way towards cyber defense strategies that leverage additional human biases to thwart future, sophisticated adversarial attacks.

摘要: 网络空间是一个不断发展的战场，对手试图绕过现有的保障措施，而防御者则试图通过预测和减轻下一个威胁来领先一步。现有的缓解策略主要集中在考虑软件或硬件方面的解决方案上，通常忽视人为因素。本文朝着基于心理的主动防御策略迈出了第一步，我们针对人类在不确定条件下容易受到的偏见。   使用夺旗事件，我们创造了现实的挑战，这些挑战利用了特定的认知偏见：代表性。这项研究发现，这种偏见可以被触发来阻止黑客企图并将黑客转移到非脆弱的攻击路径上。参与者面临两种不同的挑战，旨在利用代表性偏见。代表性挑战之一显着阻止攻击者远离脆弱的攻击载体并转向非脆弱的路径，这意味着有效的基于偏差的防御机制。这项工作为网络防御策略铺平了道路，这些策略利用额外的人类偏见来阻止未来复杂的对抗性攻击。



## **16. DROP: Poison Dilution via Knowledge Distillation for Federated Learning**

Drop：通过联邦学习的知识蒸馏进行毒药稀释 cs.LG

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2502.07011v2) [paper-pdf](http://arxiv.org/pdf/2502.07011v2)

**Authors**: Georgios Syros, Anshuman Suri, Farinaz Koushanfar, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Federated Learning is vulnerable to adversarial manipulation, where malicious clients can inject poisoned updates to influence the global model's behavior. While existing defense mechanisms have made notable progress, they fail to protect against adversaries that aim to induce targeted backdoors under different learning and attack configurations. To address this limitation, we introduce DROP (Distillation-based Reduction Of Poisoning), a novel defense mechanism that combines clustering and activity-tracking techniques with extraction of benign behavior from clients via knowledge distillation to tackle stealthy adversaries that manipulate low data poisoning rates and diverse malicious client ratios within the federation. Through extensive experimentation, our approach demonstrates superior robustness compared to existing defenses across a wide range of learning configurations. Finally, we evaluate existing defenses and our method under the challenging setting of non-IID client data distribution and highlight the challenges of designing a resilient FL defense in this setting.

摘要: 联邦学习很容易受到对抗操纵的影响，恶意客户端可以注入有毒更新来影响全局模型的行为。虽然现有的防御机制取得了显着进展，但它们未能防止对手在不同的学习和攻击配置下诱导有针对性的后门。为了解决这一局限性，我们引入了Dopp（基于蒸馏的中毒减少），这是一种新型防御机制，它将集群和活动跟踪技术与通过知识蒸馏从客户端提取良性行为相结合，以应对操纵低数据中毒率和联邦内各种恶意客户比例的隐形对手。通过广泛的实验，与广泛的学习配置中的现有防御相比，我们的方法表现出了卓越的鲁棒性。最后，我们在非IID客户端数据分发的具有挑战性的环境下评估了现有的防御措施和我们的方法，并强调了在这种环境下设计弹性FL防御的挑战。



## **17. Evaluate-and-Purify: Fortifying Code Language Models Against Adversarial Attacks Using LLM-as-a-Judge**

评估和净化：使用LLM作为法官来加强代码语言模型对抗性攻击 cs.SE

25 pages, 6 figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19730v1) [paper-pdf](http://arxiv.org/pdf/2504.19730v1)

**Authors**: Wenhan Mu, Ling Xu, Shuren Pei, Le Mi, Huichi Zhou

**Abstract**: The widespread adoption of code language models in software engineering tasks has exposed vulnerabilities to adversarial attacks, especially the identifier substitution attacks. Although existing identifier substitution attackers demonstrate high success rates, they often produce adversarial examples with unnatural code patterns. In this paper, we systematically assess the quality of adversarial examples using LLM-as-a-Judge. Our analysis reveals that over 80% of adversarial examples generated by state-of-the-art identifier substitution attackers (e.g., ALERT) are actually detectable. Based on this insight, we propose EP-Shield, a unified framework for evaluating and purifying identifier substitution attacks via naturalness-aware reasoning. Specifically, we first evaluate the naturalness of code and identify the perturbed adversarial code, then purify it so that the victim model can restore correct prediction. Extensive experiments demonstrate the superiority of EP-Shield over adversarial fine-tuning (up to 83.36% improvement) and its lightweight design 7B parameters) with GPT-4-level performance.

摘要: 代码语言模型在软件工程任务中的广泛采用暴露了对抗攻击的脆弱性，尤其是标识符替换攻击。尽管现有的标识符替换攻击者表现出很高的成功率，但他们经常产生具有不自然代码模式的对抗性示例。在本文中，我们使用LLM作为法官系统评估了对抗性示例的质量。我们的分析表明，超过80%的对抗示例是由最先进的标识符替换攻击者生成的（例如，警报）实际上是可检测的。基于这一见解，我们提出了EP-Shield，一个统一的框架，通过自然感知推理评估和净化标识符替换攻击。具体来说，我们首先评估代码的自然性并识别出受干扰的对抗代码，然后对其进行净化，以便受害者模型能够恢复正确的预测。大量的实验证明了EP-Shield在对抗性微调方面的优越性（高达83.36%的改进）及其轻量级设计（7 B参数），具有GPT-4级性能。



## **18. Fooling the Decoder: An Adversarial Attack on Quantum Error Correction**

欺骗解码器：对量子纠错的对抗性攻击 quant-ph

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19651v1) [paper-pdf](http://arxiv.org/pdf/2504.19651v1)

**Authors**: Jerome Lenssen, Alexandru Paler

**Abstract**: Neural network decoders are becoming essential for achieving fault-tolerant quantum computations. However, their internal mechanisms are poorly understood, hindering our ability to ensure their reliability and security against adversarial attacks. Leading machine learning decoders utilize recurrent and transformer models (e.g., AlphaQubit), with reinforcement learning (RL) playing a key role in training advanced transformer models (e.g., DeepSeek R1). In this work, we target a basic RL surface code decoder (DeepQ) to create the first adversarial attack on quantum error correction. By applying state-of-the-art white-box methods, we uncover vulnerabilities in this decoder, demonstrating an attack that reduces the logical qubit lifetime in memory experiments by up to five orders of magnitude. We validate that this attack exploits a genuine weakness, as the decoder exhibits robustness against noise fluctuations, is largely unaffected by substituting the referee decoder, responsible for episode termination, with an MWPM decoder, and demonstrates fault tolerance at checkable code distances. This attack highlights the susceptibility of machine learning-based QEC and underscores the importance of further research into robust QEC methods.

摘要: 神经网络解码器对于实现耐故障量子计算来说变得至关重要。然而，人们对它们的内部机制知之甚少，这阻碍了我们确保其可靠性和安全性抵御对抗攻击的能力。领先的机器学习解码器利用循环模型和Transformer模型（例如，AlphaQubit），强化学习（RL）在训练高级Transformer模型（例如，DeepSeek R1）。在这项工作中，我们以基本RL表面码解码器（DeepQ）为目标，以创建对量子错误纠正的第一次对抗攻击。通过应用最先进的白盒方法，我们发现了这个解码器中的漏洞，展示了一种将内存实验中逻辑量子位寿命缩短多达五个数量级的攻击。我们验证了这种攻击利用了一个真正的弱点，因为解码器表现出对噪音波动的鲁棒性，用MWPM解码器替换负责剧集终止的裁判解码器在很大程度上不受影响，并且在可检查的代码距离上表现出了故障容忍性。这次攻击凸显了基于机器学习的QEC的易感性，并强调了进一步研究稳健的QEC方法的重要性。



## **19. FCGHunter: Towards Evaluating Robustness of Graph-Based Android Malware Detection**

FCGHunter：评估基于图形的Android恶意软件检测的稳健性 cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19456v1) [paper-pdf](http://arxiv.org/pdf/2504.19456v1)

**Authors**: Shiwen Song, Xiaofei Xie, Ruitao Feng, Qi Guo, Sen Chen

**Abstract**: Graph-based detection methods leveraging Function Call Graphs (FCGs) have shown promise for Android malware detection (AMD) due to their semantic insights. However, the deployment of malware detectors in dynamic and hostile environments raises significant concerns about their robustness. While recent approaches evaluate the robustness of FCG-based detectors using adversarial attacks, their effectiveness is constrained by the vast perturbation space, particularly across diverse models and features.   To address these challenges, we introduce FCGHunter, a novel robustness testing framework for FCG-based AMD systems. Specifically, FCGHunter employs innovative techniques to enhance exploration and exploitation within this huge search space. Initially, it identifies critical areas within the FCG related to malware behaviors to narrow down the perturbation space. We then develop a dependency-aware crossover and mutation method to enhance the validity and diversity of perturbations, generating diverse FCGs. Furthermore, FCGHunter leverages multi-objective feedback to select perturbed FCGs, significantly improving the search process with interpretation-based feature change feedback.   Extensive evaluations across 40 scenarios demonstrate that FCGHunter achieves an average attack success rate of 87.9%, significantly outperforming baselines by at least 44.7%. Notably, FCGHunter achieves a 100% success rate on robust models (e.g., AdaBoost with MalScan), where baselines achieve only 11% or are inapplicable.

摘要: 利用函数调用图（FCG）的基于图的检测方法因其语义洞察而显示出Android恶意软件检测（AMD）的前景。然而，在动态和敌对环境中部署恶意软件检测器引发了对其稳健性的严重担忧。虽然最近的方法使用对抗性攻击来评估基于FCG的检测器的稳健性，但其有效性受到巨大扰动空间的限制，特别是在不同的模型和特征中。   为了应对这些挑战，我们引入了FCGHunter，这是一个针对基于FCG的AMD系统的新型稳健性测试框架。具体来说，FCGHunter采用创新技术来加强这个巨大搜索空间中的探索和利用。最初，它会识别FCG中与恶意软件行为相关的关键区域，以缩小干扰空间。然后，我们开发了一种依赖性感知的交叉和突变方法，以增强扰动的有效性和多样性，生成不同的FCG。此外，FCGHunter利用多目标反馈来选择受干扰的FCG，通过基于解释的特征更改反馈显着改进搜索过程。   对40个场景的广泛评估表明，FCGHunter的平均攻击成功率为87.9%，明显优于基线至少44.7%。值得注意的是，FCGHunter在稳健模型上实现了100%的成功率（例如，AdaBoost with MalScan），基线仅达到11%或不适用。



## **20. Mitigating Evasion Attacks in Federated Learning-Based Signal Classifiers**

缓解基于联邦学习的信号分类器中的逃避攻击 eess.SP

Accepted for publication in IEEE Transactions on Network Science and  Engineering. arXiv admin note: substantial text overlap with arXiv:2301.08866

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2306.04872v2) [paper-pdf](http://arxiv.org/pdf/2306.04872v2)

**Authors**: Su Wang, Rajeev Sahay, Adam Piaseczny, Christopher G. Brinton

**Abstract**: Recent interest in leveraging federated learning (FL) for radio signal classification (SC) tasks has shown promise but FL-based SC remains susceptible to model poisoning adversarial attacks. These adversarial attacks mislead the ML model training process, damaging ML models across the network and leading to lower SC performance. In this work, we seek to mitigate model poisoning adversarial attacks on FL-based SC by proposing the Underlying Server Defense of Federated Learning (USD-FL). Unlike existing server-driven defenses, USD-FL does not rely on perfect network information, i.e., knowing the quantity of adversaries, the adversarial attack architecture, or the start time of the adversarial attacks. Our proposed USD-FL methodology consists of deriving logits for devices' ML models on a reserve dataset, comparing pair-wise logits via 1-Wasserstein distance and then determining a time-varying threshold for adversarial detection. As a result, USD-FL effectively mitigates model poisoning attacks introduced in the FL network. Specifically, when baseline server-driven defenses do have perfect network information, USD-FL outperforms them by (i) improving final ML classification accuracies by at least 6%, (ii) reducing false positive adversary detection rates by at least 10%, and (iii) decreasing the total number of misclassified signals by over 8%. Moreover, when baseline defenses do not have perfect network information, we show that USD-FL achieves accuracies of approximately 74.1% and 62.5% in i.i.d. and non-i.i.d. settings, outperforming existing server-driven baselines, which achieve 52.1% and 39.2% in i.i.d. and non-i.i.d. settings, respectively.

摘要: 最近人们对利用联邦学习（FL）进行无线电信号分类（SC）任务的兴趣已经显示出希望，但基于FL的SC仍然容易受到模型中毒对抗攻击的影响。这些对抗性攻击误导了ML模型训练过程，损害了整个网络的ML模型，并导致SC性能下降。在这项工作中，我们试图通过提出联邦学习的底层服务器防御（USD-FL）来减轻对基于FL的SC的模型中毒对抗攻击。与现有的服务器驱动防御不同，USD-FL不依赖于完美的网络信息，即了解对手的数量、对抗性攻击架构或对抗性攻击的开始时间。我们提出的USD-FL方法包括在储备数据集中推导设备ML模型的logit，通过1-Wasserstein距离比较成对logit，然后确定对抗检测的时变阈值。因此，USD-FL有效缓解了FL网络中引入的模型中毒攻击。具体来说，当基线服务器驱动的防御确实具有完美的网络信息时，USD-FL通过以下方式优于它们：（i）将最终ML分类准确性提高至少6%，（ii）将误报对手检测率降低至少10%，以及（iii）将错误分类的信号总数减少超过8%。此外，当基线防御没有完美的网络信息时，我们表明USD-FL在i. i. d方面的准确性约为74.1%和62.5%。和非i.i.d.设置的表现优于现有的服务器驱动基准，后者的i. i. d分别达到52.1%和39.2%和非i.i.d.设置，分别。



## **21. Forging and Removing Latent-Noise Diffusion Watermarks Using a Single Image**

基于单幅图像的隐噪声扩散水印的伪造与去除 cs.CV

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2504.20111v1) [paper-pdf](http://arxiv.org/pdf/2504.20111v1)

**Authors**: Anubhav Jain, Yuya Kobayashi, Naoki Murata, Yuhta Takida, Takashi Shibuya, Yuki Mitsufuji, Niv Cohen, Nasir Memon, Julian Togelius

**Abstract**: Watermarking techniques are vital for protecting intellectual property and preventing fraudulent use of media. Most previous watermarking schemes designed for diffusion models embed a secret key in the initial noise. The resulting pattern is often considered hard to remove and forge into unrelated images. In this paper, we propose a black-box adversarial attack without presuming access to the diffusion model weights. Our attack uses only a single watermarked example and is based on a simple observation: there is a many-to-one mapping between images and initial noises. There are regions in the clean image latent space pertaining to each watermark that get mapped to the same initial noise when inverted. Based on this intuition, we propose an adversarial attack to forge the watermark by introducing perturbations to the images such that we can enter the region of watermarked images. We show that we can also apply a similar approach for watermark removal by learning perturbations to exit this region. We report results on multiple watermarking schemes (Tree-Ring, RingID, WIND, and Gaussian Shading) across two diffusion models (SDv1.4 and SDv2.0). Our results demonstrate the effectiveness of the attack and expose vulnerabilities in the watermarking methods, motivating future research on improving them.

摘要: 水印技术对于保护知识产权和防止媒体欺诈性使用至关重要。大多数以前为扩散模型设计的水印方案都在初始噪音中嵌入秘密密钥。生成的图案通常被认为很难删除并伪造成不相关的图像。在本文中，我们提出了一种黑匣子对抗攻击，而不假设访问扩散模型权重。我们的攻击仅使用一个带水印的示例，并且基于一个简单的观察：图像和初始噪音之间存在多对一的映射。干净图像潜空间中存在与每个水印相关的区域，这些区域在倒置时映射到相同的初始噪音。基于这一直觉，我们提出了一种对抗攻击来通过向图像引入扰动来伪造水印，以便我们可以进入带有水印的图像的区域。我们表明，我们还可以通过学习扰动退出该区域来应用类似的方法来去除水印。我们报告了跨两个扩散模型（SDv1.4和SDv2.0）的多种水印方案（Tree-Ring、RingID、WIND和高斯着色）的结果。我们的结果证明了攻击的有效性，并暴露了水印方法中的漏洞，激励了未来改进它们的研究。



## **22. CapsFake: A Multimodal Capsule Network for Detecting Instruction-Guided Deepfakes**

CapsFake：用于检测指令引导Deepfake的多模式胶囊网络 cs.CV

20 pages

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2504.19212v1) [paper-pdf](http://arxiv.org/pdf/2504.19212v1)

**Authors**: Tuan Nguyen, Naseem Khan, Issa Khalil

**Abstract**: The rapid evolution of deepfake technology, particularly in instruction-guided image editing, threatens the integrity of digital images by enabling subtle, context-aware manipulations. Generated conditionally from real images and textual prompts, these edits are often imperceptible to both humans and existing detection systems, revealing significant limitations in current defenses. We propose a novel multimodal capsule network, CapsFake, designed to detect such deepfake image edits by integrating low-level capsules from visual, textual, and frequency-domain modalities. High-level capsules, predicted through a competitive routing mechanism, dynamically aggregate local features to identify manipulated regions with precision. Evaluated on diverse datasets, including MagicBrush, Unsplash Edits, Open Images Edits, and Multi-turn Edits, CapsFake outperforms state-of-the-art methods by up to 20% in detection accuracy. Ablation studies validate its robustness, achieving detection rates above 94% under natural perturbations and 96% against adversarial attacks, with excellent generalization to unseen editing scenarios. This approach establishes a powerful framework for countering sophisticated image manipulations.

摘要: Deepfake技术的快速发展，特别是在描述引导图像编辑方面，通过实现微妙的、上下文感知的操纵，威胁到数字图像的完整性。这些编辑是根据真实图像和文本提示有条件地生成的，人类和现有的检测系统通常都无法察觉，从而揭示了当前防御的显着局限性。我们提出了一种新型的多模式胶囊网络CapsFake，旨在通过集成来自视觉、文本和频域模式的低级胶囊来检测此类深度伪造图像编辑。通过竞争路由机制预测的高级胶囊动态聚合局部特征，以精确地识别操纵区域。经过多种数据集的评估，包括MagicBrush、Unsplash Edits、Open Images Edits和Multi-turn Edits，CapsFake的检测准确率比最先进的方法高出20%。消融研究验证了其稳健性，在自然扰动下检测率超过94%，在对抗性攻击下检测率达到96%，并且对不可见的编辑场景具有出色的概括性。这种方法建立了一个强大的框架来对抗复杂的图像操纵。



## **23. Support is All You Need for Certified VAE Training**

支持就是您认证VAE培训所需的一切 cs.LG

21 pages, 3 figures, ICLR '25

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2504.11831v2) [paper-pdf](http://arxiv.org/pdf/2504.11831v2)

**Authors**: Changming Xu, Debangshu Banerjee, Deepak Vasisht, Gagandeep Singh

**Abstract**: Variational Autoencoders (VAEs) have become increasingly popular and deployed in safety-critical applications. In such applications, we want to give certified probabilistic guarantees on performance under adversarial attacks. We propose a novel method, CIVET, for certified training of VAEs. CIVET depends on the key insight that we can bound worst-case VAE error by bounding the error on carefully chosen support sets at the latent layer. We show this point mathematically and present a novel training algorithm utilizing this insight. We show in an extensive evaluation across different datasets (in both the wireless and vision application areas), architectures, and perturbation magnitudes that our method outperforms SOTA methods achieving good standard performance with strong robustness guarantees.

摘要: 变分自动编码器（VAE）变得越来越受欢迎，并部署在安全关键应用中。在此类应用程序中，我们希望为对抗性攻击下的性能提供经过认证的概率保证。我们提出了一种新的方法CIVET，用于VAE的认证培训。CIVET取决于一个关键见解，即我们可以通过将误差限制在潜在层精心选择的支持集上来限制最坏情况的VAE误差。我们以数学方式展示了这一点，并利用这一见解提出了一种新颖的训练算法。我们在对不同数据集（无线和视觉应用领域）、架构和扰动幅度的广泛评估中表明，我们的方法优于SOTA方法，可以在强大的稳健性保证下实现良好的标准性能。



## **24. Graph of Attacks: Improved Black-Box and Interpretable Jailbreaks for LLMs**

攻击图表：LLM的改进黑匣子和可解释越狱 cs.CL

19 pages, 1 figure, 6 tables

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.19019v1) [paper-pdf](http://arxiv.org/pdf/2504.19019v1)

**Authors**: Mohammad Akbar-Tajari, Mohammad Taher Pilehvar, Mohammad Mahmoody

**Abstract**: The challenge of ensuring Large Language Models (LLMs) align with societal standards is of increasing interest, as these models are still prone to adversarial jailbreaks that bypass their safety mechanisms. Identifying these vulnerabilities is crucial for enhancing the robustness of LLMs against such exploits. We propose Graph of ATtacks (GoAT), a method for generating adversarial prompts to test the robustness of LLM alignment using the Graph of Thoughts framework [Besta et al., 2024]. GoAT excels at generating highly effective jailbreak prompts with fewer queries to the victim model than state-of-the-art attacks, achieving up to five times better jailbreak success rate against robust models like Llama. Notably, GoAT creates high-quality, human-readable prompts without requiring access to the targeted model's parameters, making it a black-box attack. Unlike approaches constrained by tree-based reasoning, GoAT's reasoning is based on a more intricate graph structure. By making simultaneous attack paths aware of each other's progress, this dynamic framework allows a deeper integration and refinement of reasoning paths, significantly enhancing the collaborative exploration of adversarial vulnerabilities in LLMs. At a technical level, GoAT starts with a graph structure and iteratively refines it by combining and improving thoughts, enabling synergy between different thought paths. The code for our implementation can be found at: https://github.com/GoAT-pydev/Graph_of_Attacks.

摘要: 确保大型语言模型（LLM）与社会标准保持一致的挑战越来越受到关注，因为这些模型仍然容易出现绕过其安全机制的对抗性越狱。识别这些漏洞对于增强LLM针对此类漏洞的稳健性至关重要。我们提出了攻击图形（GoAT），这是一种用于生成对抗提示的方法，以使用思想图形框架测试LLM对齐的稳健性[Besta等人，2024]。GoAT擅长生成高效的越狱提示，对受害者模型的查询比最先进的攻击更少，针对Llama等稳健模型，越狱成功率高出五倍。值得注意的是，GoAT可以创建高质量、人类可读的提示，而不需要访问目标模型的参数，使其成为黑匣子攻击。与受基于树的推理约束的方法不同，GoAT的推理基于更复杂的图结构。通过让同时攻击路径了解彼此的进展，这个动态框架允许更深入地集成和细化推理路径，显着增强了对LLM对抗漏洞的协作探索。在技术层面，GoAT从图形结构开始，通过组合和改进思想来迭代细化它，实现不同思维路径之间的协同。我们的实现代码可在：https://github.com/GoAT-pydev/Graph_of_Attacks上找到。



## **25. Unveiling and Mitigating Adversarial Vulnerabilities in Iterative Optimizers**

揭露和缓解迭代优化器中的对抗漏洞 cs.LG

Under review for publication in the IEEE

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.19000v1) [paper-pdf](http://arxiv.org/pdf/2504.19000v1)

**Authors**: Elad Sofer, Tomer Shaked, Caroline Chaux, Nir Shlezinger

**Abstract**: Machine learning (ML) models are often sensitive to carefully crafted yet seemingly unnoticeable perturbations. Such adversarial examples are considered to be a property of ML models, often associated with their black-box operation and sensitivity to features learned from data. This work examines the adversarial sensitivity of non-learned decision rules, and particularly of iterative optimizers. Our analysis is inspired by the recent developments in deep unfolding, which cast such optimizers as ML models. We show that non-learned iterative optimizers share the sensitivity to adversarial examples of ML models, and that attacking iterative optimizers effectively alters the optimization objective surface in a manner that modifies the minima sought. We then leverage the ability to cast iteration-limited optimizers as ML models to enhance robustness via adversarial training. For a class of proximal gradient optimizers, we rigorously prove how their learning affects adversarial sensitivity. We numerically back our findings, showing the vulnerability of various optimizers, as well as the robustness induced by unfolding and adversarial training.

摘要: 机器学习（ML）模型通常对精心设计但看似不明显的扰动敏感。此类对抗性示例被认为是ML模型的一个属性，通常与它们的黑匣子操作和对从数据中学习到的特征的敏感性相关。这项工作考察了非学习决策规则，特别是迭代优化器的对抗敏感性。我们的分析受到深度展开领域的最新发展的启发，这些发展将此类优化器投射为ML模型。我们表明，非学习迭代优化器对ML模型的对抗性示例都有敏感性，并且攻击迭代优化器可以以修改所寻求的最小值的方式有效地改变优化目标表面。然后，我们利用将迭代限制优化器投射为ML模型的能力，通过对抗训练增强稳健性。对于一类近端梯度优化器，我们严格证明了他们的学习如何影响对抗敏感性。我们从数字上支持了我们的发现，展示了各种优化器的脆弱性，以及展开和对抗训练引起的鲁棒性。



## **26. Federated Learning-based Semantic Segmentation for Lane and Object Detection in Autonomous Driving**

基于联邦学习的自动驾驶车道和物体检测语义分割 eess.SY

This paper has been accepted for publication in Scientific Reports

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.18939v1) [paper-pdf](http://arxiv.org/pdf/2504.18939v1)

**Authors**: Gharbi Khamis Alshammari, Ahmad Abubakar, Nada M. O. Sid Ahmed, Naif Khalaf Alshammari

**Abstract**: Autonomous Vehicles (AVs) require precise lane and object detection to ensure safe navigation. However, centralized deep learning (DL) approaches for semantic segmentation raise privacy and scalability challenges, particularly when handling sensitive data. This research presents a new federated learning (FL) framework that integrates secure deep Convolutional Neural Networks (CNNs) and Differential Privacy (DP) to address these issues. The core contribution of this work involves: (1) developing a new hybrid UNet-ResNet34 architecture for centralized semantic segmentation to achieve high accuracy and tackle privacy concerns due to centralized training, and (2) implementing the privacy-preserving FL model, distributed across AVs to enhance performance through secure CNNs and DP mechanisms. In the proposed FL framework, the methodology distinguishes itself from the existing approach through the following: (a) ensuring data decentralization through FL to uphold user privacy by eliminating the need for centralized data aggregation, (b) integrating DP mechanisms to secure sensitive model updates against potential adversarial inference attacks, and (c) evaluating the frameworks performance and generalizability using RGB and semantic segmentation datasets derived from the CARLA simulator. Experimental results show significant improvements in accuracy, from 81.5% to 88.7% for the RGB dataset and from 79.3% to 86.9% for the SEG dataset over 20 to 70 Communication Rounds (CRs). Global loss was reduced by over 60%, and minor accuracy trade-offs from DP were observed. This study contributes by offering a scalable, privacy-preserving FL framework tailored for AVs, optimizing communication efficiency while balancing performance and data security.

摘要: 自动驾驶汽车（AV）需要精确的车道和物体检测以确保安全导航。然而，用于语义分割的集中式深度学习（DL）方法会带来隐私和可扩展性挑战，特别是在处理敏感数据时。这项研究提出了一种新的联邦学习（FL）框架，该框架集成了安全的深度卷积神经网络（CNN）和差异隐私（DP）来解决这些问题。这项工作的核心贡献包括：（1）开发一种新的混合UNet-ResNet 34架构，用于集中式语义分割，以实现高准确性并解决由于集中式训练而产生的隐私问题，以及（2）实施保护隐私的FL模型，分布在AV中，以通过安全的CNN和DP机制增强性能。在拟议的FL框架中，该方法通过以下几点与现有方法区分开来：（a）确保通过FL实现数据去中心化，通过消除集中式数据聚合的需要来维护用户隐私，（b）集成DP机制以保护敏感模型更新免受潜在的对抗性推断攻击，以及（c）使用从CARLA模拟器中获得的Ruby和语义分割数据集来评估框架的性能和可概括性。实验结果显示，在20至70个通信轮（CR）中，准确性显着提高，从81.5%提高到88.7%，从79.3%提高到86.9%。全局损失减少了60%以上，并且观察到DP的微小精度折衷。这项研究通过提供一个可扩展的，隐私保护的FL框架为AV量身定制，优化通信效率，同时平衡性能和数据安全性。



## **27. Latent Adversarial Training Improves the Representation of Refusal**

隐性对抗训练改善了拒绝的表现 cs.CL

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.18872v1) [paper-pdf](http://arxiv.org/pdf/2504.18872v1)

**Authors**: Alexandra Abbas, Nora Petrova, Helios Ael Lyons, Natalia Perez-Campanero

**Abstract**: Recent work has shown that language models' refusal behavior is primarily encoded in a single direction in their latent space, making it vulnerable to targeted attacks. Although Latent Adversarial Training (LAT) attempts to improve robustness by introducing noise during training, a key question remains: How does this noise-based training affect the underlying representation of refusal behavior? Understanding this encoding is crucial for evaluating LAT's effectiveness and limitations, just as the discovery of linear refusal directions revealed vulnerabilities in traditional supervised safety fine-tuning (SSFT).   Through the analysis of Llama 2 7B, we examine how LAT reorganizes the refusal behavior in the model's latent space compared to SSFT and embedding space adversarial training (AT). By computing activation differences between harmful and harmless instruction pairs and applying Singular Value Decomposition (SVD), we find that LAT significantly alters the refusal representation, concentrating it in the first two SVD components which explain approximately 75 percent of the activation differences variance - significantly higher than in reference models. This concentrated representation leads to more effective and transferable refusal vectors for ablation attacks: LAT models show improved robustness when attacked with vectors from reference models but become more vulnerable to self-generated vectors compared to SSFT and AT. Our findings suggest that LAT's training perturbations enable a more comprehensive representation of refusal behavior, highlighting both its potential strengths and vulnerabilities for improving model safety.

摘要: 最近的工作表明，语言模型的拒绝行为主要在其潜在空间中编码在单一方向上，使其容易受到有针对性的攻击。尽管潜在对抗训练（LAT）试图通过在训练期间引入噪音来提高稳健性，但一个关键问题仍然存在：这种基于噪音的训练如何影响拒绝行为的基本表示？了解这种编码对于评估LAT的有效性和局限性至关重要，就像线性拒绝方向的发现揭示了传统监督安全微调（SSFT）中的漏洞一样。   通过对Llama 2 7 B的分析，我们研究了与SSFT和嵌入空间对抗训练（AT）相比，LAT如何在模型的潜在空间中重组拒绝行为。通过计算有害和无害指令对之间的激活差异并应用奇异值分解（奇异值分解），我们发现LAT显着改变了拒绝表示，将其集中在前两个奇异值分解部分中，这解释了大约75%的激活差异方差-显着高于参考模型。这种集中的表示为消融攻击带来了更有效、更可转移的拒绝载体：当用来自参考模型的载体进行攻击时，LAT模型显示出更好的鲁棒性，但与SSFT和AT相比，更容易受到自生成载体的影响。我们的研究结果表明，LAT的训练扰动能够更全面地表示拒绝行为，凸显其在提高模型安全性方面的潜在优势和弱点。



## **28. SynFuzz: Leveraging Fuzzing of Netlist to Detect Synthesis Bugs**

SynFuzz：利用网表的模糊化来检测合成错误 cs.CR

14 pages, 10 figures, 4 tables

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.18812v1) [paper-pdf](http://arxiv.org/pdf/2504.18812v1)

**Authors**: Raghul Saravanan, Sudipta Paria, Aritra Dasgupta, Venkat Nitin Patnala, Swarup Bhunia, Sai Manoj P D

**Abstract**: In the evolving landscape of integrated circuit (IC) design, the increasing complexity of modern processors and intellectual property (IP) cores has introduced new challenges in ensuring design correctness and security. The recent advancements in hardware fuzzing techniques have shown their efficacy in detecting hardware bugs and vulnerabilities at the RTL abstraction level of hardware. However, they suffer from several limitations, including an inability to address vulnerabilities introduced during synthesis and gate-level transformations. These methods often fail to detect issues arising from library adversaries, where compromised or malicious library components can introduce backdoors or unintended behaviors into the design. In this paper, we present a novel hardware fuzzer, SynFuzz, designed to overcome the limitations of existing hardware fuzzing frameworks. SynFuzz focuses on fuzzing hardware at the gate-level netlist to identify synthesis bugs and vulnerabilities that arise during the transition from RTL to the gate-level. We analyze the intrinsic hardware behaviors using coverage metrics specifically tailored for the gate-level. Furthermore, SynFuzz implements differential fuzzing to uncover bugs associated with EDA libraries. We evaluated SynFuzz on popular open-source processors and IP designs, successfully identifying 7 new synthesis bugs. Additionally, by exploiting the optimization settings of EDA tools, we performed a compromised library mapping attack (CLiMA), creating a malicious version of hardware designs that remains undetectable by traditional verification methods. We also demonstrate how SynFuzz overcomes the limitations of the industry-standard formal verification tool, Cadence Conformal, providing a more robust and comprehensive approach to hardware verification.

摘要: 在集成电路（IC）设计不断发展的格局中，现代处理器和知识产权（IP）核的复杂性日益增加，为确保设计正确性和安全性带来了新的挑战。硬件模糊技术的最新进展表明，它们在硬件RTL抽象级别检测硬件错误和漏洞方面的功效。然而，它们存在一些限制，包括无法解决合成和门级转换期间引入的漏洞。这些方法通常无法检测到库对手引起的问题，其中受损害或恶意的库组件可能会在设计中引入后门或意外行为。在本文中，我们提出了一种新型的硬件模糊器SynFuzz，旨在克服现有硬件模糊框架的局限性。SynFuzz专注于在门级网表上模糊硬件，以识别从RTL过渡到门级过程中出现的合成错误和漏洞。我们使用专门为门户级定制的覆盖指标来分析固有的硬件行为。此外，SynFuzz还实现了差异模糊化来发现与EDA库相关的错误。我们在流行的开源处理器和IP设计上评估了SynFuzz，成功识别出7个新的合成错误。此外，通过利用EDA工具的优化设置，我们执行了受损库映射攻击（CLiMA），创建了传统验证方法仍然无法检测到的硬件设计的恶意版本。我们还展示了SynFuzz如何克服行业标准形式验证工具Cadence Conformal的局限性，为硬件验证提供更稳健、更全面的方法。



## **29. Intelligent Attacks and Defense Methods in Federated Learning-enabled Energy-Efficient Wireless Networks**

支持联邦学习的节能无线网络中的智能攻击和防御方法 cs.LG

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18519v1) [paper-pdf](http://arxiv.org/pdf/2504.18519v1)

**Authors**: Han Zhang, Hao Zhou, Medhat Elsayed, Majid Bavand, Raimundas Gaigalas, Yigit Ozcan, Melike Erol-Kantarci

**Abstract**: Federated learning (FL) is a promising technique for learning-based functions in wireless networks, thanks to its distributed implementation capability. On the other hand, distributed learning may increase the risk of exposure to malicious attacks where attacks on a local model may spread to other models by parameter exchange. Meanwhile, such attacks can be hard to detect due to the dynamic wireless environment, especially considering local models can be heterogeneous with non-independent and identically distributed (non-IID) data. Therefore, it is critical to evaluate the effect of malicious attacks and develop advanced defense techniques for FL-enabled wireless networks. In this work, we introduce a federated deep reinforcement learning-based cell sleep control scenario that enhances the energy efficiency of the network. We propose multiple intelligent attacks targeting the learning-based approach and we propose defense methods to mitigate such attacks. In particular, we have designed two attack models, generative adversarial network (GAN)-enhanced model poisoning attack and regularization-based model poisoning attack. As a counteraction, we have proposed two defense schemes, autoencoder-based defense, and knowledge distillation (KD)-enabled defense. The autoencoder-based defense method leverages an autoencoder to identify the malicious participants and only aggregate the parameters of benign local models during the global aggregation, while KD-based defense protects the model from attacks by controlling the knowledge transferred between the global model and local models.

摘要: 联邦学习（FL）是一种很有前途的技术，在无线网络中的学习为基础的功能，由于其分布式实现能力。另一方面，分布式学习可能会增加遭受恶意攻击的风险，其中对本地模型的攻击可能会通过参数交换传播到其他模型。同时，由于动态无线环境，这种攻击可能难以检测，特别是考虑到本地模型可能是异构的，具有非独立同分布（非IID）数据。因此，评估恶意攻击的影响并开发先进的防御技术对于FL使能的无线网络至关重要。在这项工作中，我们引入了一种基于联合深度强化学习的单元睡眠控制场景，该场景可以提高网络的能源效率。我们提出了针对基于学习的方法的多种智能攻击，并提出了减轻此类攻击的防御方法。特别是，我们设计了两种攻击模型：生成对抗网络（GAN）增强模型中毒攻击和基于正规化的模型中毒攻击。作为反击，我们提出了两种防御方案，基于自动编码器的防御和基于知识蒸馏（KD）的防御。基于自动编码器的防御方法利用自动编码器来识别恶意参与者，并在全局聚合期间仅聚合良性本地模型的参数，而基于KD的防御通过控制全局模型和本地模型之间传输的知识来保护模型免受攻击。



## **30. Distributed Multiple Testing with False Discovery Rate Control in the Presence of Byzantines**

在拜占庭存在的情况下具有错误发现率控制的分布式多重测试 eess.SP

Accepted to the 2025 International Symposium on Information Theory  (ISIT)

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2501.13242v2) [paper-pdf](http://arxiv.org/pdf/2501.13242v2)

**Authors**: Daofu Zhang, Mehrdad Pournaderi, Yu Xiang, Pramod Varshney

**Abstract**: This work studies distributed multiple testing with false discovery rate (FDR) control in the presence of Byzantine attacks, where an adversary captures a fraction of the nodes and corrupts their reported p-values. We focus on two baseline attack models: an oracle model with the full knowledge of which hypotheses are true nulls, and a practical attack model that leverages the Benjamini-Hochberg (BH) procedure locally to classify which p-values follow the true null hypotheses. We provide a thorough characterization of how both attack models affect the global FDR, which in turn motivates counter-attack strategies and stronger attack models. Our extensive simulation studies confirm the theoretical results, highlight key design trade-offs under attacks and countermeasures, and provide insights into more sophisticated attacks.

摘要: 这项工作研究了在存在拜占庭攻击的情况下具有错误发现率（HDR）控制的分布式多重测试，其中对手捕获了一小部分节点并破坏了它们报告的p值。我们重点关注两种基线攻击模型：一种是完全了解哪些假设是真空的Oracle模型，另一种是本地利用Benjamini-Hochberg（BH）过程对哪些p值遵循真空假设进行分类的实用攻击模型。我们彻底描述了这两种攻击模型如何影响全球HDR，这反过来又激励了反击策略和更强的攻击模型。我们广泛的模拟研究证实了理论结果，强调了攻击和对策下的关键设计权衡，并提供了对更复杂攻击的见解。



## **31. Adversarial Attacks on LLM-as-a-Judge Systems: Insights from Prompt Injections**

对LLM-as-a-Judge系统的对抗性攻击：来自即时注入的见解 cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18333v1) [paper-pdf](http://arxiv.org/pdf/2504.18333v1)

**Authors**: Narek Maloyan, Dmitry Namiot

**Abstract**: LLM as judge systems used to assess text quality code correctness and argument strength are vulnerable to prompt injection attacks. We introduce a framework that separates content author attacks from system prompt attacks and evaluate five models Gemma 3.27B Gemma 3.4B Llama 3.2 3B GPT 4 and Claude 3 Opus on four tasks with various defenses using fifty prompts per condition. Attacks achieved up to seventy three point eight percent success smaller models proved more vulnerable and transferability ranged from fifty point five to sixty two point six percent. Our results contrast with Universal Prompt Injection and AdvPrompter We recommend multi model committees and comparative scoring and release all code and datasets

摘要: LLM作为用于评估文本质量、代码正确性和论点强度的判断系统，很容易受到即时注入攻击。我们引入了一个将内容作者攻击与系统提示攻击分开的框架，并评估了五个模型Gemma 3.27B Gemma 3.4B Llama 3.2 3B GPT 4和Claude 3 Opus，用于四个具有各种防御的任务，每个条件使用50个提示。攻击的成功率高达73.8%，事实证明，较小的模型更容易受到攻击，可转移性从5.5%到62.6%不等。我们的结果与通用提示注入和Advancer形成鲜明对比我们建议多模型委员会和比较评分并发布所有代码和数据集



## **32. Contrastive Learning and Adversarial Disentanglement for Task-Oriented Semantic Communications**

面向任务的语义通信的对比学习和对抗解纠缠 cs.LG

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2410.22784v2) [paper-pdf](http://arxiv.org/pdf/2410.22784v2)

**Authors**: Omar Erak, Omar Alhussein, Wen Tong

**Abstract**: Task-oriented semantic communication systems have emerged as a promising approach to achieving efficient and intelligent data transmission, where only information relevant to a specific task is communicated. However, existing methods struggle to fully disentangle task-relevant and task-irrelevant information, leading to privacy concerns and subpar performance. To address this, we propose an information-bottleneck method, named CLAD (contrastive learning and adversarial disentanglement). CLAD utilizes contrastive learning to effectively capture task-relevant features while employing adversarial disentanglement to discard task-irrelevant information. Additionally, due to the lack of reliable and reproducible methods to gain insight into the informativeness and minimality of the encoded feature vectors, we introduce a new technique to compute the information retention index (IRI), a comparative metric used as a proxy for the mutual information between the encoded features and the input, reflecting the minimality of the encoded features. The IRI quantifies the minimality and informativeness of the encoded feature vectors across different task-oriented communication techniques. Our extensive experiments demonstrate that CLAD outperforms state-of-the-art baselines in terms of semantic extraction, task performance, privacy preservation, and IRI. CLAD achieves a predictive performance improvement of around 2.5-3%, along with a 77-90% reduction in IRI and a 57-76% decrease in adversarial attribute inference attack accuracy.

摘要: 面向任务的语义通信系统已成为实现高效和智能数据传输的一种有希望的方法，其中仅通信与特定任务相关的信息。然而，现有的方法很难完全区分与任务相关和与任务无关的信息，从而导致隐私问题和性能不佳。为了解决这个问题，我们提出了一种信息瓶颈方法，名为CLAD（对比学习和对抗解纠缠）。CLAD利用对比学习来有效地捕获任务相关特征，同时利用对抗解纠缠来丢弃任务无关的信息。此外，由于缺乏可靠且可重复的方法来深入了解编码特征载体的信息性和最小性，我们引入了一种新技术来计算信息保留指数（IRI），这是一种比较指标，用作编码特征和输入之间的互信息的代理，反映了编码特征的最小性。IRI量化了不同面向任务的通信技术中编码特征载体的最小性和信息性。我们广泛的实验表明，CLAD在语义提取、任务性能、隐私保护和IRI方面优于最先进的基线。CLAD实现了约2.5- 3%的预测性能改进，IRI降低了77-90%，对抗属性推断攻击准确性降低了57-76%。



## **33. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

通过跨模式提示注射操纵多模式代理 cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.14348v3) [paper-pdf](http://arxiv.org/pdf/2504.14348v3)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.

摘要: 多模式大型语言模型的出现通过将语言和视觉模式与外部数据源集成来重新定义了代理范式，使代理能够更好地解释人类指令并执行日益复杂的任务。然而，在这项工作中，我们发现了多模式代理中一个以前被忽视的关键安全漏洞：跨模式提示注入攻击。为了利用这个漏洞，我们提出了CrossInib，这是一种新型攻击框架，其中攻击者在多种模式中嵌入对抗性扰动，以与目标恶意内容保持一致，允许外部指令劫持代理的决策过程并执行未经授权的任务。我们的方法由两个关键部分组成。首先，我们引入了视觉潜在对齐，基于文本到图像生成模型，优化视觉嵌入空间中恶意指令的对抗特征，确保对抗图像巧妙地编码恶意任务执行的线索。随后，我们提出了文本指导增强，其中利用大型语言模型通过对抗性Meta提示来推断黑匣子防御系统提示，并生成恶意文本命令，该命令引导代理的输出更好地遵守攻击者的请求。大量实验表明，我们的方法优于现有的注入攻击，在不同任务中的攻击成功率至少增加了+26.4%。此外，我们还验证了攻击在现实世界的多模式自治代理中的有效性，强调了其对安全关键应用程序的潜在影响。



## **34. Robust Kernel Hypothesis Testing under Data Corruption**

数据腐败下的鲁棒核假设测试 stat.ML

22 pages, 2 figures, 2 algorithms

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2405.19912v3) [paper-pdf](http://arxiv.org/pdf/2405.19912v3)

**Authors**: Antonin Schrab, Ilmun Kim

**Abstract**: We propose a general method for constructing robust permutation tests under data corruption. The proposed tests effectively control the non-asymptotic type I error under data corruption, and we prove their consistency in power under minimal conditions. This contributes to the practical deployment of hypothesis tests for real-world applications with potential adversarial attacks. For the two-sample and independence settings, we show that our kernel robust tests are minimax optimal, in the sense that they are guaranteed to be non-asymptotically powerful against alternatives uniformly separated from the null in the kernel MMD and HSIC metrics at some optimal rate (tight with matching lower bound). We point out that existing differentially private tests can be adapted to be robust to data corruption, and we demonstrate in experiments that our proposed tests achieve much higher power than these private tests. Finally, we provide publicly available implementations and empirically illustrate the practicality of our robust tests.

摘要: 我们提出了一种在数据损坏情况下构造鲁棒排列测试的通用方法。提出的测试有效地控制了数据损坏下的非渐进I型错误，并且我们证明了它们在最低条件下的功效一致性。这有助于对具有潜在对抗攻击的现实世界应用程序进行假设测试的实际部署。对于双样本和独立性设置，我们表明我们的内核鲁棒性测试是极小极大最优的，从某种意义上说，它们保证对以某个最优速率均匀分离于内核MMD和HSIC指标中的零值的替代方案具有非渐进的强大性（与匹配的下限紧密）。我们指出，现有的差异私密测试可以被调整为对数据损坏具有鲁棒性，并且我们在实验中证明，我们提出的测试比这些私密测试具有更高的能力。最后，我们提供了公开的实现，并以经验方式说明了我们稳健测试的实用性。



## **35. Generative AI for Physical-Layer Authentication**

用于物理层身份验证的生成人工智能 eess.SP

10 pages, 3 figures

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18175v1) [paper-pdf](http://arxiv.org/pdf/2504.18175v1)

**Authors**: Rui Meng, Xiqi Cheng, Song Gao, Xiaodong Xu, Chen Dong, Guoshun Nan, Xiaofeng Tao, Ping Zhang, Tony Q. S. Quek

**Abstract**: In recent years, Artificial Intelligence (AI)-driven Physical-Layer Authentication (PLA), which focuses on achieving endogenous security and intelligent identity authentication, has attracted considerable interest. When compared with Discriminative AI (DAI), Generative AI (GAI) offers several advantages, such as fingerprint data augmentation, fingerprint denoising and reconstruction, and protection against adversarial attacks. Inspired by these innovations, this paper provides a systematic exploration of GAI's integration into PLA frameworks. We commence with a review of representative authentication techniques, emphasizing PLA's inherent strengths. Following this, we revisit four typical GAI models and contrast the limitations of DAI with the potential of GAI in addressing PLA challenges, including insufficient fingerprint data, environment noises and inferences, perturbations in fingerprint data, and complex tasks. Specifically, we delve into providing GAI-enhance methods for PLA across the data, model, and application layers in detail. Moreover, we present a case study that combines fingerprint extrapolation, generative diffusion models, and cooperative nodes to illustrate the superiority of GAI in bolstering the reliability of PLA compared to DAI. Additionally, we outline potential future research directions for GAI-based PLA.

摘要: 近年来，专注于实现内生安全和智能身份认证的人工智能（AI）驱动的物理层认证（PLA）引起了相当大的兴趣。与区分性人工智能（DAI）相比，生成性人工智能（GAI）提供了多个优势，例如指纹数据增强、指纹去噪和重建以及对抗攻击的保护。受这些创新的启发，本文对GAI融入解放军框架进行了系统探索。我们首先回顾代表性的认证技术，强调解放军的固有优势。随后，我们重新审视了四种典型的GAI模型，并将DAI的局限性与GAI在应对解放军挑战方面的潜力进行了比较，包括指纹数据不足、环境噪音和推断、指纹数据的扰动以及复杂的任务。具体来说，我们详细研究了跨数据、模型和应用层为PLA提供GAI增强方法。此外，我们还提出了一个结合指纹外推、生成扩散模型和合作节点的案例研究，以说明GAI在增强PLA可靠性方面与DAI相比的优势。此外，我们还概述了基于GAI的解放军未来潜在的研究方向。



## **36. Revisiting Locally Differentially Private Protocols: Towards Better Trade-offs in Privacy, Utility, and Attack Resistance**

重新审视本地差异私有协议：在隐私、实用性和抗攻击方面实现更好的权衡 cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2503.01482v2) [paper-pdf](http://arxiv.org/pdf/2503.01482v2)

**Authors**: Héber H. Arcolezi, Sébastien Gambs

**Abstract**: Local Differential Privacy (LDP) offers strong privacy protection, especially in settings in which the server collecting the data is untrusted. However, designing LDP mechanisms that achieve an optimal trade-off between privacy, utility and robustness to adversarial inference attacks remains challenging. In this work, we introduce a general multi-objective optimization framework for refining LDP protocols, enabling the joint optimization of privacy and utility under various adversarial settings. While our framework is flexible to accommodate multiple privacy and security attacks as well as utility metrics, in this paper, we specifically optimize for Attacker Success Rate (ASR) under \emph{data reconstruction attack} as a concrete measure of privacy leakage and Mean Squared Error (MSE) as a measure of utility. More precisely, we systematically revisit these trade-offs by analyzing eight state-of-the-art LDP protocols and proposing refined counterparts that leverage tailored optimization techniques. Experimental results demonstrate that our proposed adaptive mechanisms consistently outperform their non-adaptive counterparts, achieving substantial reductions in ASR while preserving utility, and pushing closer to the ASR-MSE Pareto frontier. By bridging the gap between theoretical guarantees and real-world vulnerabilities, our framework enables modular and context-aware deployment of LDP mechanisms with tunable privacy-utility trade-offs.

摘要: 本地差异隐私（SDP）提供强大的隐私保护，尤其是在收集数据的服务器不受信任的设置中。然而，设计在隐私性、效用和对抗性推理攻击的稳健性之间实现最佳权衡的LDP机制仍然具有挑战性。在这项工作中，我们引入了一个通用的多目标优化框架，用于完善LDP协议，从而能够在各种对抗环境下联合优化隐私和效用。虽然我们的框架可以灵活地适应多种隐私和安全攻击以及效用指标，但在本文中，我们专门优化了\{数据重建攻击}下的攻击者成功率（ASB）作为隐私泄露的具体指标，并优化均方误差（SSE）作为效用指标。更确切地说，我们系统地重新审视这些权衡，分析八个国家的最先进的LDP协议，并提出完善的同行，利用量身定制的优化技术。实验结果表明，我们提出的自适应机制始终优于其非自适应同行，实现大幅减少ASR，同时保持效用，并推动更接近ASR-MSE帕累托边界。通过弥合理论保证和现实世界的漏洞之间的差距，我们的框架，使模块化和上下文感知部署的LDP机制与可调的隐私效用权衡。



## **37. A Parametric Approach to Adversarial Augmentation for Cross-Domain Iris Presentation Attack Detection**

跨域虹膜呈现攻击检测的对抗增强参数方法 cs.CV

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV),  2025

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2412.07199v2) [paper-pdf](http://arxiv.org/pdf/2412.07199v2)

**Authors**: Debasmita Pal, Redwan Sony, Arun Ross

**Abstract**: Iris-based biometric systems are vulnerable to presentation attacks (PAs), where adversaries present physical artifacts (e.g., printed iris images, textured contact lenses) to defeat the system. This has led to the development of various presentation attack detection (PAD) algorithms, which typically perform well in intra-domain settings. However, they often struggle to generalize effectively in cross-domain scenarios, where training and testing employ different sensors, PA instruments, and datasets. In this work, we use adversarial training samples of both bonafide irides and PAs to improve the cross-domain performance of a PAD classifier. The novelty of our approach lies in leveraging transformation parameters from classical data augmentation schemes (e.g., translation, rotation) to generate adversarial samples. We achieve this through a convolutional autoencoder, ADV-GEN, that inputs original training samples along with a set of geometric and photometric transformations. The transformation parameters act as regularization variables, guiding ADV-GEN to generate adversarial samples in a constrained search space. Experiments conducted on the LivDet-Iris 2017 database, comprising four datasets, and the LivDet-Iris 2020 dataset, demonstrate the efficacy of our proposed method. The code is available at https://github.com/iPRoBe-lab/ADV-GEN-IrisPAD.

摘要: 基于虹膜的生物识别系统容易受到呈现攻击（PA），其中对手呈现物理伪影（例如，印刷虹膜图像、纹理隐形眼镜）来击败系统。这导致了各种表示攻击检测（PAD）算法的开发，这些算法通常在域内设置中表现良好。然而，他们通常很难在跨领域场景中进行有效的概括，其中训练和测试使用不同的传感器、PA仪器和数据集。在这项工作中，我们使用bondefide irides和PA的对抗训练样本来提高PAD分类器的跨域性能。我们方法的新颖之处在于利用经典数据增强方案中的转换参数（例如，翻译、旋转）以生成对抗样本。我们通过卷积自动编码器ADV-GER实现这一目标，该编码器输入原始训练样本以及一组几何和光感变换。转换参数充当正规化变量，指导ADV-GER在受约束的搜索空间中生成对抗样本。在由四个数据集组成的LivDet-Iris 2017数据库和LivDet-Iris 2020数据集上进行的实验证明了我们提出的方法的有效性。该代码可在https://github.com/iPRoBe-lab/ADV-GEN-IrisPAD上获取。



## **38. Edge-Based Learning for Improved Classification Under Adversarial Noise**

基于边缘的学习在对抗性噪音下改进分类 cs.CV

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.20077v1) [paper-pdf](http://arxiv.org/pdf/2504.20077v1)

**Authors**: Manish Kansana, Keyan Alexander Rahimi, Elias Hossain, Iman Dehzangi, Noorbakhsh Amiri Golilarz

**Abstract**: Adversarial noise introduces small perturbations in images, misleading deep learning models into misclassification and significantly impacting recognition accuracy. In this study, we analyzed the effects of Fast Gradient Sign Method (FGSM) adversarial noise on image classification and investigated whether training on specific image features can improve robustness. We hypothesize that while adversarial noise perturbs various regions of an image, edges may remain relatively stable and provide essential structural information for classification. To test this, we conducted a series of experiments using brain tumor and COVID datasets. Initially, we trained the models on clean images and then introduced subtle adversarial perturbations, which caused deep learning models to significantly misclassify the images. Retraining on a combination of clean and noisy images led to improved performance. To evaluate the robustness of the edge features, we extracted edges from the original/clean images and trained the models exclusively on edge-based representations. When noise was introduced to the images, the edge-based models demonstrated greater resilience to adversarial attacks compared to those trained on the original or clean images. These results suggest that while adversarial noise is able to exploit complex non-edge regions significantly more than edges, the improvement in the accuracy after retraining is marginally more in the original data as compared to the edges. Thus, leveraging edge-based learning can improve the resilience of deep learning models against adversarial perturbations.

摘要: 对抗性噪音在图像中引入了微小的扰动，误导深度学习模型进行错误分类，并显着影响识别准确性。在这项研究中，我们分析了快速梯度符号法（FGSM）对抗性噪音对图像分类的影响，并研究了对特定图像特征的训练是否可以提高鲁棒性。我们假设，虽然对抗性噪音扰乱图像的各个区域，但边缘可能保持相对稳定，并为分类提供必要的结构信息。为了测试这一点，我们使用脑肿瘤和COVID数据集进行了一系列实验。最初，我们在干净的图像上训练模型，然后引入微妙的对抗性扰动，这导致深度学习模型对图像进行严重错误分类。在干净和有噪图像的组合上重新训练可以提高性能。为了评估边缘特征的稳健性，我们从原始/干净图像中提取边缘，并专门根据基于边缘的表示来训练模型。当将噪音引入图像时，与在原始或干净图像上训练的模型相比，基于边缘的模型表现出更强的对抗攻击弹性。这些结果表明，虽然对抗性噪音能够比边缘更多地利用复杂的非边缘区域，但与边缘相比，重新训练后的准确性提高在原始数据中略高于边缘。因此，利用基于边缘的学习可以提高深度学习模型对对抗性扰动的弹性。



## **39. Diffusion-Driven Universal Model Inversion Attack for Face Recognition**

用于人脸识别的扩散驱动通用模型倒置攻击 cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18015v1) [paper-pdf](http://arxiv.org/pdf/2504.18015v1)

**Authors**: Hanrui Wang, Shuo Wang, Chun-Shien Lu, Isao Echizen

**Abstract**: Facial recognition technology poses significant privacy risks, as it relies on biometric data that is inherently sensitive and immutable if compromised. To mitigate these concerns, face recognition systems convert raw images into embeddings, traditionally considered privacy-preserving. However, model inversion attacks pose a significant privacy threat by reconstructing these private facial images, making them a crucial tool for evaluating the privacy risks of face recognition systems. Existing methods usually require training individual generators for each target model, a computationally expensive process. In this paper, we propose DiffUMI, a training-free diffusion-driven universal model inversion attack for face recognition systems. DiffUMI is the first approach to apply a diffusion model for unconditional image generation in model inversion. Unlike other methods, DiffUMI is universal, eliminating the need for training target-specific generators. It operates within a fixed framework and pretrained diffusion model while seamlessly adapting to diverse target identities and models. DiffUMI breaches privacy-preserving face recognition systems with state-of-the-art success, demonstrating that an unconditional diffusion model, coupled with optimized adversarial search, enables efficient and high-fidelity facial reconstruction. Additionally, we introduce a novel application of out-of-domain detection (OODD), marking the first use of model inversion to distinguish non-face inputs from face inputs based solely on embeddings.

摘要: 面部识别技术带来了巨大的隐私风险，因为它依赖于生物识别数据，这些数据本质上是敏感的，并且如果受到损害，也是不可改变的。为了减轻这些担忧，面部识别系统将原始图像转换为嵌入，传统上被认为是保护隐私的。然而，模型倒置攻击通过重建这些私人面部图像构成了重大的隐私威胁，使其成为评估面部识别系统隐私风险的重要工具。现有的方法通常需要为每个目标模型训练单个生成器，这是一个计算昂贵的过程。在本文中，我们提出了一种针对人脸识别系统的免训练扩散驱动通用模型倒置攻击。迪夫UMI是第一个在模型逆求中应用扩散模型进行无条件图像生成的方法。与其他方法不同，迪夫UMI是通用的，无需训练特定目标的生成器。它在固定的框架和预先训练的扩散模型中运行，同时无缝适应不同的目标身份和模型。迪夫UMI以最先进的成功突破了保护隐私的面部识别系统，证明无条件扩散模型加上优化的对抗性搜索可以实现高效且高保真的面部重建。此外，我们还引入了域外检测（OODD）的一种新颖应用，标志着首次使用模型倒置来区分非面部输入和仅基于嵌入的面部输入。



## **40. Adversarial Attacks to Latent Representations of Distributed Neural Networks in Split Computing**

分裂计算中对分布式神经网络潜在表示的对抗攻击 cs.LG

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2309.17401v4) [paper-pdf](http://arxiv.org/pdf/2309.17401v4)

**Authors**: Milin Zhang, Mohammad Abdi, Jonathan Ashdown, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge, the resilience of distributed DNNs to adversarial action remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and rigorously proved that (i) the compressed latent dimension improves the robustness but also affect task-oriented performance; and (ii) the deeper splitting point enhances the robustness but also increases the computational burden. These two trade-offs provide a novel perspective to design robust distributed DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN architectures, 6 different approaches for distributed DNN and 10 different adversarial attacks using the ImageNet-1K dataset.

摘要: 分布式深度神经网络（DNN）已被证明可以减少移动设备的计算负担并减少边缘计算场景中的端到端推理延迟。虽然分布式DNN已经被研究过，但据我们所知，分布式DNN对对抗行为的弹性仍然是一个悬而未决的问题。在本文中，我们通过严格分析分布式DNN对对抗行为的鲁棒性来填补现有的研究空白。我们将这个问题置于信息论的背景下，并严格证明了（i）压缩的潜在维度提高了鲁棒性，但也会影响面向任务的性能;（ii）更深的分裂点增强了鲁棒性，但也增加了计算负担。这两种权衡为设计稳健的分布式DNN提供了一个新颖的视角。为了测试我们的理论发现，我们通过考虑6种不同的DNN架构、6种不同的分布式DNN方法以及使用ImageNet-1 K数据集的10种不同的对抗性攻击来进行广泛的实验分析。



## **41. Cluster-Aware Attacks on Graph Watermarks**

对图形水印的搜索者感知攻击 cs.CR

15 pages, 16 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17971v1) [paper-pdf](http://arxiv.org/pdf/2504.17971v1)

**Authors**: Alexander Nemecek, Emre Yilmaz, Erman Ayday

**Abstract**: Data from domains such as social networks, healthcare, finance, and cybersecurity can be represented as graph-structured information. Given the sensitive nature of this data and their frequent distribution among collaborators, ensuring secure and attributable sharing is essential. Graph watermarking enables attribution by embedding user-specific signatures into graph-structured data. While prior work has addressed random perturbation attacks, the threat posed by adversaries leveraging structural properties through community detection remains unexplored. In this work, we introduce a cluster-aware threat model in which adversaries apply community-guided modifications to evade detection. We propose two novel attack strategies and evaluate them on real-world social network graphs. Our results show that cluster-aware attacks can reduce attribution accuracy by up to 80% more than random baselines under equivalent perturbation budgets on sparse graphs. To mitigate this threat, we propose a lightweight embedding enhancement that distributes watermark nodes across graph communities. This approach improves attribution accuracy by up to 60% under attack on dense graphs, without increasing runtime or structural distortion. Our findings underscore the importance of cluster-topological awareness in both watermarking design and adversarial modeling.

摘要: 来自社交网络、医疗保健、金融和网络安全等领域的数据可以表示为图形结构信息。鉴于这些数据的敏感性及其在合作者之间的频繁分布，确保安全且可归因的共享至关重要。图水印通过将用户特定的签名嵌入到图结构数据中来实现归因。虽然之前的工作已经解决了随机扰动攻击，但对手通过社区检测利用结构属性所构成的威胁仍然未被探索。在这项工作中，我们引入了一种集群感知威胁模型，其中对手应用社区指导的修改来逃避检测。我们提出了两种新颖的攻击策略，并在现实世界的社交网络图上对其进行了评估。我们的结果表明，在稀疏图上相同的扰动预算下，集群感知攻击可以比随机基线降低高达80%。为了减轻这种威胁，我们提出了一种轻量级嵌入增强，可以在图社区中分布水印节点。在密集图的攻击下，这种方法将归因准确性提高了高达60%，而不会增加运行时间或结构失真。我们的研究结果强调了集群布局意识在水印设计和对抗建模中的重要性。



## **42. ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech**

ASVspoof 5：使用众包语音设计、收集和验证用于欺骗、Deepfake和对抗性攻击检测的资源 eess.AS

Database link: https://zenodo.org/records/14498691, Database mirror  link: https://huggingface.co/datasets/jungjee/asvspoof5, ASVspoof 5 Challenge  Workshop Proceeding: https://www.isca-archive.org/asvspoof_2024/index.html

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2502.08857v4) [paper-pdf](http://arxiv.org/pdf/2502.08857v4)

**Authors**: Xin Wang, Héctor Delgado, Hemlata Tak, Jee-weon Jung, Hye-jin Shim, Massimiliano Todisco, Ivan Kukanov, Xuechen Liu, Md Sahidullah, Tomi Kinnunen, Nicholas Evans, Kong Aik Lee, Junichi Yamagishi, Myeonghun Jeong, Ge Zhu, Yongyi Zang, You Zhang, Soumi Maiti, Florian Lux, Nicolas Müller, Wangyou Zhang, Chengzhe Sun, Shuwei Hou, Siwei Lyu, Sébastien Le Maguer, Cheng Gong, Hanjie Guo, Liping Chen, Vishwanath Singh

**Abstract**: ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake attacks as well as the design of detection solutions. We introduce the ASVspoof 5 database which is generated in a crowdsourced fashion from data collected in diverse acoustic conditions (cf. studio-quality data for earlier ASVspoof databases) and from ~2,000 speakers (cf. ~100 earlier). The database contains attacks generated with 32 different algorithms, also crowdsourced, and optimised to varying degrees using new surrogate detection models. Among them are attacks generated with a mix of legacy and contemporary text-to-speech synthesis and voice conversion models, in addition to adversarial attacks which are incorporated for the first time. ASVspoof 5 protocols comprise seven speaker-disjoint partitions. They include two distinct partitions for the training of different sets of attack models, two more for the development and evaluation of surrogate detection models, and then three additional partitions which comprise the ASVspoof 5 training, development and evaluation sets. An auxiliary set of data collected from an additional 30k speakers can also be used to train speaker encoders for the implementation of attack algorithms. Also described herein is an experimental validation of the new ASVspoof 5 database using a set of automatic speaker verification and spoof/deepfake baseline detectors. With the exception of protocols and tools for the generation of spoofed/deepfake speech, the resources described in this paper, already used by participants of the ASVspoof 5 challenge in 2024, are now all freely available to the community.

摘要: ASVspoof 5是一系列挑战的第五版，该挑战促进了语音欺骗和深度伪造攻击的研究以及检测解决方案的设计。我们引入了ASVspoof 5数据库，该数据库是以众包方式根据在不同声学条件下收集的数据生成的（参见早期ASVspoof数据库的演播室质量数据）以及来自约2，000名发言者（参见~100之前）。该数据库包含由32种不同算法生成的攻击，这些算法也是众包的，并使用新的代理检测模型进行了不同程度的优化。其中包括由传统和现代文本到语音合成和语音转换模型混合产生的攻击，以及首次纳入的对抗性攻击。ASVspoof 5协议包含七个说话者不相交的分区。它们包括两个不同的分区，用于训练不同的攻击模型集，另外两个用于开发和评估代理检测模型，以及三个额外的分区，其中包括ASVspoof 5训练、开发和评估集。从另外30 k个扬声器收集的辅助数据集还可用于训练扬声器编码器以实施攻击算法。本文还描述了使用一组自动说话者验证和欺骗/深度伪造基线检测器对新ASVspoof 5数据库进行的实验验证。除了用于生成欺骗/深度伪造语音的协议和工具外，本文中描述的资源已被2024年ASVspoof 5挑战的参与者使用，现在都可以免费供社区使用。



## **43. DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing**

DCT-Shield：针对恶意图像编辑的稳健频域防御 cs.CV

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17894v1) [paper-pdf](http://arxiv.org/pdf/2504.17894v1)

**Authors**: Aniruddha Bala, Rohit Chowdhury, Rohan Jaiswal, Siddharth Roheda

**Abstract**: Advancements in diffusion models have enabled effortless image editing via text prompts, raising concerns about image security. Attackers with access to user images can exploit these tools for malicious edits. Recent defenses attempt to protect images by adding a limited noise in the pixel space to disrupt the functioning of diffusion-based editing models. However, the adversarial noise added by previous methods is easily noticeable to the human eye. Moreover, most of these methods are not robust to purification techniques like JPEG compression under a feasible pixel budget. We propose a novel optimization approach that introduces adversarial perturbations directly in the frequency domain by modifying the Discrete Cosine Transform (DCT) coefficients of the input image. By leveraging the JPEG pipeline, our method generates adversarial images that effectively prevent malicious image editing. Extensive experiments across a variety of tasks and datasets demonstrate that our approach introduces fewer visual artifacts while maintaining similar levels of edit protection and robustness to noise purification techniques.

摘要: 扩散模型的进步使得通过文本提示轻松编辑图像成为可能，这引发了人们对图像安全性的担忧。可以访问用户图像的攻击者可以利用这些工具进行恶意编辑。最近的防御措施试图通过在像素空间中添加有限的噪音来破坏基于扩散的编辑模型的功能来保护图像。然而，以前方法添加的对抗性噪音很容易被人眼注意到。此外，这些方法中的大多数对于在可行的像素预算下的JPEG压缩等净化技术并不鲁棒。我们提出了一种新颖的优化方法，通过修改输入图像的离散Cosine变换（离散Cosine变换）系数，直接在频域中引入对抗性扰动。通过利用JPEG管道，我们的方法生成对抗图像，可以有效防止恶意图像编辑。针对各种任务和数据集的广泛实验表明，我们的方法引入了更少的视觉伪影，同时保持了类似水平的编辑保护和对噪音净化技术的鲁棒性。



## **44. Unsupervised Corpus Poisoning Attacks in Continuous Space for Dense Retrieval**

连续空间中密集检索的无监督数据库中毒攻击 cs.IR

This paper has been accepted as a full paper at SIGIR 2025 and will  be presented orally

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17884v1) [paper-pdf](http://arxiv.org/pdf/2504.17884v1)

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Simon Lupart, Evangelos Kanoulas

**Abstract**: This paper concerns corpus poisoning attacks in dense information retrieval, where an adversary attempts to compromise the ranking performance of a search algorithm by injecting a small number of maliciously generated documents into the corpus. Our work addresses two limitations in the current literature. First, attacks that perform adversarial gradient-based word substitution search do so in the discrete lexical space, while retrieval itself happens in the continuous embedding space. We thus propose an optimization method that operates in the embedding space directly. Specifically, we train a perturbation model with the objective of maintaining the geometric distance between the original and adversarial document embeddings, while also maximizing the token-level dissimilarity between the original and adversarial documents. Second, it is common for related work to have a strong assumption that the adversary has prior knowledge about the queries. In this paper, we focus on a more challenging variant of the problem where the adversary assumes no prior knowledge about the query distribution (hence, unsupervised). Our core contribution is an adversarial corpus attack that is fast and effective. We present comprehensive experimental results on both in- and out-of-domain datasets, focusing on two related tasks: a top-1 attack and a corpus poisoning attack. We consider attacks under both a white-box and a black-box setting. Notably, our method can generate successful adversarial examples in under two minutes per target document; four times faster compared to the fastest gradient-based word substitution methods in the literature with the same hardware. Furthermore, our adversarial generation method generates text that is more likely to occur under the distribution of natural text (low perplexity), and is therefore more difficult to detect.

摘要: 本文关注密集信息检索中的数据库中毒攻击，其中对手试图通过将少量恶意生成的文档注入到数据库中来损害搜索算法的排名性能。我们的工作解决了当前文献中的两个局限性。首先，执行对抗性基于梯度的单词替换搜索的攻击是在离散词汇空间中进行的，而检索本身则发生在连续嵌入空间中。因此，我们提出了一种直接在嵌入空间中操作的优化方法。具体来说，我们训练一个扰动模型，目标是保持原始文档和对抗文档嵌入之间的几何距离，同时最大化原始文档和对抗文档之间的标记级差异。其次，相关工作通常强烈假设对手拥有有关查询的先验知识。在本文中，我们重点关注该问题的一个更具挑战性的变体，其中对手假设没有有关查询分布的先验知识（因此，无监督）。我们的核心贡献是快速有效的对抗性语料库攻击。我们在域内和域外数据集上展示了全面的实验结果，重点关注两个相关的任务：顶级攻击和语料库中毒攻击。我们考虑白盒和黑匣子设置下的攻击。值得注意的是，我们的方法可以在每个目标文档两分钟内生成成功的对抗性示例;与文献中使用相同硬件的最快的基于梯度的单词替换方法相比，速度快了四倍。此外，我们的对抗生成方法生成的文本更有可能在自然文本的分布下发生（低困惑度），因此更难以检测。



## **45. Siren -- Advancing Cybersecurity through Deception and Adaptive Analysis**

警报器--通过欺骗和适应性分析推进网络安全 cs.CR

14 pages, 5 figures, 13th Computing Conference 2025 - London, United  Kingdom

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2406.06225v2) [paper-pdf](http://arxiv.org/pdf/2406.06225v2)

**Authors**: Samhruth Ananthanarayanan, Girish Kulathumani, Ganesh Narayanan

**Abstract**: Siren represents a pioneering research effort aimed at fortifying cybersecurity through strategic integration of deception, machine learning, and proactive threat analysis. Drawing inspiration from mythical sirens, this project employs sophisticated methods to lure potential threats into controlled environments. The system features a dynamic machine learning model for realtime analysis and classification, ensuring continuous adaptability to emerging cyber threats. The architectural framework includes a link monitoring proxy, a purpose-built machine learning model for dynamic link analysis, and a honeypot enriched with simulated user interactions to intensify threat engagement. Data protection within the honeypot is fortified with probabilistic encryption. Additionally, the incorporation of simulated user activity extends the system's capacity to capture and learn from potential attackers even after user disengagement. Overall, Siren introduces a paradigm shift in cybersecurity, transforming traditional defense mechanisms into proactive systems that actively engage and learn from potential adversaries. The research strives to enhance user protection while yielding valuable insights for ongoing refinement in response to the evolving landscape of cybersecurity threats.

摘要: Siren代表了一项开创性的研究工作，旨在通过欺骗、机器学习和主动威胁分析的战略集成来加强网络安全。该项目从神话警报中汲取灵感，采用复杂的方法将潜在威胁引诱到受控环境中。该系统具有动态机器学习模型，用于实时分析和分类，确保对新兴网络威胁的持续适应性。该架构框架包括一个链接监控代理、一个专门构建的用于动态链接分析的机器学习模型，以及一个富含模拟用户交互以加强威胁参与的蜜罐。蜜罐内的数据保护通过概率加密得到加强。此外，模拟用户活动的结合扩展了系统捕获潜在攻击者并从其学习的能力，即使在用户脱离接触之后也是如此。总体而言，Siren引入了网络安全的范式转变，将传统防御机制转变为积极参与潜在对手并向其学习的主动系统。该研究致力于加强用户保护，同时为持续改进提供有价值的见解，以应对不断变化的网络安全威胁格局。



## **46. On the Generalization of Adversarially Trained Quantum Classifiers**

关于对抗训练量子分类器的推广 quant-ph

22 pages, 6 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17690v1) [paper-pdf](http://arxiv.org/pdf/2504.17690v1)

**Authors**: Petros Georgiou, Aaron Mark Thomas, Sharu Theresa Jose, Osvaldo Simeone

**Abstract**: Quantum classifiers are vulnerable to adversarial attacks that manipulate their input classical or quantum data. A promising countermeasure is adversarial training, where quantum classifiers are trained by using an attack-aware, adversarial loss function. This work establishes novel bounds on the generalization error of adversarially trained quantum classifiers when tested in the presence of perturbation-constrained adversaries. The bounds quantify the excess generalization error incurred to ensure robustness to adversarial attacks as scaling with the training sample size $m$ as $1/\sqrt{m}$, while yielding insights into the impact of the quantum embedding. For quantum binary classifiers employing \textit{rotation embedding}, we find that, in the presence of adversarial attacks on classical inputs $\mathbf{x}$, the increase in sample complexity due to adversarial training over conventional training vanishes in the limit of high dimensional inputs $\mathbf{x}$. In contrast, when the adversary can directly attack the quantum state $\rho(\mathbf{x})$ encoding the input $\mathbf{x}$, the excess generalization error depends on the choice of embedding only through its Hilbert space dimension. The results are also extended to multi-class classifiers. We validate our theoretical findings with numerical experiments.

摘要: 量子分类器容易受到操纵其输入经典或量子数据的对抗性攻击。一个有希望的对策是对抗训练，其中量子分类器通过使用攻击感知的对抗损失函数来训练。这项工作建立了新的边界对抗训练的量子分类器的泛化错误时，在扰动约束的对手的存在下进行测试。边界量化了为确保对抗性攻击的鲁棒性而产生的过度泛化误差，并将训练样本大小$m$缩放为$1/\sqrt{m}$，同时深入了解量子嵌入的影响。对于采用旋转嵌入的量子二进制分类器，我们发现，在经典输入$\mathbf{x}$上存在对抗性攻击的情况下，由于对抗性训练而导致的样本复杂度增加在高维输入$\mathbf{x}$的限制下消失。相比之下，当对手可以直接攻击量子态$\rho（\mathbf{x}）$编码输入$\mathbf{x}$时，多余的泛化误差取决于仅通过其希尔伯特空间维度嵌入的选择。结果也被扩展到多类分类器。我们验证了我们的理论研究结果与数值实验。



## **47. Evaluating the Vulnerability of ML-Based Ethereum Phishing Detectors to Single-Feature Adversarial Perturbations**

评估基于ML的以太坊网络钓鱼检测器对单一特征对抗扰动的脆弱性 cs.CR

24 pages; an extension of a paper that appeared at WISA 2024

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17684v1) [paper-pdf](http://arxiv.org/pdf/2504.17684v1)

**Authors**: Ahod Alghuried, Ali Alkinoon, Abdulaziz Alghamdi, Soohyeon Choi, Manar Mohaisen, David Mohaisen

**Abstract**: This paper explores the vulnerability of machine learning models to simple single-feature adversarial attacks in the context of Ethereum fraudulent transaction detection. Through comprehensive experimentation, we investigate the impact of various adversarial attack strategies on model performance metrics. Our findings, highlighting how prone those techniques are to simple attacks, are alarming, and the inconsistency in the attacks' effect on different algorithms promises ways for attack mitigation. We examine the effectiveness of different mitigation strategies, including adversarial training and enhanced feature selection, in enhancing model robustness and show their effectiveness.

摘要: 本文探讨了以太坊欺诈交易检测背景下机器学习模型对简单单特征对抗攻击的脆弱性。通过全面的实验，我们研究了各种对抗攻击策略对模型性能指标的影响。我们的研究结果强调了这些技术容易受到简单攻击的影响，这令人震惊，而且攻击对不同算法的影响的不一致性为缓解攻击提供了方法。我们研究了不同缓解策略（包括对抗训练和增强的特征选择）在增强模型稳健性方面的有效性，并展示了它们的有效性。



## **48. Regulatory Markets for AI Safety**

人工智能安全监管市场 cs.CY

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2001.00078v2) [paper-pdf](http://arxiv.org/pdf/2001.00078v2)

**Authors**: Jack Clark, Gillian K. Hadfield

**Abstract**: We propose a new model for regulation to achieve AI safety: global regulatory markets. We first sketch the model in general terms and provide an overview of the costs and benefits of this approach. We then demonstrate how the model might work in practice: responding to the risk of adversarial attacks on AI models employed in commercial drones.

摘要: 我们提出了一种实现人工智能安全的新监管模式：全球监管市场。我们首先概述该模型，并概述这种方法的成本和收益。然后，我们演示了该模型在实践中如何工作：应对商用无人机中使用的人工智能模型的对抗攻击风险。



## **49. A Simple DropConnect Approach to Transfer-based Targeted Attack**

基于传输的目标攻击的简单DropConnect方法 cs.LG

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.18594v1) [paper-pdf](http://arxiv.org/pdf/2504.18594v1)

**Authors**: Tongrui Su, Qingbin Li, Shengyu Zhu, Wei Chen, Xueqi Cheng

**Abstract**: We study the problem of transfer-based black-box attack, where adversarial samples generated using a single surrogate model are directly applied to target models. Compared with untargeted attacks, existing methods still have lower Attack Success Rates (ASRs) in the targeted setting, i.e., the obtained adversarial examples often overfit the surrogate model but fail to mislead other models. In this paper, we hypothesize that the pixels or features in these adversarial examples collaborate in a highly dependent manner to maximize the success of an adversarial attack on the surrogate model, which we refer to as perturbation co-adaptation. Then, we propose to Mitigate perturbation Co-adaptation by DropConnect (MCD) to enhance transferability, by creating diverse variants of surrogate model at each optimization iteration. We conduct extensive experiments across various CNN- and Transformer-based models to demonstrate the effectiveness of MCD. In the challenging scenario of transferring from a CNN-based model to Transformer-based models, MCD achieves 13% higher average ASRs compared with state-of-the-art baselines. MCD boosts the performance of self-ensemble methods by bringing in more diversification across the variants while reserving sufficient semantic information for each variant. In addition, MCD attains the highest performance gain when scaling the compute of crafting adversarial examples.

摘要: 我们研究基于传输的黑匣子攻击问题，其中使用单个代理模型生成的对抗样本直接应用于目标模型。与非目标攻击相比，现有方法在目标环境中仍然具有较低的攻击成功率（ASB），即获得的对抗性示例常常过度适合代理模型，但无法误导其他模型。在本文中，我们假设这些对抗性示例中的像素或特征以高度依赖的方式协作，以最大限度地提高对代理模型的对抗性攻击的成功，我们将其称为扰动协同适应。然后，我们建议通过DropConnect（MCB）缓解扰动协同适应，通过在每次优化迭代中创建代理模型的不同变体来增强可移植性。我们对各种基于CNN和Transformer的模型进行了广泛的实验，以证明MCB的有效性。在从基于CNN的模型转移到基于Transformer的模型这一具有挑战性的场景中，与最先进的基线相比，BCD的平均ASB高出13%。BCD通过在变体之间带来更多的多样化，同时为每个变体保留足够的语义信息来提高自集成方法的性能。此外，MCB在扩展制作对抗性示例的计算时获得了最高的性能提升。



## **50. GRANITE : a Byzantine-Resilient Dynamic Gossip Learning Framework**

GRANITE：一个具有拜占庭弹性的动态八卦学习框架 cs.LG

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17471v1) [paper-pdf](http://arxiv.org/pdf/2504.17471v1)

**Authors**: Yacine Belal, Mohamed Maouche, Sonia Ben Mokhtar, Anthony Simonet-Boulogne

**Abstract**: Gossip Learning (GL) is a decentralized learning paradigm where users iteratively exchange and aggregate models with a small set of neighboring peers. Recent GL approaches rely on dynamic communication graphs built and maintained using Random Peer Sampling (RPS) protocols. Thanks to graph dynamics, GL can achieve fast convergence even over extremely sparse topologies. However, the robustness of GL over dy- namic graphs to Byzantine (model poisoning) attacks remains unaddressed especially when Byzantine nodes attack the RPS protocol to scale up model poisoning. We address this issue by introducing GRANITE, a framework for robust learning over sparse, dynamic graphs in the presence of a fraction of Byzantine nodes. GRANITE relies on two key components (i) a History-aware Byzantine-resilient Peer Sampling protocol (HaPS), which tracks previously encountered identifiers to reduce adversarial influence over time, and (ii) an Adaptive Probabilistic Threshold (APT), which leverages an estimate of Byzantine presence to set aggregation thresholds with formal guarantees. Empirical results confirm that GRANITE maintains convergence with up to 30% Byzantine nodes, improves learning speed via adaptive filtering of poisoned models and obtains these results in up to 9 times sparser graphs than dictated by current theory.

摘要: Gossip Learning（GL）是一种去中心化的学习范式，用户与一小群邻近的对等点迭代地交换和聚合模型。最近的GL方法依赖于使用随机对等采样（RPS）协议构建和维护的动态通信图。得益于图形动态学，GL即使在极其稀疏的拓扑上也可以实现快速收敛。然而，GL在动态图上对拜占庭（模型中毒）攻击的鲁棒性仍然没有得到解决，尤其是当拜占庭节点攻击RPS协议以扩大模型中毒规模时。我们通过引入GRANITE来解决这个问题，GRANITE是一个在存在一小部分拜占庭节点的情况下对稀疏动态图进行鲁棒学习的框架。GRANITE依赖于两个关键组件（i）历史感知拜占庭弹性对等采样协议（HaPS），它跟踪之前遇到的标识符，以随着时间的推移减少对抗影响，以及（ii）自适应概率阈值（APT），它利用拜占庭存在的估计来设置具有正式保证的聚合阈值。经验结果证实，GRANITE可以在高达30%的拜占庭节点上保持收敛，通过有毒模型的自适应过滤提高学习速度，并在比当前理论规定的稀疏9倍的图中获得这些结果。



