# Latest Adversarial Attack Papers
**update at 2025-03-05 10:19:31**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Assessing Robustness via Score-Based Adversarial Image Generation**

通过基于分数的对抗图像生成评估稳健性 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2310.04285v3) [paper-pdf](http://arxiv.org/pdf/2310.04285v3)

**Authors**: Marcel Kollovieh, Lukas Gosch, Marten Lienen, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantics-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate unrestricted adversarial examples that overcome the limitations of $\ell_p$-norm constraints. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG improves upon the majority of state-of-the-art attacks and defenses across multiple benchmarks. This work highlights the importance of investigating adversarial examples bounded by semantics rather than $\ell_p$-norm constraints. ScoreAG represents an important step towards more encompassing robustness assessments.

摘要: 大多数对抗性攻击和防御都集中在较小的$\ell_p$-范数约束内的扰动。然而，$\ell_p$威胁模型不能捕获所有相关的语义保持扰动，因此健壮性评估的范围是有限的。在这项工作中，我们介绍了基于分数的对抗性生成(ScoreAG)，这是一个新的框架，它利用基于分数的生成模型的进步来生成不受限制的对抗性实例，克服了$\ell_p$-范数约束的限制。与传统方法不同，ScoreAG在生成敌意示例的同时保持了图像的核心语义，要么转换现有图像，要么完全从头开始合成新的图像。我们进一步利用ScoreAG的生成能力来净化图像，经验上增强了分类器的稳健性。我们广泛的经验评估表明，ScoreAG在多个基准上改进了大多数最先进的攻击和防御。这项工作强调了研究受语义约束的对抗性例子的重要性，而不是$\ell_p$-范数约束。ScoreAG代表着朝着更全面的稳健性评估迈出的重要一步。



## **2. Realizing Quantum Adversarial Defense on a Trapped-ion Quantum Processor**

在俘获离子量子处理器上实现量子对抗防御 quant-ph

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02436v1) [paper-pdf](http://arxiv.org/pdf/2503.02436v1)

**Authors**: Alex Jin, Tarun Dutta, Anh Tu Ngo, Anupam Chattopadhyay, Manas Mukherjee

**Abstract**: Classification is a fundamental task in machine learning, typically performed using classical models. Quantum machine learning (QML), however, offers distinct advantages, such as enhanced representational power through high-dimensional Hilbert spaces and energy-efficient reversible gate operations. Despite these theoretical benefits, the robustness of QML classifiers against adversarial attacks and inherent quantum noise remains largely under-explored. In this work, we implement a data re-uploading-based quantum classifier on an ion-trap quantum processor using a single qubit to assess its resilience under realistic conditions. We introduce a novel convolutional quantum classifier architecture leveraging data re-uploading and demonstrate its superior robustness on the MNIST dataset. Additionally, we quantify the effects of polarization noise in a realistic setting, where both bit and phase noises are present, further validating the classifier's robustness. Our findings provide insights into the practical security and reliability of quantum classifiers, bridging the gap between theoretical potential and real-world deployment.

摘要: 分类是机器学习中的一项基本任务，通常使用经典模型执行。然而，量子机器学习(QML)提供了独特的优势，例如通过高维希尔伯特空间和节能的可逆门操作来增强表征能力。尽管有这些理论上的好处，但QML分类器对敌意攻击和固有的量子噪声的稳健性仍未得到很大程度的研究。在这项工作中，我们使用单个量子比特在离子陷阱量子处理器上实现了一个基于数据重传的量子分类器，以评估其在现实条件下的弹性。我们介绍了一种利用数据重传的卷积量子分类器结构，并在MNIST数据集上展示了其优越的健壮性。此外，我们在实际环境中量化了极化噪声的影响，其中位噪声和相位噪声都存在，进一步验证了分类器的稳健性。我们的发现为量子分类器的实际安全性和可靠性提供了洞察力，弥合了理论潜力和现实部署之间的差距。



## **3. Trace of the Times: Rootkit Detection through Temporal Anomalies in Kernel Activity**

时代的痕迹：通过核心活动中的时间异常进行RootKit检测 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02402v1) [paper-pdf](http://arxiv.org/pdf/2503.02402v1)

**Authors**: Max Landauer, Leonhard Alton, Martina Lindorfer, Florian Skopik, Markus Wurzenberger, Wolfgang Hotwagner

**Abstract**: Kernel rootkits provide adversaries with permanent high-privileged access to compromised systems and are often a key element of sophisticated attack chains. At the same time, they enable stealthy operation and are thus difficult to detect. Thereby, they inject code into kernel functions to appear invisible to users, for example, by manipulating file enumerations. Existing detection approaches are insufficient, because they rely on signatures that are unable to detect novel rootkits or require domain knowledge about the rootkits to be detected. To overcome this challenge, our approach leverages the fact that runtimes of kernel functions targeted by rootkits increase when additional code is executed. The framework outlined in this paper injects probes into the kernel to measure time stamps of functions within relevant system calls, computes distributions of function execution times, and uses statistical tests to detect time shifts. The evaluation of our open-source implementation on publicly available data sets indicates high detection accuracy with an F1 score of 98.7\% across five scenarios with varying system states.

摘要: 内核Rootkit为攻击者提供了对受攻击系统的永久高权限访问权限，通常是复杂攻击链的关键元素。同时，它们实现了隐形操作，因此很难被检测到。因此，它们向内核函数注入代码，使其对用户不可见，例如，通过操作文件枚举。现有的检测方法是不够的，因为它们依赖于不能检测到新的Rootkit或需要关于Rootkit的领域知识来检测的签名。为了克服这一挑战，我们的方法利用了这样一个事实，即Rootkit所针对的内核函数的运行时间会随着额外代码的执行而增加。本文概述的框架将探测器注入内核，以测量相关系统调用中函数的时间戳，计算函数执行时间的分布，并使用统计测试来检测时间偏移。在公开可用的数据集上对我们的开源实现进行的评估表明，在五个不同系统状态的场景中，检测准确率很高，F1得分为98.7\%。



## **4. Evaluating the Robustness of LiDAR Point Cloud Tracking Against Adversarial Attack**

评估LiDART点云跟踪对抗攻击的鲁棒性 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2410.20893v2) [paper-pdf](http://arxiv.org/pdf/2410.20893v2)

**Authors**: Shengjing Tian, Yinan Han, Xiantong Zhao, Bin Liu, Xiuping Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.

摘要: 在这项研究中，我们深入研究了基于神经网络的LiDAR点云跟踪模型在对抗攻击下的稳健性，这是一个经常被忽略的关键方面，有助于提高性能。尽管这些模型集成了Transformer或Bird‘s Eye View(BEV)等高级架构，但在面临对手攻击、域转移或数据损坏等挑战时往往忽略了健壮性。相反，我们关注的是跟踪模型在对抗性攻击威胁下的健壮性。我们首先建立一个统一的框架，在3D对象跟踪的背景下进行对抗性攻击，这使我们能够彻底调查白盒和黑盒攻击策略。对于白盒攻击，我们定制了特定的损失函数以适应不同的跟踪范例，并将现有的方法如FGSM、C\&W和PGD扩展到点云域。针对黑盒攻击场景，我们引入了一种新的基于传输的方法，目标感知扰动生成(TAPG)算法，其双重目标是实现高攻击性能和保持低可感知性。该方法使用启发式策略来实施稀疏攻击约束，并利用随机子向量分解来增强可转移性。我们的实验结果揭示了高级跟踪方法在同时受到黑盒和白盒攻击时的显著漏洞，强调了在设计LiDAR点云跟踪模型时考虑对对手攻击的健壮性的必要性。值得注意的是，与现有方法相比，TAPG还在攻击的有效性和扰动的隐蔽性之间取得了最佳平衡。



## **5. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

NoPain：通过最佳传输奇异边界进行无箱点云攻击 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.00063v2) [paper-pdf](http://arxiv.org/pdf/2503.00063v2)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. The source code will be publicly available.

摘要: 对抗性攻击利用深度模型针对对抗性样本的脆弱性。现有的点云攻击者是为特定模型量身定做的，基于白盒或黑盒设置中的渐变迭代优化扰动。尽管它们的攻击性能很有希望，但由于代理模型的特定参数过高，它们经常难以产生可转移的对抗性样本。为了解决这一问题，我们将重点转移到数据分布本身，并引入了一种名为NoPain的新方法，该方法使用最优传输(OT)来识别跨网络点云攻击数据流形的固有奇异边界。具体地，我们首先计算噪声到目标特征空间的OT映射，然后通过定位不可微位置来识别奇异边界。最后，我们沿着奇异边界进行采样以生成对抗性点云。一旦确定了奇异边界，NoPain就可以有效地产生对抗性样本，而不需要迭代更新或来自代理分类器的指导。大量的实验表明，端到端方法在可转移性和效率方面都优于基线方法，同时在与防御策略相比也保持了显著的优势。源代码将向公众开放。



## **6. Game-Theoretic Defenses for Robust Conformal Prediction Against Adversarial Attacks in Medical Imaging**

针对医学成像中对抗攻击的鲁棒共形预测的游戏理论防御 cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2411.04376v2) [paper-pdf](http://arxiv.org/pdf/2411.04376v2)

**Authors**: Rui Luo, Jie Bao, Zhixin Zhou, Chuangyin Dang

**Abstract**: Adversarial attacks pose significant threats to the reliability and safety of deep learning models, especially in critical domains such as medical imaging. This paper introduces a novel framework that integrates conformal prediction with game-theoretic defensive strategies to enhance model robustness against both known and unknown adversarial perturbations. We address three primary research questions: constructing valid and efficient conformal prediction sets under known attacks (RQ1), ensuring coverage under unknown attacks through conservative thresholding (RQ2), and determining optimal defensive strategies within a zero-sum game framework (RQ3). Our methodology involves training specialized defensive models against specific attack types and employing maximum and minimum classifiers to aggregate defenses effectively. Extensive experiments conducted on the MedMNIST datasets, including PathMNIST, OrganAMNIST, and TissueMNIST, demonstrate that our approach maintains high coverage guarantees while minimizing prediction set sizes. The game-theoretic analysis reveals that the optimal defensive strategy often converges to a singular robust model, outperforming uniform and simple strategies across all evaluated datasets. This work advances the state-of-the-art in uncertainty quantification and adversarial robustness, providing a reliable mechanism for deploying deep learning models in adversarial environments.

摘要: 对抗性攻击对深度学习模型的可靠性和安全性构成了严重威胁，特别是在医学成像等关键领域。本文介绍了一种新的框架，它将保形预测与博弈论防御策略相结合，以增强模型对已知和未知对手扰动的稳健性。我们主要研究了三个问题：在已知攻击(RQ1)下构造有效且高效的共形预测集，通过保守阈值(RQ2)确保未知攻击下的覆盖，以及在零和博弈框架(RQ3)下确定最优防御策略。我们的方法包括针对特定的攻击类型训练专门的防御模型，并使用最大和最小分类器来有效地聚合防御。在包括PathMNIST、OrganAMNIST和TIseMNIST在内的MedMNIST数据集上进行的大量实验表明，我们的方法在保持高覆盖率的同时最小化了预测集的大小。博弈论分析表明，最优防御策略往往收敛到一个奇异的稳健模型，在所有评估的数据集上表现优于统一和简单的策略。这项工作推进了不确定性量化和对抗稳健性方面的最新进展，为在对抗环境中部署深度学习模型提供了可靠的机制。



## **7. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

TPIA：针对面向代码的大型语言模型的特定目标提示注入攻击 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2407.09164v5) [paper-pdf](http://arxiv.org/pdf/2407.09164v5)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully exploited to simplify and facilitate programming. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. In this paper, we reveal that both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process, which may not be practical; adversarial attacks struggle with fulfilling specific malicious purposes. To alleviate these problems, this paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an attack success rate of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an attack success rate of over 90%) in all threat cases, using only a 12-token non-functional perturbation.

摘要: 最近，面向代码的大型语言模型(Code LLM)已经被广泛并成功地利用来简化和促进编程。不幸的是，一些开创性的工作表明，这些代码LLM容易受到后门和对手的攻击。前者毒化训练数据或模型参数，在遇到触发器时劫持LLMS生成恶意代码片段。后者制作恶意敌意输入代码以降低生成代码的质量。在本文中，我们揭示了这两种攻击都有一些固有的局限性：后门攻击依赖于对手控制模型训练过程的能力，这可能是不实用的；对抗性攻击难以实现特定的恶意目的。针对这些问题，提出了一种新的针对代码LLMS的攻击范式，即目标特定的即时注入攻击(TPIA)。TPIA生成包含恶意指令信息的非功能性扰动，并通过将它们传播到可能使用的依赖项(例如，包或RAG的知识库)，将它们插入到受害者的代码上下文中。它诱导代码LLM在目标位置生成攻击者指定的恶意代码片段。一般而言，我们通过基于贪婪令牌搜索的对抗性优化将攻击者指定的恶意目标压缩为扰动。我们收集了13个具有代表性的恶意目标，为三种流行的编程语言设计了31个威胁案例。实验表明，在所有威胁情况下，仅使用12个令牌的非功能扰动，我们的TPIA就可以成功攻击三个典型的开源代码LLM(攻击成功率高达97.9%)和两个主流商业代码LLM集成应用(攻击成功率超过90%)。



## **8. Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion**

基于属性感知文本倒置的预算驱动可转移对抗攻击 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2502.19697v2) [paper-pdf](http://arxiv.org/pdf/2502.19697v2)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yaonan Wang

**Abstract**: Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.

摘要: 人员再识别(re-id)模型在安全监控系统中是至关重要的，需要可转移的对抗性攻击来探索其脆弱性。近年来，基于视觉语言模型(VLM)的攻击通过攻击VLM的广义图像和文本特征表现出了良好的可转移性，但由于过于强调积分表示的区分语义，缺乏全面的特征破坏。本文介绍了一种新的基于属性感知的提示攻击(AP-Attack)，该方法利用VLM的图文对齐能力，通过破坏特定于属性的文本嵌入来显式破坏行人图像的细粒度语义特征。为了获得个性化的属性文本描述，文本倒置网络被设计成将行人图像映射到代表语义嵌入的伪标记，并利用图像和预定义的提示模板进行对比学习，以明确描述行人属性。反转良性和对抗性细粒度的文本语义有助于攻击者有效地进行彻底的破坏，增强了对抗性实例的可转移性。大量实验表明，AP-Attack具有最好的可转移性，在跨模型和数据集攻击场景下，平均丢失率比以前的方法高出22.9%。



## **9. Adversarial Tokenization**

对抗性代币化 cs.CL

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02174v1) [paper-pdf](http://arxiv.org/pdf/2503.02174v1)

**Authors**: Renato Lui Geh, Zilei Shao, Guy Van den Broeck

**Abstract**: Current LLM pipelines account for only one possible tokenization for a given string, ignoring exponentially many alternative tokenizations during training and inference. For example, the standard Llama3 tokenization of penguin is [p,enguin], yet [peng,uin] is another perfectly valid alternative. In this paper, we show that despite LLMs being trained solely on one tokenization, they still retain semantic understanding of other tokenizations, raising questions about their implications in LLM safety. Put succinctly, we answer the following question: can we adversarially tokenize an obviously malicious string to evade safety and alignment restrictions? We show that not only is adversarial tokenization an effective yet previously neglected axis of attack, but it is also competitive against existing state-of-the-art adversarial approaches without changing the text of the harmful request. We empirically validate this exploit across three state-of-the-art LLMs and adversarial datasets, revealing a previously unknown vulnerability in subword models.

摘要: 当前的LLM流水线只考虑了给定字符串的一个可能的标记化，在训练和推理期间忽略了指数级的许多替代标记化。例如，企鹅的标准Llama3标记化是[p，enguin]，然而[peng，uin]是另一个完全有效的替代方案。在这篇文章中，我们证明了尽管LLM只接受了一个标记化的训练，但它们仍然保留了对其他标记化的语义理解，这引发了人们对它们在LLM安全性方面的影响的疑问。简而言之，我们回答了以下问题：我们可以恶意地对一个明显恶意的字符串进行标记，以逃避安全和对齐限制吗？我们表明，对抗性标记化不仅是一个以前被忽视的有效攻击轴心，而且在不改变有害请求的案文的情况下，也是与现有最先进的对抗性方法相竞争的。我们在三个最先进的LLM和对抗性数据集上实证验证了这一利用，揭示了子词模型中一个以前未知的漏洞。



## **10. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

HoSNN：具有自适应激发阈值的对抗鲁棒自适应尖峰神经网络 cs.NE

Accepted by TMLR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2308.10373v4) [paper-pdf](http://arxiv.org/pdf/2308.10373v4)

**Authors**: Hejia Geng, Peng Li

**Abstract**: While spiking neural networks (SNNs) offer a promising neurally-inspired model of computation, they are vulnerable to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to design a threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model and utilize TA-LIF neurons to construct the adversarially robust homeostatic SNNs (HoSNNs) for improved robustness. The TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, offering a local feedback control solution to the minimization of each neuron's membrane potential error caused by adversarial disturbance. Theoretical analysis demonstrates favorable dynamic properties of TA-LIF neurons in terms of the bounded-input bounded-output stability and suppressed time growth of membrane potential error, underscoring their superior robustness compared with the standard LIF neurons. When trained with weak FGSM attacks (attack budget = 2/255) and tested with much stronger PGD attacks (attack budget = 8/255), our HoSNNs significantly improve model accuracy on several datasets: from 30.54% to 74.91% on FashionMNIST, from 0.44% to 35.06% on SVHN, from 0.56% to 42.63% on CIFAR10, from 0.04% to 16.66% on CIFAR100, over the conventional LIF-based SNNs.

摘要: 虽然尖峰神经网络(SNN)提供了一种很有前途的神经启发计算模型，但它们容易受到对手的攻击。我们首次从神经元自平衡中得到启发，设计了一种阈值自适应泄漏积分与点火(TA-LIF)神经元模型，并利用TA-LIF神经元来构造逆稳健的自平衡SNN(HoSNN)，以提高鲁棒性。TA-LIF模型引入了一种自稳定的动态阈值机制，提供了一种局部反馈控制方案来最小化对抗性干扰引起的每个神经元的膜电位误差。理论分析表明，TA-LIF神经元在有界输入有界输出稳定性和抑制膜电位误差的时间增长方面具有良好的动态特性，与标准LIF神经元相比具有更好的鲁棒性。当用弱FGSM攻击(攻击预算=2/255)训练和用更强的PGD攻击(攻击预算=8/255)测试时，我们的HoSNN在几个数据集上显著提高了模型准确率：在FashionMNIST上从30.54%到74.91%，在SVHN上从0.44%到35.06%，在CIFAR10上从0.56%到42.63%，在CIFAR100上从0.04%到16.66%，比传统的基于LIF的SNN。



## **11. DDAD: A Two-pronged Adversarial Defense Based on Distributional Discrepancy**

DDAD：基于分配差异的双管齐下的对抗防御 cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02169v1) [paper-pdf](http://arxiv.org/pdf/2503.02169v1)

**Authors**: Jiacheng Zhang, Benjamin I. P. Rubinstein, Jingfeng Zhang, Feng Liu

**Abstract**: Statistical adversarial data detection (SADD) detects whether an upcoming batch contains adversarial examples (AEs) by measuring the distributional discrepancies between clean examples (CEs) and AEs. In this paper, we reveal the potential strength of SADD-based methods by theoretically showing that minimizing distributional discrepancy can help reduce the expected loss on AEs. Nevertheless, despite these advantages, SADD-based methods have a potential limitation: they discard inputs that are detected as AEs, leading to the loss of clean information within those inputs. To address this limitation, we propose a two-pronged adversarial defense method, named Distributional-Discrepancy-based Adversarial Defense (DDAD). In the training phase, DDAD first optimizes the test power of the maximum mean discrepancy (MMD) to derive MMD-OPT, and then trains a denoiser by minimizing the MMD-OPT between CEs and AEs. In the inference phase, DDAD first leverages MMD-OPT to differentiate CEs and AEs, and then applies a two-pronged process: (1) directly feeding the detected CEs into the classifier, and (2) removing noise from the detected AEs by the distributional-discrepancy-based denoiser. Extensive experiments show that DDAD outperforms current state-of-the-art (SOTA) defense methods by notably improving clean and robust accuracy on CIFAR-10 and ImageNet-1K against adaptive white-box attacks.

摘要: 统计对抗性数据检测(SADD)通过测量干净样本(CE)和对抗性样本(AE)之间的分布差异来检测即将到来的一批是否包含对抗性样本(AE)。在本文中，我们揭示了基于SADD的方法的潜在优势，从理论上证明了最小化分布差异可以帮助减少AEs的预期损失。然而，尽管有这些优点，基于SADD的方法有一个潜在的局限性：它们丢弃被检测为AE的输入，导致在这些输入中丢失干净的信息。针对这一局限性，我们提出了一种双管齐下的对抗性防御方法，称为基于分布差异的对抗性防御(DDAD)。在训练阶段，DDAD首先优化最大平均偏差(MMD)的测试功率得到MMD-OPT，然后通过最小化CES和AES之间的MMD-OPT来训练去噪器。在推理阶段，DDAD首先利用MMD-OPT来区分CE和AE，然后采用双管齐下的方法：(1)直接将检测到的CE送入分类器；(2)通过基于分布差异的去噪器去除检测到的AE中的噪声。大量的实验表明，DDAD在CIFAR-10和ImageNet-1K上显著提高了对自适应白盒攻击的干净和健壮的准确性，从而超过了当前最先进的(SOTA)防御方法。



## **12. Towards Scalable Topological Regularizers**

迈向可扩展的布局调节器 cs.LG

31 pages, ICLR 2025 camera-ready version

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2501.14641v2) [paper-pdf](http://arxiv.org/pdf/2501.14641v2)

**Authors**: Hiu-Tung Wong, Darrick Lee, Hong Yan

**Abstract**: Latent space matching, which consists of matching distributions of features in latent space, is a crucial component for tasks such as adversarial attacks and defenses, domain adaptation, and generative modelling. Metrics for probability measures, such as Wasserstein and maximum mean discrepancy, are commonly used to quantify the differences between such distributions. However, these are often costly to compute, or do not appropriately take the geometric and topological features of the distributions into consideration. Persistent homology is a tool from topological data analysis which quantifies the multi-scale topological structure of point clouds, and has recently been used as a topological regularizer in learning tasks. However, computation costs preclude larger scale computations, and discontinuities in the gradient lead to unstable training behavior such as in adversarial tasks. We propose the use of principal persistence measures, based on computing the persistent homology of a large number of small subsamples, as a topological regularizer. We provide a parallelized GPU implementation of this regularizer, and prove that gradients are continuous for smooth densities. Furthermore, we demonstrate the efficacy of this regularizer on shape matching, image generation, and semi-supervised learning tasks, opening the door towards a scalable regularizer for topological features.

摘要: 潜在空间匹配由潜在空间中特征的匹配分布组成，是对抗性攻击和防御、领域自适应和产生式建模等任务的重要组成部分。概率度量指标，如Wasserstein和最大均值差异，通常用于量化此类分布之间的差异。然而，这些通常计算成本很高，或者没有适当地考虑分布的几何和拓扑特征。持久同调是一种从拓扑数据分析中量化点云多尺度拓扑结构的工具，近年来被用作学习任务中的拓扑正则化工具。然而，计算成本排除了更大规模的计算，并且梯度中的不连续会导致不稳定的训练行为，例如在对抗性任务中。在计算大量小样本的持久同调的基础上，我们提出使用主持久度量作为拓扑正则化。我们给出了这种正则化算法的并行GPU实现，并证明了对于光滑的密度，梯度是连续的。此外，我们展示了这种正则化算法在形状匹配、图像生成和半监督学习任务中的有效性，为拓扑特征的可伸缩正则化算法打开了大门。



## **13. OrbID: Identifying Orbcomm Satellite RF Fingerprints**

OrbID：识别OrbSYS卫星RF指纹 eess.SP

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.02118v1) [paper-pdf](http://arxiv.org/pdf/2503.02118v1)

**Authors**: Cédric Solenthaler, Joshua Smailes Martin Strohmeier

**Abstract**: An increase in availability of Software Defined Radios (SDRs) has caused a dramatic shift in the threat landscape of legacy satellite systems, opening them up to easy spoofing attacks by low-budget adversaries. Physical-layer authentication methods can help improve the security of these systems by providing additional validation without modifying the space segment. This paper extends previous research on Radio Frequency Fingerprinting (RFF) of satellite communication to the Orbcomm satellite formation. The GPS and Iridium constellations are already well covered in prior research, but the feasibility of transferring techniques to other formations has not yet been examined, and raises previously undiscussed challenges.   In this paper, we collect a novel dataset containing 8992474 packets from the Orbcom satellite constellation using different SDRs and locations. We use this dataset to train RFF systems based on convolutional neural networks. We achieve an ROC AUC score of 0.53 when distinguishing different satellites within the constellation, and 0.98 when distinguishing legitimate satellites from SDRs in a spoofing scenario. We also demonstrate the possibility of mixing datasets using different SDRs in different physical locations.

摘要: 软件定义无线电(SDR)可用性的增加导致传统卫星系统的威胁格局发生了戏剧性的变化，使它们容易受到低预算对手的欺骗攻击。物理层身份验证方法可以在不修改空间段的情况下提供额外的验证，从而帮助提高这些系统的安全性。本文将以往卫星通信射频指纹识别的研究扩展到Orbcomm卫星编队。GPS和Iridium星座已经在以前的研究中得到了很好的覆盖，但将技术转移到其他编队的可行性还没有被审查，这提出了以前没有讨论过的挑战。在本文中，我们使用不同的SDR和不同的位置从Orbcom卫星星座收集了一个包含8992474个信息包的新数据集。我们使用这个数据集来训练基于卷积神经网络的RFF系统。当区分星座内的不同卫星时，我们的ROC AUC得分为0.53，在欺骗情况下区分合法卫星和SDR时，我们的ROC AUC得分为0.98。我们还演示了在不同的物理位置使用不同的SDR混合数据集的可能性。



## **14. Jailbreaking Safeguarded Text-to-Image Models via Large Language Models**

通过大型语言模型越狱受保护的文本到图像模型 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01839v1) [paper-pdf](http://arxiv.org/pdf/2503.01839v1)

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.

摘要: 文本到图像模型可能会生成有害内容，例如色情图像，特别是在提交不安全提示时。为了解决这个问题，通常在文本到图像模型之上添加安全过滤器，或者对模型本身进行调整以减少有害输出。然而，当攻击者战略性地设计对抗提示来绕过这些安全护栏时，这些防御仍然容易受到攻击。在这项工作中，我们提出了ObjetTune，这是一种使用微调的大型语言模型来越狱具有安全护栏的文本到图像模型的方法。与其他需要对目标模型重复查询的基于查询的越狱攻击不同，我们的攻击在微调AttackLLM后有效地生成对抗提示。我们在三个不安全提示数据集和五个安全护栏上评估我们的方法。我们的结果表明，我们的方法有效地绕过了安全护栏，优于现有的无框攻击，并且还促进了其他基于查询的攻击。



## **15. AutoAdvExBench: Benchmarking autonomous exploitation of adversarial example defenses**

AutoAdvExBench：对抗性示例防御的自主利用基准 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01811v1) [paper-pdf](http://arxiv.org/pdf/2503.01811v1)

**Authors**: Nicholas Carlini, Javier Rando, Edoardo Debenedetti, Milad Nasr, Florian Tramèr

**Abstract**: We introduce AutoAdvExBench, a benchmark to evaluate if large language models (LLMs) can autonomously exploit defenses to adversarial examples. Unlike existing security benchmarks that often serve as proxies for real-world tasks, bench directly measures LLMs' success on tasks regularly performed by machine learning security experts. This approach offers a significant advantage: if a LLM could solve the challenges presented in bench, it would immediately present practical utility for adversarial machine learning researchers. We then design a strong agent that is capable of breaking 75% of CTF-like ("homework exercise") adversarial example defenses. However, we show that this agent is only able to succeed on 13% of the real-world defenses in our benchmark, indicating the large gap between difficulty in attacking "real" code, and CTF-like code. In contrast, a stronger LLM that can attack 21% of real defenses only succeeds on 54% of CTF-like defenses. We make this benchmark available at https://github.com/ethz-spylab/AutoAdvExBench.

摘要: 我们引入了AutoAdvExB边，这是一个基准，用来评估大型语言模型(LLM)是否能够自主地利用对对手例子的防御。与通常作为真实任务代理的现有安全基准不同，BASE直接衡量LLMS在机器学习安全专家定期执行的任务中的成功程度。这种方法提供了一个显著的优势：如果LLM能够解决BASE中提出的挑战，它将立即为对抗性机器学习研究人员提供实用价值。然后，我们设计了一个强大的代理，它能够打破75%的CTF类(“家庭作业练习”)对抗性范例防御。然而，我们表明，在我们的基准测试中，该代理只能够在13%的真实世界防御中成功，这表明攻击“真实”代码的难度与类似CTF的代码之间存在巨大差距。相比之下，更强大的LLM可以攻击21%的真实防御，只能在54%的CTF类防御上成功。我们在https://github.com/ethz-spylab/AutoAdvExBench.上提供此基准测试



## **16. Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models**

猫混淆推理LLM：推理模型的查询不可知对抗触发器 cs.CL

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01781v1) [paper-pdf](http://arxiv.org/pdf/2503.01781v1)

**Authors**: Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani

**Abstract**: We investigate the robustness of reasoning models trained for step-by-step problem solving by introducing query-agnostic adversarial triggers - short, irrelevant text that, when appended to math problems, systematically mislead models to output incorrect answers without altering the problem's semantics. We propose CatAttack, an automated iterative attack pipeline for generating triggers on a weaker, less expensive proxy model (DeepSeek V3) and successfully transfer them to more advanced reasoning target models like DeepSeek R1 and DeepSeek R1-distilled-Qwen-32B, resulting in greater than 300% increase in the likelihood of the target model generating an incorrect answer. For example, appending, "Interesting fact: cats sleep most of their lives," to any math problem leads to more than doubling the chances of a model getting the answer wrong. Our findings highlight critical vulnerabilities in reasoning models, revealing that even state-of-the-art models remain susceptible to subtle adversarial inputs, raising security and reliability concerns. The CatAttack triggers dataset with model responses is available at https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers.

摘要: 我们通过引入查询不可知的对抗性触发器来研究为逐步解决问题而训练的推理模型的稳健性。查询不可知的对抗性触发器是一种简短的、无关的文本，当附加到数学问题后，系统地误导模型输出不正确的答案，而不改变问题的语义。我们提出了CatAttack，这是一种自动迭代攻击管道，用于在较弱、较便宜的代理模型(DeepSeek V3)上生成触发器，并成功地将它们传输到更高级的推理目标模型，如DeepSeek R1和DeepSeek R1-Distilleed-Qwen-32B，导致目标模型生成错误答案的可能性增加300%以上。例如，在任何一道数学题上加上“有趣的事实：猫一生中的大部分时间都在睡觉”，就会使模型答错答案的几率增加一倍以上。我们的发现突显了推理模型中的关键漏洞，揭示了即使是最先进的模型也仍然容易受到微妙的对手输入的影响，这引发了安全和可靠性方面的担忧。带有模型响应的CatAttack触发器数据集可在https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers.上获得



## **17. Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning**

对抗性代理：利用强化学习进行黑匣子规避攻击 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01734v1) [paper-pdf](http://arxiv.org/pdf/2503.01734v1)

**Authors**: Kyle Domico, Jean-Charles Noirot Ferrand, Ryan Sheatsley, Eric Pauley, Josiah Hanna, Patrick McDaniel

**Abstract**: Reinforcement learning (RL) offers powerful techniques for solving complex sequential decision-making tasks from experience. In this paper, we demonstrate how RL can be applied to adversarial machine learning (AML) to develop a new class of attacks that learn to generate adversarial examples: inputs designed to fool machine learning models. Unlike traditional AML methods that craft adversarial examples independently, our RL-based approach retains and exploits past attack experience to improve future attacks. We formulate adversarial example generation as a Markov Decision Process and evaluate RL's ability to (a) learn effective and efficient attack strategies and (b) compete with state-of-the-art AML. On CIFAR-10, our agent increases the success rate of adversarial examples by 19.4% and decreases the median number of victim model queries per adversarial example by 53.2% from the start to the end of training. In a head-to-head comparison with a state-of-the-art image attack, SquareAttack, our approach enables an adversary to generate adversarial examples with 13.1% more success after 5000 episodes of training. From a security perspective, this work demonstrates a powerful new attack vector that uses RL to attack ML models efficiently and at scale.

摘要: 强化学习(RL)为从经验中解决复杂的顺序决策任务提供了强大的技术。在本文中，我们演示了如何将RL应用于对抗性机器学习(AML)，以开发一类新的攻击，这些攻击学习生成对抗性示例：旨在愚弄机器学习模型的输入。与独立制作对抗性例子的传统AML方法不同，我们基于RL的方法保留并利用过去的攻击经验来改进未来的攻击。我们将敌意实例生成描述为一个马尔可夫决策过程，并评估了RL在(A)学习有效和高效的攻击策略和(B)与最先进的AML竞争的能力。在CIFAR-10上，从训练开始到训练结束，我们的代理将对抗性实例的成功率提高了19.4%，并将每个对抗性实例的受害者模型查询的中位数减少了53.2%。在与最先进的图像攻击SquareAttack进行面对面的比较中，我们的方法使对手能够在5000集的训练后生成对抗性例子，成功率提高13.1%。从安全的角度来看，这项工作展示了一种强大的新攻击载体，它使用RL来高效和大规模地攻击ML模型。



## **18. Towards Physically Realizable Adversarial Attacks in Embodied Vision Navigation**

视觉导航中实现物理可实现的对抗攻击 cs.CV

7 pages, 7 figures, submitted to IEEE/RSJ International Conference on  Intelligent Robots and Systems (IROS) 2025

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.10071v4) [paper-pdf](http://arxiv.org/pdf/2409.10071v4)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The significant advancements in embodied vision navigation have raised concerns about its susceptibility to adversarial attacks exploiting deep neural networks. Investigating the adversarial robustness of embodied vision navigation is crucial, especially given the threat of 3D physical attacks that could pose risks to human safety. However, existing attack methods for embodied vision navigation often lack physical feasibility due to challenges in transferring digital perturbations into the physical world. Moreover, current physical attacks for object detection struggle to achieve both multi-view effectiveness and visual naturalness in navigation scenarios. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches to objects, where both opacity and textures are learnable. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which optimizes the patch's texture based on feedback from the vision-based perception model used in navigation. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, in which opacity is fine-tuned after texture optimization. Experimental results demonstrate that our adversarial patches decrease the navigation success rate by an average of 22.39%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav

摘要: 体现视觉导航的重大进步引起了人们对其易受利用深度神经网络的敌意攻击的担忧。研究体现视觉导航的对抗健壮性至关重要，特别是考虑到可能对人类安全构成风险的3D物理攻击的威胁。然而，现有的具身视觉导航攻击方法往往缺乏物理可行性，这是因为将数字扰动转换到物理世界的挑战。此外，目前用于目标检测的物理攻击难以在导航场景中实现多视角有效性和视觉自然度。为了解决这个问题，我们提出了一种实用的具身导航攻击方法，通过在对象上附加敌意补丁来实现，其中不透明度和纹理都是可学习的。具体地说，为了确保不同视点的有效性，我们采用了一种基于对象感知采样的多视点优化策略，该策略根据导航中使用的基于视觉的感知模型的反馈来优化面片的纹理。为了使面片不易被人察觉，我们引入了一种两阶段不透明度优化机制，在纹理优化后对不透明度进行微调。实验结果表明，我们的对抗性补丁使导航成功率平均降低了22.39%，在实用性、有效性和自然性方面都优于以往的方法。代码可从以下网址获得：https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav



## **19. Revisiting Locally Differentially Private Protocols: Towards Better Trade-offs in Privacy, Utility, and Attack Resistance**

重新审视本地差异私有协议：在隐私、实用性和抗攻击方面实现更好的权衡 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01482v1) [paper-pdf](http://arxiv.org/pdf/2503.01482v1)

**Authors**: Héber H. Arcolezi, Sébastien Gambs

**Abstract**: Local Differential Privacy (LDP) offers strong privacy protection, especially in settings in which the server collecting the data is untrusted. However, designing LDP mechanisms that achieve an optimal trade-off between privacy, utility, and robustness to adversarial inference attacks remains challenging. In this work, we introduce a general multi-objective optimization framework for refining LDP protocols, enabling the joint optimization of privacy and utility under various adversarial settings. While our framework is flexible enough to accommodate multiple privacy and security attacks as well as utility metrics, in this paper we specifically optimize for Attacker Success Rate (ASR) under distinguishability attack as a measure of privacy and Mean Squared Error (MSE) as a measure of utility. We systematically revisit these trade-offs by analyzing eight state-of-the-art LDP protocols and proposing refined counterparts that leverage tailored optimization techniques. Experimental results demonstrate that our proposed adaptive mechanisms consistently outperform their non-adaptive counterparts, reducing ASR by up to five orders of magnitude while maintaining competitive utility. Analytical derivations also confirm the effectiveness of our mechanisms, moving them closer to the ASR-MSE Pareto frontier.

摘要: 本地差异隐私(LDP)提供强大的隐私保护，尤其是在收集数据的服务器不受信任的情况下。然而，设计LDP机制以实现隐私、效用和对抗推理攻击的稳健性之间的最佳平衡仍然具有挑战性。在这项工作中，我们引入了一个通用的多目标优化框架来精化LDP协议，使得在不同的敌意环境下能够联合优化隐私和效用。虽然我们的框架足够灵活，可以适应多种隐私和安全攻击以及效用度量，但在本文中，我们特别针对可区分攻击下的攻击者成功率(ASR)和效用度量的均方误差(MSE)进行了优化。我们通过分析八种最先进的LDP协议并提出利用定制优化技术的改进的对等协议，系统地重新审视了这些权衡。实验结果表明，我们提出的自适应机制的性能始终优于非自适应机制，在保持竞争效用的同时，ASR降低了高达五个数量级。解析推导也证实了我们的机制的有效性，使它们更接近ASR-MSE的帕累托前沿。



## **20. Poison-splat: Computation Cost Attack on 3D Gaussian Splatting**

毒药飞溅：对3D高斯飞溅的计算成本攻击 cs.CV

Accepted by ICLR 2025 as a spotlight paper

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2410.08190v2) [paper-pdf](http://arxiv.org/pdf/2410.08190v2)

**Authors**: Jiahao Lu, Yifan Zhang, Qiuhong Shen, Xinchao Wang, Shuicheng Yan

**Abstract**: 3D Gaussian splatting (3DGS), known for its groundbreaking performance and efficiency, has become a dominant 3D representation and brought progress to many 3D vision tasks. However, in this work, we reveal a significant security vulnerability that has been largely overlooked in 3DGS: the computation cost of training 3DGS could be maliciously tampered by poisoning the input data. By developing an attack named Poison-splat, we reveal a novel attack surface where the adversary can poison the input images to drastically increase the computation memory and time needed for 3DGS training, pushing the algorithm towards its worst computation complexity. In extreme cases, the attack can even consume all allocable memory, leading to a Denial-of-Service (DoS) that disrupts servers, resulting in practical damages to real-world 3DGS service vendors. Such a computation cost attack is achieved by addressing a bi-level optimization problem through three tailored strategies: attack objective approximation, proxy model rendering, and optional constrained optimization. These strategies not only ensure the effectiveness of our attack but also make it difficult to defend with simple defensive measures. We hope the revelation of this novel attack surface can spark attention to this crucial yet overlooked vulnerability of 3DGS systems. Our code is available at https://github.com/jiahaolu97/poison-splat .

摘要: 三维高斯飞溅(3DGS)以其开创性的性能和效率而闻名，已经成为一种占主导地位的3D表示，并为许多3D视觉任务带来了进展。然而，在这项工作中，我们揭示了一个在3DGS中被很大程度上忽视的重大安全漏洞：通过毒化输入数据，可以恶意篡改训练3DGS的计算成本。通过开发一种名为Poison-Splat的攻击，我们揭示了一种新颖的攻击面，在该攻击面上，攻击者可以毒化输入图像，从而大幅增加3DGS训练所需的计算内存和时间，从而将算法推向最差的计算复杂度。在极端情况下，攻击甚至会耗尽所有可分配的内存，导致服务器中断的拒绝服务(DoS)，从而对现实世界中的3DGS服务供应商造成实际损害。这样的计算代价攻击是通过三种定制的策略来解决双层优化问题来实现的：攻击目标近似、代理模型渲染和可选的约束优化。这些策略不仅确保了我们进攻的有效性，而且使我们很难用简单的防御措施进行防守。我们希望这一新颖攻击面的揭示能引起人们对3DGS系统这一关键但被忽视的漏洞的关注。我们的代码可以在https://github.com/jiahaolu97/poison-splat上找到。



## **21. Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification**

分而治：基于扩散的对抗净化的异类噪音集成 cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01407v1) [paper-pdf](http://arxiv.org/pdf/2503.01407v1)

**Authors**: Gaozheng Pei, Shaojie Lyu, Gong Chen, Ke Ma, Qianqian Xu, Yingfei Sun, Qingming Huang

**Abstract**: Existing diffusion-based purification methods aim to disrupt adversarial perturbations by introducing a certain amount of noise through a forward diffusion process, followed by a reverse process to recover clean examples. However, this approach is fundamentally flawed: the uniform operation of the forward process across all pixels compromises normal pixels while attempting to combat adversarial perturbations, resulting in the target model producing incorrect predictions. Simply relying on low-intensity noise is insufficient for effective defense. To address this critical issue, we implement a heterogeneous purification strategy grounded in the interpretability of neural networks. Our method decisively applies higher-intensity noise to specific pixels that the target model focuses on while the remaining pixels are subjected to only low-intensity noise. This requirement motivates us to redesign the sampling process of the diffusion model, allowing for the effective removal of varying noise levels. Furthermore, to evaluate our method against strong adaptative attack, our proposed method sharply reduces time cost and memory usage through a single-step resampling. The empirical evidence from extensive experiments across three datasets demonstrates that our method outperforms most current adversarial training and purification techniques by a substantial margin.

摘要: 现有的基于扩散的净化方法旨在通过正向扩散过程引入一定量的噪声，然后通过反向过程来恢复干净的样本，从而破坏对抗性扰动。然而，这种方法从根本上是有缺陷的：在试图对抗对抗性扰动的同时，在所有像素上的前向过程的统一操作损害了正常像素，导致目标模型产生错误的预测。单纯依靠低强度噪声是不能有效防御的。为了解决这一关键问题，我们实施了一种基于神经网络可解释性的异质净化策略。我们的方法果断地将较高强度的噪声应用于目标模型关注的特定像素，而其余像素仅受到低强度噪声的影响。这一要求促使我们重新设计扩散模型的采样过程，允许有效地去除不同的噪声水平。此外，为了评估我们的方法对强适应性攻击的抵抗力，我们提出的方法通过一步重采样大大减少了时间开销和内存使用。来自三个数据集的广泛实验的经验证据表明，我们的方法比目前大多数对抗性训练和净化技术有很大的优势。



## **22. Attacking Large Language Models with Projected Gradient Descent**

使用投影梯度下降攻击大型语言模型 cs.LG

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2402.09154v2) [paper-pdf](http://arxiv.org/pdf/2402.09154v2)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Johannes Gasteiger, Stephan Günnemann

**Abstract**: Current LLM alignment methods are readily broken through specifically crafted adversarial prompts. While crafting adversarial prompts using discrete optimization is highly effective, such attacks typically use more than 100,000 LLM calls. This high computational cost makes them unsuitable for, e.g., quantitative analyses and adversarial training. To remedy this, we revisit Projected Gradient Descent (PGD) on the continuously relaxed input prompt. Although previous attempts with ordinary gradient-based attacks largely failed, we show that carefully controlling the error introduced by the continuous relaxation tremendously boosts their efficacy. Our PGD for LLMs is up to one order of magnitude faster than state-of-the-art discrete optimization to achieve the same devastating attack results.

摘要: 当前的LLM对齐方法很容易通过专门设计的对抗提示来突破。虽然使用离散优化制作对抗提示非常有效，但此类攻击通常使用超过100，000次LLM调用。这种高计算成本使它们不适合例如定量分析和对抗训练。为了解决这个问题，我们在持续放松的输入提示下重新审视投影梯度下降（PVD）。尽管之前对普通的基于梯度的攻击的尝试基本上失败了，但我们表明，仔细控制持续放松带来的错误可以极大地提高它们的功效。我们的LLM PGO比最先进的离散优化快一个数量级，以实现相同的毁灭性攻击结果。



## **23. Exact Certification of (Graph) Neural Networks Against Label Poisoning**

（图）神经网络对抗标签中毒的精确认证 cs.LG

Published as a spotlight presentation at ICLR 2025

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2412.00537v2) [paper-pdf](http://arxiv.org/pdf/2412.00537v2)

**Authors**: Mahalakshmi Sabanayagam, Lukas Gosch, Stephan Günnemann, Debarghya Ghoshdastidar

**Abstract**: Machine learning models are highly vulnerable to label flipping, i.e., the adversarial modification (poisoning) of training labels to compromise performance. Thus, deriving robustness certificates is important to guarantee that test predictions remain unaffected and to understand worst-case robustness behavior. However, for Graph Neural Networks (GNNs), the problem of certifying label flipping has so far been unsolved. We change this by introducing an exact certification method, deriving both sample-wise and collective certificates. Our method leverages the Neural Tangent Kernel (NTK) to capture the training dynamics of wide networks enabling us to reformulate the bilevel optimization problem representing label flipping into a Mixed-Integer Linear Program (MILP). We apply our method to certify a broad range of GNN architectures in node classification tasks. Thereby, concerning the worst-case robustness to label flipping: $(i)$ we establish hierarchies of GNNs on different benchmark graphs; $(ii)$ quantify the effect of architectural choices such as activations, depth and skip-connections; and surprisingly, $(iii)$ uncover a novel phenomenon of the robustness plateauing for intermediate perturbation budgets across all investigated datasets and architectures. While we focus on GNNs, our certificates are applicable to sufficiently wide NNs in general through their NTK. Thus, our work presents the first exact certificate to a poisoning attack ever derived for neural networks, which could be of independent interest. The code is available at https://github.com/saper0/qpcert.

摘要: 机器学习模型很容易受到标签翻转的影响，即对训练标签进行对抗性修改(中毒)以损害性能。因此，派生健壮性证书对于保证测试预测不受影响以及了解最坏情况下的健壮性行为非常重要。然而，对于图神经网络(GNN)来说，证明标签翻转的问题到目前为止还没有解决。我们通过引入一种精确的认证方法来改变这一点，即同时派生样本证书和集合证书。我们的方法利用神经切核(NTK)来捕捉广域网络的训练动态，使我们能够将表示标签翻转的双层优化问题重新描述为混合整数线性规划(MILP)。我们应用我们的方法在节点分类任务中验证了广泛的GNN体系结构。因此，关于标签翻转的最坏情况的稳健性：$(I)$我们在不同的基准图上建立了GNN的层次结构；$(Ii)$量化了体系结构选择的影响，例如激活、深度和跳过连接；令人惊讶的是，$(Iii)$发现了一个新的现象，即在所有调查的数据集和体系结构中，中间扰动预算的稳健性停滞不前。虽然我们专注于GNN，但我们的证书一般通过其NTK适用于足够广泛的NN。因此，我们的工作提供了有史以来第一个针对神经网络的中毒攻击的确切证书，这可能是独立的兴趣。代码可在https://github.com/saper0/qpcert.上获得



## **24. AdvLogo: Adversarial Patch Attack against Object Detectors based on Diffusion Models**

AdvLogo：针对基于扩散模型的对象检测器的对抗补丁攻击 cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.07002v2) [paper-pdf](http://arxiv.org/pdf/2409.07002v2)

**Authors**: Boming Miao, Chunxiao Li, Yao Zhu, Weixiang Sun, Zizhe Wang, Xiaoyi Wang, Chuanlong Xie

**Abstract**: With the rapid development of deep learning, object detectors have demonstrated impressive performance; however, vulnerabilities still exist in certain scenarios. Current research exploring the vulnerabilities using adversarial patches often struggles to balance the trade-off between attack effectiveness and visual quality. To address this problem, we propose a novel framework of patch attack from semantic perspective, which we refer to as AdvLogo. Based on the hypothesis that every semantic space contains an adversarial subspace where images can cause detectors to fail in recognizing objects, we leverage the semantic understanding of the diffusion denoising process and drive the process to adversarial subareas by perturbing the latent and unconditional embeddings at the last timestep. To mitigate the distribution shift that exposes a negative impact on image quality, we apply perturbation to the latent in frequency domain with the Fourier Transform. Experimental results demonstrate that AdvLogo achieves strong attack performance while maintaining high visual quality.

摘要: 随着深度学习的快速发展，目标检测器表现出了令人印象深刻的性能，但在某些场景下仍然存在漏洞。目前使用对抗性补丁探索漏洞的研究往往难以在攻击效率和视觉质量之间取得平衡。针对这一问题，我们从语义的角度提出了一种新的补丁攻击框架，称为AdvLogo。基于每个语义空间包含一个对抗性的子空间的假设，在这个子空间中，图像可能导致检测器无法识别目标，我们利用扩散去噪过程的语义理解，通过在最后一个时间步扰动潜在的和无条件的嵌入来驱动该过程到对抗性的子区域。为了减少分布漂移对图像质量的负面影响，我们利用傅里叶变换对频域中的潜伏点进行扰动。实验结果表明，AdvLogo在保持较高视觉质量的同时，具有较强的攻击性能。



## **25. Exploring Adversarial Robustness in Classification tasks using DNA Language Models**

使用DNA语言模型探索分类任务中的对抗鲁棒性 cs.CL

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.19788v2) [paper-pdf](http://arxiv.org/pdf/2409.19788v2)

**Authors**: Hyunwoo Yoo, Haebin Shin, Kaidi Xu, Gail Rosen

**Abstract**: DNA Language Models, such as GROVER, DNABERT2 and the Nucleotide Transformer, operate on DNA sequences that inherently contain sequencing errors, mutations, and laboratory-induced noise, which may significantly impact model performance. Despite the importance of this issue, the robustness of DNA language models remains largely underexplored. In this paper, we comprehensivly investigate their robustness in DNA classification by applying various adversarial attack strategies: the character (nucleotide substitutions), word (codon modifications), and sentence levels (back-translation-based transformations) to systematically analyze model vulnerabilities. Our results demonstrate that DNA language models are highly susceptible to adversarial attacks, leading to significant performance degradation. Furthermore, we explore adversarial training method as a defense mechanism, which enhances both robustness and classification accuracy. This study highlights the limitations of DNA language models and underscores the necessity of robustness in bioinformatics.

摘要: DNA语言模型，如Grover、DNABERT2和核苷酸转换器，对DNA序列进行操作，这些序列固有地包含测序错误、突变和实验室诱导的噪声，这些可能会显著影响模型的性能。尽管这个问题很重要，但DNA语言模型的稳健性在很大程度上仍然没有得到充分的研究。在本文中，我们通过应用各种对抗性攻击策略：字符(核苷酸替换)、单词(密码子修改)和句子级别(基于反向翻译的转换)来系统地分析模型的脆弱性，全面地研究了它们在DNA分类中的稳健性。我们的结果表明，DNA语言模型非常容易受到对抗性攻击，导致性能显著下降。此外，我们还探索了对抗性训练方法作为一种防御机制，提高了鲁棒性和分类准确率。这项研究突出了DNA语言模型的局限性，并强调了生物信息学中稳健性的必要性。



## **26. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

MAA：针对视觉语言预训练模型的强力对抗攻击 cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2502.08079v3) [paper-pdf](http://arxiv.org/pdf/2502.08079v3)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.

摘要: 当前用于评估视觉语言预训练(VLP)模型在多模式任务中的稳健性的对抗性攻击存在可转移性有限的问题，其中针对特定模型的攻击往往难以在不同的模型上有效地泛化，从而限制了它们在更广泛地评估稳健性方面的有效性。这主要归因于过度依赖特定型号的特征和区域，特别是在图像模式方面。在本文中，我们提出了一种优雅而高效的方法，称为精细攻击(MAA)，它充分利用了个体样本的模型无关特性和脆弱性，从而增强了泛化能力，降低了模型依赖。MAA通过开发一种新的调整大小和滑动裁剪(RSCrop)技术，结合多粒度相似破坏(MGSD)策略，强调对抗性图像的细粒度优化。在不同的VLP模型、多个基准数据集和各种下游任务上的广泛实验表明，MAA显著增强了对抗性攻击的有效性和可转移性。我们进行了大量的性能研究，以深入了解各种型号配置的有效性，从而指导该领域的未来发展。



## **27. Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation**

$\ell_\infty$-扰动下对抗训练估计的渐进行为 math.ST

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2401.15262v2) [paper-pdf](http://arxiv.org/pdf/2401.15262v2)

**Authors**: Yiling Xie, Xiaoming Huo

**Abstract**: Adversarial training has been proposed to protect machine learning models against adversarial attacks. This paper focuses on adversarial training under $\ell_\infty$-perturbation, which has recently attracted much research attention. The asymptotic behavior of the adversarial training estimator is investigated in the generalized linear model. The results imply that the asymptotic distribution of the adversarial training estimator under $\ell_\infty$-perturbation could put a positive probability mass at $0$ when the true parameter is $0$, providing a theoretical guarantee of the associated sparsity-recovery ability. Alternatively, a two-step procedure is proposed -- adaptive adversarial training, which could further improve the performance of adversarial training under $\ell_\infty$-perturbation. Specifically, the proposed procedure could achieve asymptotic variable-selection consistency and unbiasedness. Numerical experiments are conducted to show the sparsity-recovery ability of adversarial training under $\ell_\infty$-perturbation and to compare the empirical performance between classic adversarial training and adaptive adversarial training.

摘要: 对抗性训练已被提出用来保护机器学习模型免受对抗性攻击。本文主要研究近年来备受关注的对抗性训练问题。研究了广义线性模型中对抗性训练估计器的渐近行为。结果表明，当真参数为$0时，当真参数为$0时，对抗性训练估计量的渐近分布在$0-扰动下可以使正概率质量为$0，从而为相关稀疏恢复能力提供了理论保证。或者，提出了一种分两步进行的训练方法--自适应对抗性训练，它可以进一步提高对抗性训练在干扰下的性能。具体地说，该方法可以实现变量选择的渐近一致性和无偏性。通过数值实验验证了对抗性训练在扰动下的稀疏性恢复能力，并比较了经典对抗性训练和自适应对抗性训练的经验性能。



## **28. Exploiting Vulnerabilities in Speech Translation Systems through Targeted Adversarial Attacks**

通过定向对抗攻击利用语音翻译系统中的漏洞 cs.SD

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2503.00957v1) [paper-pdf](http://arxiv.org/pdf/2503.00957v1)

**Authors**: Chang Liu, Haolin Wu, Xi Yang, Kui Zhang, Cong Wu, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: As speech translation (ST) systems become increasingly prevalent, understanding their vulnerabilities is crucial for ensuring robust and reliable communication. However, limited work has explored this issue in depth. This paper explores methods of compromising these systems through imperceptible audio manipulations. Specifically, we present two innovative approaches: (1) the injection of perturbation into source audio, and (2) the generation of adversarial music designed to guide targeted translation, while also conducting more practical over-the-air attacks in the physical world. Our experiments reveal that carefully crafted audio perturbations can mislead translation models to produce targeted, harmful outputs, while adversarial music achieve this goal more covertly, exploiting the natural imperceptibility of music. These attacks prove effective across multiple languages and translation models, highlighting a systemic vulnerability in current ST architectures. The implications of this research extend beyond immediate security concerns, shedding light on the interpretability and robustness of neural speech processing systems. Our findings underscore the need for advanced defense mechanisms and more resilient architectures in the realm of audio systems. More details and samples can be found at https://adv-st.github.io.

摘要: 随着语音翻译(ST)系统变得越来越普遍，了解它们的漏洞对于确保健壮可靠的通信至关重要。然而，深入探讨这一问题的工作有限。本文探索了通过不可感知的音频操作来危害这些系统的方法。具体地说，我们提出了两种创新的方法：(1)在源音频中注入扰动，(2)生成对抗性音乐，旨在指导定向翻译，同时也在物理世界中进行更实际的空中攻击。我们的实验表明，精心制作的音频扰动可能会误导翻译模型产生目标明确的有害输出，而对抗性音乐则更隐蔽地实现这一目标，利用音乐的自然不可感知性。事实证明，这些攻击在多种语言和翻译模型中都有效，突显了当前ST架构中的系统性漏洞。这项研究的意义超越了直接的安全问题，揭示了神经语音处理系统的可解释性和稳健性。我们的发现强调了音频系统领域需要先进的防御机制和更具弹性的架构。欲了解更多详情和样品，请访问https://adv-st.github.io.。



## **29. Improving the Transferability of Adversarial Attacks by an Input Transpose**

通过输入转置提高对抗性攻击的可转移性 cs.CV

15 pages, 11 figures

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2503.00932v1) [paper-pdf](http://arxiv.org/pdf/2503.00932v1)

**Authors**: Qing Wan, Shilong Deng, Xun Wang

**Abstract**: Deep neural networks (DNNs) are highly susceptible to adversarial examples--subtle perturbations applied to inputs that are often imperceptible to humans yet lead to incorrect model predictions. In black-box scenarios, however, existing adversarial examples exhibit limited transferability and struggle to effectively compromise multiple unseen DNN models. Previous strategies enhance the cross-model generalization of adversarial examples by introducing versatility into adversarial perturbations, thereby improving transferability. However, further refining perturbation versatility often demands intricate algorithm development and substantial computation consumption. In this work, we propose an input transpose method that requires almost no additional labor and computation costs but can significantly improve the transferability of existing adversarial strategies. Even without adding adversarial perturbations, our method demonstrates considerable effectiveness in cross-model attacks. Our exploration finds that on specific datasets, a mere $1^\circ$ left or right rotation might be sufficient for most adversarial examples to deceive unseen models. Our further analysis suggests that this transferability improvement triggered by rotating only $1^\circ$ may stem from visible pattern shifts in the DNN's low-level feature maps. Moreover, this transferability exhibits optimal angles that, when identified under unrestricted query conditions, could potentially yield even greater performance.

摘要: 深度神经网络(DNN)非常容易受到敌意示例的影响--对输入进行微妙的扰动，这些输入通常是人类无法察觉的，但会导致错误的模型预测。然而，在黑盒场景中，现有的对抗性例子表现出有限的可转移性，并且难以有效地折衷多个看不见的DNN模型。以前的策略通过在对抗性扰动中引入通用性来增强对抗性实例的跨模型泛化，从而提高了可转移性。然而，进一步提高扰动的通用性往往需要复杂的算法开发和大量的计算消耗。在这项工作中，我们提出了一种输入转置方法，该方法几乎不需要额外的人力和计算代价，但可以显著提高现有对抗性策略的可转移性。即使在不增加对抗性扰动的情况下，我们的方法在跨模型攻击中也表现出相当的有效性。我们的探索发现，在特定的数据集上，仅仅1美元的向左或向右旋转可能就足以让大多数对抗性例子欺骗看不见的模型。我们的进一步分析表明，仅循环$1^\circ$所引发的可转移性改善可能源于DNN低层特征映射中的可见模式变化。此外，这种可转移性展示了最佳角度，当在不受限制的查询条件下确定时，可能会潜在地产生更好的性能。



## **30. AMUN: Adversarial Machine UNlearning**

AMUN：对抗性机器取消学习 cs.LG

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2503.00917v1) [paper-pdf](http://arxiv.org/pdf/2503.00917v1)

**Authors**: Ali Ebrahimpour-Boroojeny, Hari Sundaram, Varun Chandrasekaran

**Abstract**: Machine unlearning, where users can request the deletion of a forget dataset, is becoming increasingly important because of numerous privacy regulations. Initial works on ``exact'' unlearning (e.g., retraining) incur large computational overheads. However, while computationally inexpensive, ``approximate'' methods have fallen short of reaching the effectiveness of exact unlearning: models produced fail to obtain comparable accuracy and prediction confidence on both the forget and test (i.e., unseen) dataset. Exploiting this observation, we propose a new unlearning method, Adversarial Machine UNlearning (AMUN), that outperforms prior state-of-the-art (SOTA) methods for image classification. AMUN lowers the confidence of the model on the forget samples by fine-tuning the model on their corresponding adversarial examples. Adversarial examples naturally belong to the distribution imposed by the model on the input space; fine-tuning the model on the adversarial examples closest to the corresponding forget samples (a) localizes the changes to the decision boundary of the model around each forget sample and (b) avoids drastic changes to the global behavior of the model, thereby preserving the model's accuracy on test samples. Using AMUN for unlearning a random $10\%$ of CIFAR-10 samples, we observe that even SOTA membership inference attacks cannot do better than random guessing.

摘要: 机器遗忘--用户可以请求删除忘记数据集--正变得越来越重要，因为有许多隐私法规。关于“精确”遗忘的初始工作(例如，再培训)会产生很大的计算开销。然而，虽然计算成本不高，但“近似”方法无法达到精确遗忘的有效性：所产生的模型无法在忘记和测试(即看不见)的数据集上获得类似的精度和预测置信度。利用这一观察结果，我们提出了一种新的遗忘方法，对抗性机器遗忘(AMUN)，其性能优于现有的图像分类方法(SOTA)。Amun通过在相应的对抗性样本上微调模型来降低模型在忘记样本上的置信度。对抗性样本自然属于模型在输入空间上强加的分布；在最接近相应遗忘样本的对抗性样本上微调模型：(A)在每个遗忘样本周围定位对模型的决策边界的改变；(B)避免对模型的全局行为进行剧烈改变，从而保持模型在测试样本上的准确性。使用AMUN对随机的CIFAR-10样本进行遗忘，我们观察到即使是SOTA成员推理攻击也不能比随机猜测做得更好。



## **31. DEAL: Data-Efficient Adversarial Learning for High-Quality Infrared Imaging**

DEAL：高质量红外成像的数据高效对抗学习 cs.CV

The source code will be available at  https://github.com/LiuZhu-CV/DEAL

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2503.00905v1) [paper-pdf](http://arxiv.org/pdf/2503.00905v1)

**Authors**: Zhu Liu, Zijun Wang, Jinyuan Liu, Fanqi Meng, Long Ma, Risheng Liu

**Abstract**: Thermal imaging is often compromised by dynamic, complex degradations caused by hardware limitations and unpredictable environmental factors. The scarcity of high-quality infrared data, coupled with the challenges of dynamic, intricate degradations, makes it difficult to recover details using existing methods. In this paper, we introduce thermal degradation simulation integrated into the training process via a mini-max optimization, by modeling these degraded factors as adversarial attacks on thermal images. The simulation is dynamic to maximize objective functions, thus capturing a broad spectrum of degraded data distributions. This approach enables training with limited data, thereby improving model performance.Additionally, we introduce a dual-interaction network that combines the benefits of spiking neural networks with scale transformation to capture degraded features with sharp spike signal intensities. This architecture ensures compact model parameters while preserving efficient feature representation. Extensive experiments demonstrate that our method not only achieves superior visual quality under diverse single and composited degradation, but also delivers a significant reduction in processing when trained on only fifty clear images, outperforming existing techniques in efficiency and accuracy. The source code will be available at https://github.com/LiuZhu-CV/DEAL.

摘要: 由于硬件限制和不可预测的环境因素，热成像经常受到动态、复杂的退化的影响。高质量红外数据的缺乏，加上动态、复杂的退化挑战，使得使用现有方法恢复细节变得困难。在本文中，我们通过极小极大优化将热退化模拟集成到训练过程中，通过将这些退化因素建模为对热图像的敌意攻击。模拟是动态的，以最大化目标函数，从而捕获广泛的降级数据分布。该方法可以用有限的数据进行训练，从而提高模型的性能。此外，我们还引入了一个双重交互网络，它结合了尖峰神经网络和尺度变换的优点来捕获具有尖锐尖峰信号强度的退化特征。该体系结构在保证模型参数紧凑的同时，保持了有效的特征表示。大量的实验表明，我们的方法不仅在不同的单一和复合退化情况下获得了优异的视觉质量，而且当仅对50幅清晰图像进行训练时，处理速度显著减少，在效率和准确性方面优于现有的技术。源代码将在https://github.com/LiuZhu-CV/DEAL.上提供



## **32. Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives**

从时间和形态角度重新思考视听对抗脆弱性 cs.SD

Accepted by ICLR 2025

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2502.11858v3) [paper-pdf](http://arxiv.org/pdf/2502.11858v3)

**Authors**: Zeliang Zhang, Susan Liang, Daiki Shimada, Chenliang Xu

**Abstract**: While audio-visual learning equips models with a richer understanding of the real world by leveraging multiple sensory modalities, this integration also introduces new vulnerabilities to adversarial attacks.   In this paper, we present a comprehensive study of the adversarial robustness of audio-visual models, considering both temporal and modality-specific vulnerabilities. We propose two powerful adversarial attacks: 1) a temporal invariance attack that exploits the inherent temporal redundancy across consecutive time segments and 2) a modality misalignment attack that introduces incongruence between the audio and visual modalities. These attacks are designed to thoroughly assess the robustness of audio-visual models against diverse threats. Furthermore, to defend against such attacks, we introduce a novel audio-visual adversarial training framework. This framework addresses key challenges in vanilla adversarial training by incorporating efficient adversarial perturbation crafting tailored to multi-modal data and an adversarial curriculum strategy. Extensive experiments in the Kinetics-Sounds dataset demonstrate that our proposed temporal and modality-based attacks in degrading model performance can achieve state-of-the-art performance, while our adversarial training defense largely improves the adversarial robustness as well as the adversarial training efficiency.

摘要: 虽然视听学习通过利用多种感官模式使模型对真实世界有了更丰富的理解，但这种集成也引入了新的易受对手攻击的漏洞。在本文中，我们对视听模型的对抗健壮性进行了全面的研究，同时考虑了时间和通道特定的脆弱性。我们提出了两个强大的对抗性攻击：1)利用连续时间段固有的时间冗余性的时间不变性攻击；2)引入视听通道不一致的通道失准攻击。这些攻击旨在彻底评估视听模型对各种威胁的稳健性。此外，为了防御此类攻击，我们引入了一种新的视听对抗性训练框架。这一框架通过结合为多模式数据量身定做的高效对抗性扰动制作和对抗性课程战略，解决了普通对抗性训练中的关键挑战。在Kinetics-Sound数据集上的大量实验表明，我们提出的基于时间和通道的攻击在降低模型性能的同时可以获得最先进的性能，而我们的对抗性训练防御在很大程度上提高了对抗性的健壮性和对抗性训练的效率。



## **33. MOVE: Effective and Harmless Ownership Verification via Embedded External Features**

MOVE：通过嵌入式外部功能进行有效且无害的所有权验证 cs.CR

This paper has been accepted by IEEE TPAMI 2025. It is the journal  extension of our conference paper in AAAI 2022  (https://ojs.aaai.org/index.php/AAAI/article/view/20036). 18 pages

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2208.02820v2) [paper-pdf](http://arxiv.org/pdf/2208.02820v2)

**Authors**: Yiming Li, Linghui Zhu, Xiaojun Jia, Yang Bai, Yong Jiang, Shu-Tao Xia, Xiaochun Cao, Kui Ren

**Abstract**: Currently, deep neural networks (DNNs) are widely adopted in different applications. Despite its commercial values, training a well-performing DNN is resource-consuming. Accordingly, the well-trained model is valuable intellectual property for its owner. However, recent studies revealed the threats of model stealing, where the adversaries can obtain a function-similar copy of the victim model, even when they can only query the model. In this paper, we propose an effective and harmless model ownership verification (MOVE) to defend against different types of model stealing simultaneously, without introducing new security risks. In general, we conduct the ownership verification by verifying whether a suspicious model contains the knowledge of defender-specified external features. Specifically, we embed the external features by modifying a few training samples with style transfer. We then train a meta-classifier to determine whether a model is stolen from the victim. This approach is inspired by the understanding that the stolen models should contain the knowledge of features learned by the victim model. In particular, \revision{we develop our MOVE method under both white-box and black-box settings and analyze its theoretical foundation to provide comprehensive model protection.} Extensive experiments on benchmark datasets verify the effectiveness of our method and its resistance to potential adaptive attacks. The codes for reproducing the main experiments of our method are available at https://github.com/THUYimingLi/MOVE.

摘要: 目前，深度神经网络(DNN)被广泛应用于不同的领域。尽管DNN具有商业价值，但培训一名表现良好的DNN是一种资源消耗。因此，训练有素的车型对其所有者来说是宝贵的知识产权。然而，最近的研究揭示了模型窃取的威胁，其中攻击者可以获得与受害者模型功能相似的副本，即使他们只能查询模型。在本文中，我们提出了一种有效且无害的模型所有权验证(MOVE)来同时防御不同类型的模型窃取，而不会引入新的安全风险。一般来说，我们通过验证可疑模型是否包含防御者指定的外部特征的知识来进行所有权验证。具体地说，我们通过使用样式转移修改一些训练样本来嵌入外部特征。然后，我们训练元分类器来确定模型是否从受害者那里被盗。这种方法的灵感来自于这样一种理解，即被盗模型应该包含受害者模型学习的特征知识。特别是，我们在白盒和黑盒环境下发展了我们的Move方法，并分析了它的理论基础，以提供全面的模型保护。}在基准数据集上的大量实验验证了我们方法的有效性和对潜在自适应攻击的抵抗力。重现我们方法的主要实验的代码可在https://github.com/THUYimingLi/MOVE.上找到



## **34. Boosting Jailbreak Attack with Momentum**

以势头助推越狱攻击 cs.LG

Accepted by ICASSP 2025

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2405.01229v2) [paper-pdf](http://arxiv.org/pdf/2405.01229v2)

**Authors**: Yihao Zhang, Zeming Wei

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, yet they remain vulnerable to adversarial attacks, notably the well-known jailbreak attack. In particular, the Greedy Coordinate Gradient (GCG) attack has demonstrated efficacy in exploiting this vulnerability by optimizing adversarial prompts through a combination of gradient heuristics and greedy search. However, the efficiency of this attack has become a bottleneck in the attacking process. To mitigate this limitation, in this paper we rethink the generation of the adversarial prompts through an optimization lens, aiming to stabilize the optimization process and harness more heuristic insights from previous optimization iterations. Specifically, we propose the \textbf{M}omentum \textbf{A}ccelerated G\textbf{C}G (\textbf{MAC}) attack, which integrates a momentum term into the gradient heuristic to boost and stabilize the random search for tokens in adversarial prompts. Experimental results showcase the notable enhancement achieved by MAC over baselines in terms of attack success rate and optimization efficiency. Moreover, we demonstrate that MAC can still exhibit superior performance for transfer attacks and models under defense mechanisms. Our code is available at https://github.com/weizeming/momentum-attack-llm.

摘要: 大型语言模型(LLM)在不同的任务中取得了显著的成功，但它们仍然容易受到对手攻击，特别是众所周知的越狱攻击。特别是，贪婪坐标梯度(GCG)攻击已经证明了通过结合梯度启发式和贪婪搜索来优化敌意提示来利用该漏洞的有效性。然而，这种攻击的效率已经成为攻击过程中的瓶颈。为了缓解这一局限性，在本文中，我们通过优化镜头重新考虑敌意提示的生成，目的是稳定优化过程，并从先前的优化迭代中获得更多启发式的见解。具体地说，我们提出了加速G/Textbf{C}G(Textbf{MAC})攻击，该攻击将动量项融入到梯度启发式中，以增强和稳定敌意提示中随机搜索令牌的能力。实验结果表明，在攻击成功率和优化效率方面，MAC算法在攻击成功率和优化效率方面都有明显的提高。此外，我们还证明了在防御机制下，MAC对于传输攻击和模型仍然表现出优越的性能。我们的代码可以在https://github.com/weizeming/momentum-attack-llm.上找到



## **35. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

CLIPure：通过CLIP在潜空间中净化，以实现对抗鲁棒零镜头分类 cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2502.18176v2) [paper-pdf](http://arxiv.org/pdf/2502.18176v2)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.

摘要: 在这篇文章中，我们的目标是建立一个对抗性稳健的零镜头图像分类器。我们的工作基于CLIP，这是一个视觉语言预先训练的编码器模型，它可以通过将图像与文本提示进行匹配来执行零镜头分类。净化是我们选择的路径，因为它不需要针对特定攻击类型的对抗性训练，因此可以应对任何可预见的攻击。然后，我们通过双向随机微分方程(SDE)将净化风险表示为对敌方样本去噪的净化过程和对良性样本添加扰动的攻击过程的联合分布之间的KL发散。最终得出的结果启发我们去探索CLIP的多峰潜伏空间中的净化。我们为我们的CLIPure方法提出了两种变体：CLIPure-Diff和CLIPure-Cos，CLIPure-Diff使用DALE-2中的DiffusionPrior模块(对剪辑的潜在向量的生成过程进行建模)来模拟图像的潜在向量的可能性，CLIPure-Cos使用图像的嵌入和“a的照片”之间的余弦相似性来建模可能性。据我们所知，CLIPure是第一个在多峰潜在空间中进行净化的方法，而CLIPure-Cos是第一个不基于产生式模型的净化方法，大大提高了防御效率。我们在CIFAR-10、ImageNet和13个数据集上进行了广泛的实验，这些数据集是以前基于剪辑的防御方法用于评估零镜头分类稳健性的。结果表明，CLIPure在很大程度上提高了SOTA的健壮性，例如，在CIFAR10上从71.7%提高到91.1%，在ImageNet上从59.6%提高到72.6%，在13个数据集上的平均健壮性比以前的SOTA提高了108%。代码可在https://github.com/TMLResearchGroup-CAS/CLIPure.上获得



## **36. Output Length Effect on DeepSeek-R1's Safety in Forced Thinking**

输出长度对DeepSeek-R1在强迫思维中安全性的影响 cs.CL

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2503.01923v1) [paper-pdf](http://arxiv.org/pdf/2503.01923v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Victor Bian

**Abstract**: Large Language Models (LLMs) have demonstrated strong reasoning capabilities, but their safety under adversarial conditions remains a challenge. This study examines the impact of output length on the robustness of DeepSeek-R1, particularly in Forced Thinking scenarios. We analyze responses across various adversarial prompts and find that while longer outputs can improve safety through self-correction, certain attack types exploit extended generations. Our findings suggest that output length should be dynamically controlled to balance reasoning effectiveness and security. We propose reinforcement learning-based policy adjustments and adaptive token length regulation to enhance LLM safety.

摘要: 大型语言模型（LLM）已表现出强大的推理能力，但它们在对抗条件下的安全性仍然是一个挑战。本研究考察了输出长度对DeepSeek-R1稳健性的影响，特别是在强迫思维场景中。我们分析了各种对抗提示的响应，发现虽然更长的输出可以通过自我纠正来提高安全性，但某些攻击类型会利用延长的世代。我们的研究结果表明，应该动态控制输出长度，以平衡推理有效性和安全性。我们提出基于强化学习的政策调整和自适应代币长度监管，以增强LLM安全性。



## **37. Unmasking Digital Falsehoods: A Comparative Analysis of LLM-Based Misinformation Detection Strategies**

揭露数字谎言：基于LLM的错误信息检测策略的比较分析 cs.CL

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2503.00724v1) [paper-pdf](http://arxiv.org/pdf/2503.00724v1)

**Authors**: Tianyi Huang, Jingyuan Yi, Peiyang Yu, Xiaochuan Xu

**Abstract**: The proliferation of misinformation on social media has raised significant societal concerns, necessitating robust detection mechanisms. Large Language Models such as GPT-4 and LLaMA2 have been envisioned as possible tools for detecting misinformation based on their advanced natural language understanding and reasoning capabilities. This paper conducts a comparison of LLM-based approaches to detecting misinformation between text-based, multimodal, and agentic approaches. We evaluate the effectiveness of fine-tuned models, zero-shot learning, and systematic fact-checking mechanisms in detecting misinformation across different topic domains like public health, politics, and finance. We also discuss scalability, generalizability, and explainability of the models and recognize key challenges such as hallucination, adversarial attacks on misinformation, and computational resources. Our findings point towards the importance of hybrid approaches that pair structured verification protocols with adaptive learning techniques to enhance detection accuracy and explainability. The paper closes by suggesting potential avenues of future work, including real-time tracking of misinformation, federated learning, and cross-platform detection models.

摘要: 社交媒体上虚假信息的泛滥引发了重大的社会担忧，需要强有力的检测机制。大型语言模型，如GPT-4和LLaMA2，已被设想为基于其先进的自然语言理解和推理能力而可能用于检测错误信息的工具。本文对基于LLM的错误信息检测方法在基于文本的方法、多通道方法和代理方法之间进行了比较。我们评估了微调模型、零距离学习和系统的事实核查机制在检测公共卫生、政治和金融等不同主题领域的错误信息方面的有效性。我们还讨论了模型的可伸缩性、通用性和可解释性，并认识到关键挑战，如幻觉、对错误信息的敌意攻击和计算资源。我们的发现指出了将结构化验证协议与自适应学习技术配对以提高检测准确性和可解释性的混合方法的重要性。论文最后提出了未来工作的潜在途径，包括错误信息的实时跟踪、联邦学习和跨平台检测模型。



## **38. ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech**

ASVspoof 5：使用众包语音设计、收集和验证用于欺骗、Deepfake和对抗性攻击检测的资源 eess.AS

Database link: https://zenodo.org/records/14498691, Database mirror  link: https://huggingface.co/datasets/jungjee/asvspoof5, ASVspoof 5 Challenge  Workshop Proceeding: https://www.isca-archive.org/asvspoof_2024/index.html

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2502.08857v3) [paper-pdf](http://arxiv.org/pdf/2502.08857v3)

**Authors**: Xin Wang, Héctor Delgado, Hemlata Tak, Jee-weon Jung, Hye-jin Shim, Massimiliano Todisco, Ivan Kukanov, Xuechen Liu, Md Sahidullah, Tomi Kinnunen, Nicholas Evans, Kong Aik Lee, Junichi Yamagishi, Myeonghun Jeong, Ge Zhu, Yongyi Zang, You Zhang, Soumi Maiti, Florian Lux, Nicolas Müller, Wangyou Zhang, Chengzhe Sun, Shuwei Hou, Siwei Lyu, Sébastien Le Maguer, Cheng Gong, Hanjie Guo, Liping Chen, Vishwanath Singh

**Abstract**: ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake attacks as well as the design of detection solutions. We introduce the ASVspoof 5 database which is generated in crowdsourced fashion from data collected in diverse acoustic conditions (cf. studio-quality data for earlier ASVspoof databases) and from ~2,000 speakers (cf. ~100 earlier). The database contains attacks generated with 32 different algorithms, also crowdsourced, and optimised to varying degrees using new surrogate detection models. Among them are attacks generated with a mix of legacy and contemporary text-to-speech synthesis and voice conversion models, in addition to adversarial attacks which are incorporated for the first time. ASVspoof 5 protocols comprise seven speaker-disjoint partitions. They include two distinct partitions for the training of different sets of attack models, two more for the development and evaluation of surrogate detection models, and then three additional partitions which comprise the ASVspoof 5 training, development and evaluation sets. An auxiliary set of data collected from an additional 30k speakers can also be used to train speaker encoders for the implementation of attack algorithms. Also described herein is an experimental validation of the new ASVspoof 5 database using a set of automatic speaker verification and spoof/deepfake baseline detectors. With the exception of protocols and tools for the generation of spoofed/deepfake speech, the resources described in this paper, already used by participants of the ASVspoof 5 challenge in 2024, are now all freely available to the community.

摘要: ASVspoof5是一系列挑战中的第五版，这些挑战促进了对语音欺骗和深度假冒攻击的研究以及检测解决方案的设计。我们介绍了ASVspoof5数据库，它是以众包方式从不同声学条件下收集的数据生成的(参见。较早ASVspoof数据库的演播室质量数据)和来自约2,000名演讲者的数据(参见~100早)。该数据库包含32种不同的算法产生的攻击，也是众包的，并使用新的代理检测模型在不同程度上进行了优化。其中包括使用传统和当代文本到语音合成和语音转换模型的混合生成的攻击，以及首次纳入的对抗性攻击。ASVspoof 5协议包括七个说话人不相交的分区。它们包括用于训练不同攻击模型集的两个不同的分区，用于开发和评估代理检测模型的另外两个分区，以及组成ASVspoof5训练、开发和评估集的另外三个分区。从另外30,000个说话者收集的辅助数据集也可用于训练说话人编码者以实现攻击算法。这里还描述了使用一组自动说话人验证和欺骗/深度伪基线检测器对新的ASVspoof5数据库进行的实验验证。除了用于生成欺骗/深度假语音的协议和工具外，本文描述的资源现在都可以免费向社区提供，这些资源已经被2024年ASVspoof 5挑战赛的参与者使用。



## **39. RAIFLE: Reconstruction Attacks on Interaction-based Federated Learning with Adversarial Data Manipulation**

RAIFLE：对具有对抗性数据操纵的基于交互的联邦学习的重建攻击 cs.CR

Published in NDSS 2025

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2310.19163v3) [paper-pdf](http://arxiv.org/pdf/2310.19163v3)

**Authors**: Dzung Pham, Shreyas Kulkarni, Amir Houmansadr

**Abstract**: Federated learning has emerged as a promising privacy-preserving solution for machine learning domains that rely on user interactions, particularly recommender systems and online learning to rank. While there has been substantial research on the privacy of traditional federated learning, little attention has been paid to the privacy properties of these interaction-based settings. In this work, we show that users face an elevated risk of having their private interactions reconstructed by the central server when the server can control the training features of the items that users interact with. We introduce RAIFLE, a novel optimization-based attack framework where the server actively manipulates the features of the items presented to users to increase the success rate of reconstruction. Our experiments with federated recommendation and online learning-to-rank scenarios demonstrate that RAIFLE is significantly more powerful than existing reconstruction attacks like gradient inversion, achieving high performance consistently in most settings. We discuss the pros and cons of several possible countermeasures to defend against RAIFLE in the context of interaction-based federated learning. Our code is open-sourced at https://github.com/dzungvpham/raifle.

摘要: 联合学习已经成为一种很有前途的隐私保护解决方案，适用于依赖用户交互的机器学习领域，特别是推荐系统和在线学习来排名。虽然已经有大量关于传统联合学习隐私的研究，但很少有人关注这些基于交互的设置的隐私属性。在这项工作中，我们表明，当中央服务器可以控制用户交互的项目的训练特征时，用户面临由中央服务器重建他们的私人交互的风险增加。我们引入了RAIFLE，一个新的基于优化的攻击框架，服务器主动操纵呈现给用户的项目的特征，以提高重建的成功率。我们在联邦推荐和在线学习排名场景下的实验表明，RAIFLE比现有的重建攻击(如梯度反转)要强大得多，在大多数情况下都能获得一致的高性能。我们讨论了在基于交互的联邦学习背景下防御RAIFLE的几种可能对策的利弊。我们的代码在https://github.com/dzungvpham/raifle.上是开源的



## **40. BadJudge: Backdoor Vulnerabilities of LLM-as-a-Judge**

BadJudge：法学硕士担任法官的后门漏洞 cs.CL

Published to ICLR 2025

**SubmitDate**: 2025-03-01    [abs](http://arxiv.org/abs/2503.00596v1) [paper-pdf](http://arxiv.org/pdf/2503.00596v1)

**Authors**: Terry Tong, Fei Wang, Zhe Zhao, Muhao Chen

**Abstract**: This paper proposes a novel backdoor threat attacking the LLM-as-a-Judge evaluation regime, where the adversary controls both the candidate and evaluator model. The backdoored evaluator victimizes benign users by unfairly assigning inflated scores to adversary. A trivial single token backdoor poisoning 1% of the evaluator training data triples the adversary's score with respect to their legitimate score. We systematically categorize levels of data access corresponding to three real-world settings, (1) web poisoning, (2) malicious annotator, and (3) weight poisoning. These regimes reflect a weak to strong escalation of data access that highly correlates with attack severity. Under the weakest assumptions - web poisoning (1), the adversary still induces a 20% score inflation. Likewise, in the (3) weight poisoning regime, the stronger assumptions enable the adversary to inflate their scores from 1.5/5 to 4.9/5. The backdoor threat generalizes across different evaluator architectures, trigger designs, evaluation tasks, and poisoning rates. By poisoning 10% of the evaluator training data, we control toxicity judges (Guardrails) to misclassify toxic prompts as non-toxic 89% of the time, and document reranker judges in RAG to rank the poisoned document first 97% of the time. LLM-as-a-Judge is uniquely positioned at the intersection of ethics and technology, where social implications of mislead model selection and evaluation constrain the available defensive tools. Amidst these challenges, model merging emerges as a principled tool to offset the backdoor, reducing ASR to near 0% whilst maintaining SOTA performance. Model merging's low computational cost and convenient integration into the current LLM Judge training pipeline position it as a promising avenue for backdoor mitigation in the LLM-as-a-Judge setting.

摘要: 本文提出了一种新的后门威胁，攻击LLM-as-a-Court评估机制，其中对手同时控制候选者和评估者模型。背道而驰的评估者通过不公平地将夸大的分数分配给对手，使善良的用户成为受害者。一个微不足道的单令牌后门毒化1%的评估者训练数据，就会使对手的分数相对于他们的合法分数增加三倍。我们系统地对对应于三个真实世界设置的数据访问级别进行分类，(1)网络中毒，(2)恶意注释器，和(3)体重中毒。这些制度反映了与攻击严重程度高度相关的数据访问从弱到强的升级。在最弱的假设下-网络中毒(1)，对手仍然会导致20%的分数膨胀。同样，在(3)重量中毒机制中，更强的假设使对手能够将他们的分数从1.5/5夸大到4.9/5。后门威胁在不同的评估器架构、触发器设计、评估任务和中毒率中普遍存在。通过毒化10%的评估员培训数据，我们控制毒性法官(护栏)在89%的时间内将有毒提示错误分类为无毒，并在97%的时间内控制文件重排法官将有毒文件排在第一位。LLM作为一名法官处于伦理和技术的交汇点，误导性模型选择和评估的社会影响限制了可用的防御工具。在这些挑战中，模型合并成为一种重要的工具来抵消后门，在保持SOTA性能的同时将ASR降低到接近0%。模型合并的低计算成本和方便地集成到当前的LLM法官培训管道中，使其成为LLM作为法官设置的后门缓解的一个有前途的途径。



## **41. Provably Reliable Conformal Prediction Sets in the Presence of Data Poisoning**

数据中毒情况下可证明可靠的保形预测集 cs.LG

Accepted at ICLR 2025 (Spotlight)

**SubmitDate**: 2025-03-01    [abs](http://arxiv.org/abs/2410.09878v3) [paper-pdf](http://arxiv.org/pdf/2410.09878v3)

**Authors**: Yan Scholten, Stephan Günnemann

**Abstract**: Conformal prediction provides model-agnostic and distribution-free uncertainty quantification through prediction sets that are guaranteed to include the ground truth with any user-specified probability. Yet, conformal prediction is not reliable under poisoning attacks where adversaries manipulate both training and calibration data, which can significantly alter prediction sets in practice. As a solution, we propose reliable prediction sets (RPS): the first efficient method for constructing conformal prediction sets with provable reliability guarantees under poisoning. To ensure reliability under training poisoning, we introduce smoothed score functions that reliably aggregate predictions of classifiers trained on distinct partitions of the training data. To ensure reliability under calibration poisoning, we construct multiple prediction sets, each calibrated on distinct subsets of the calibration data. We then aggregate them into a majority prediction set, which includes a class only if it appears in a majority of the individual sets. Both proposed aggregations mitigate the influence of datapoints in the training and calibration data on the final prediction set. We experimentally validate our approach on image classification tasks, achieving strong reliability while maintaining utility and preserving coverage on clean data. Overall, our approach represents an important step towards more trustworthy uncertainty quantification in the presence of data poisoning.

摘要: 保角预测通过预测集提供与模型无关和无分布的不确定性量化，这些预测集保证以任何用户指定的概率包括基本事实。然而，在中毒攻击下，保角预测是不可靠的，其中对手同时操纵训练和校准数据，这在实践中可能会显著改变预测集。作为解决方案，我们提出了可靠预测集(RPS)：在中毒情况下构造具有可证明可靠性保证的共形预测集的第一种有效方法。为了确保在训练中毒情况下的可靠性，我们引入了平滑得分函数，它可靠地聚合了在不同的训练数据分区上训练的分类器的预测。为了确保在校准中毒情况下的可靠性，我们构造了多个预测集，每个预测集都在校准数据的不同子集上进行校准。然后我们将它们聚集到一个多数预测集合中，该集合只包括一个类，当它出现在大多数单独的集合中时。这两种建议的聚合都减轻了训练和校准数据中的数据点对最终预测集的影响。我们在实验上验证了我们的方法在图像分类任务上的有效性，在保持实用性和对干净数据的覆盖率的同时实现了很强的可靠性。总体而言，我们的方法代表着在存在数据中毒的情况下朝着更可信的不确定性量化迈出的重要一步。



## **42. A Survey of Adversarial Defenses in Vision-based Systems: Categorization, Methods and Challenges**

基于视觉的系统中对抗性防御概览：分类、方法和挑战 cs.CV

**SubmitDate**: 2025-03-01    [abs](http://arxiv.org/abs/2503.00384v1) [paper-pdf](http://arxiv.org/pdf/2503.00384v1)

**Authors**: Nandish Chattopadhyay, Abdul Basit, Bassem Ouni, Muhammad Shafique

**Abstract**: Adversarial attacks have emerged as a major challenge to the trustworthy deployment of machine learning models, particularly in computer vision applications. These attacks have a varied level of potency and can be implemented in both white box and black box approaches. Practical attacks include methods to manipulate the physical world and enforce adversarial behaviour by the corresponding target neural network models. Multiple different approaches to mitigate different kinds of such attacks are available in the literature, each with their own advantages and limitations. In this survey, we present a comprehensive systematization of knowledge on adversarial defenses, focusing on two key computer vision tasks: image classification and object detection. We review the state-of-the-art adversarial defense techniques and categorize them for easier comparison. In addition, we provide a schematic representation of these categories within the context of the overall machine learning pipeline, facilitating clearer understanding and benchmarking of defenses. Furthermore, we map these defenses to the types of adversarial attacks and datasets where they are most effective, offering practical insights for researchers and practitioners. This study is necessary for understanding the scope of how the available defenses are able to address the adversarial threats, and their shortcomings as well, which is necessary for driving the research in this area in the most appropriate direction, with the aim of building trustworthy AI systems for regular practical use-cases.

摘要: 对抗性攻击已经成为机器学习模型可信部署的主要挑战，特别是在计算机视觉应用中。这些攻击具有不同程度的效力，并且可以在白盒和黑盒方法中实施。实际攻击包括通过相应的目标神经网络模型操纵物理世界和实施敌对行为的方法。文献中提供了多种不同的方法来缓解不同种类的此类攻击，每种方法都有各自的优点和局限性。在这篇综述中，我们提出了一个全面的系统化的对抗性防御知识，重点是两个关键的计算机视觉任务：图像分类和目标检测。我们回顾了最先进的对抗防御技术，并对它们进行了分类，以便更容易地进行比较。此外，我们在整个机器学习管道的背景下提供了这些类别的示意性表示，有助于更清楚地理解和确定防御的基准。此外，我们将这些防御映射到最有效的对抗性攻击类型和数据集，为研究人员和从业者提供实用的见解。这项研究对于了解现有防御系统如何能够应对对抗性威胁的范围以及它们的缺点是必要的，这对于推动这一领域的研究朝着最合适的方向发展是必要的，目的是为常规的实际用例建立值得信赖的人工智能系统。



## **43. Theoretical Insights in Model Inversion Robustness and Conditional Entropy Maximization for Collaborative Inference Systems**

协作推理系统模型倒置稳健性和条件熵最大化的理论见解 cs.LG

accepted by CVPR2025

**SubmitDate**: 2025-03-01    [abs](http://arxiv.org/abs/2503.00383v1) [paper-pdf](http://arxiv.org/pdf/2503.00383v1)

**Authors**: Song Xia, Yi Yu, Wenhan Yang, Meiwen Ding, Zhuo Chen, Lingyu Duan, Alex C. Kot, Xudong Jiang

**Abstract**: By locally encoding raw data into intermediate features, collaborative inference enables end users to leverage powerful deep learning models without exposure of sensitive raw data to cloud servers. However, recent studies have revealed that these intermediate features may not sufficiently preserve privacy, as information can be leaked and raw data can be reconstructed via model inversion attacks (MIAs). Obfuscation-based methods, such as noise corruption, adversarial representation learning, and information filters, enhance the inversion robustness by obfuscating the task-irrelevant redundancy empirically. However, methods for quantifying such redundancy remain elusive, and the explicit mathematical relation between this redundancy minimization and inversion robustness enhancement has not yet been established. To address that, this work first theoretically proves that the conditional entropy of inputs given intermediate features provides a guaranteed lower bound on the reconstruction mean square error (MSE) under any MIA. Then, we derive a differentiable and solvable measure for bounding this conditional entropy based on the Gaussian mixture estimation and propose a conditional entropy maximization (CEM) algorithm to enhance the inversion robustness. Experimental results on four datasets demonstrate the effectiveness and adaptability of our proposed CEM; without compromising feature utility and computing efficiency, plugging the proposed CEM into obfuscation-based defense mechanisms consistently boosts their inversion robustness, achieving average gains ranging from 12.9\% to 48.2\%. Code is available at \href{https://github.com/xiasong0501/CEM}{https://github.com/xiasong0501/CEM}.

摘要: 通过将原始数据本地编码为中间功能，协作推理使最终用户能够利用强大的深度学习模型，而不会将敏感的原始数据暴露给云服务器。然而，最近的研究表明，这些中间特征可能不足以保护隐私，因为信息可能被泄露，原始数据可以通过模型反转攻击(MIA)重建。基于混淆的方法，如噪声破坏、对抗性表征学习和信息过滤器，通过经验地混淆与任务无关的冗余来增强求逆的稳健性。然而，量化这种冗余度的方法仍然难以捉摸，而且这种冗余度最小化和反演稳健性增强之间的明确数学关系尚未建立。为了解决这一问题，本工作首先从理论上证明了给定中间特征的输入的条件熵在任何MIA下都提供了重构均方误差(MSE)的保证下界。然后，基于混合高斯估计，给出了条件熵的可微可解度量，并提出了一种条件熵最大化(CEM)算法以增强反演的稳健性。在四个数据集上的实验结果证明了该算法的有效性和自适应性，在不影响特征效用和计算效率的情况下，将该算法嵌入到基于混淆的防御机制中，持续增强了它们的反转稳健性，获得了12.9~48.2%的平均增益。代码可在\href{https://github.com/xiasong0501/CEM}{https://github.com/xiasong0501/CEM}.上找到



## **44. Adversarial Attacks on Event-Based Pedestrian Detectors: A Physical Approach**

对基于事件的行人探测器的对抗性攻击：物理方法 cs.CV

Accepted by AAAI 2025

**SubmitDate**: 2025-03-01    [abs](http://arxiv.org/abs/2503.00377v1) [paper-pdf](http://arxiv.org/pdf/2503.00377v1)

**Authors**: Guixu Lin, Muyao Niu, Qingtian Zhu, Zhengwei Yin, Zhuoxiao Li, Shengfeng He, Yinqiang Zheng

**Abstract**: Event cameras, known for their low latency and high dynamic range, show great potential in pedestrian detection applications. However, while recent research has primarily focused on improving detection accuracy, the robustness of event-based visual models against physical adversarial attacks has received limited attention. For example, adversarial physical objects, such as specific clothing patterns or accessories, can exploit inherent vulnerabilities in these systems, leading to misdetections or misclassifications. This study is the first to explore physical adversarial attacks on event-driven pedestrian detectors, specifically investigating whether certain clothing patterns worn by pedestrians can cause these detectors to fail, effectively rendering them unable to detect the person. To address this, we developed an end-to-end adversarial framework in the digital domain, framing the design of adversarial clothing textures as a 2D texture optimization problem. By crafting an effective adversarial loss function, the framework iteratively generates optimal textures through backpropagation. Our results demonstrate that the textures identified in the digital domain possess strong adversarial properties. Furthermore, we translated these digitally optimized textures into physical clothing and tested them in real-world scenarios, successfully demonstrating that the designed textures significantly degrade the performance of event-based pedestrian detection models. This work highlights the vulnerability of such models to physical adversarial attacks.

摘要: 事件摄像机以其低延迟和高动态范围而闻名，在行人检测应用中显示出巨大的潜力。然而，虽然最近的研究主要集中在提高检测精度上，但基于事件的视觉模型对物理对手攻击的稳健性却受到了有限的关注。例如，敌意物理对象，如特定的服装图案或配饰，可以利用这些系统中的固有漏洞，导致错误检测或错误分类。这项研究首次探索了对事件驱动的行人探测器的物理对抗性攻击，特别是调查了行人穿着的某些服装图案是否会导致这些探测器失效，从而有效地使它们无法检测到人。为了解决这一问题，我们在数字领域开发了一个端到端的对抗性框架，将对抗性服装纹理的设计框定为一个2D纹理优化问题。通过构造一个有效的对抗性损失函数，该框架通过反向传播迭代地生成最优纹理。我们的结果表明，在数字域识别的纹理具有很强的对抗性。此外，我们将这些经过数字优化的纹理转化为物理服装并在真实场景中进行测试，成功地证明了所设计的纹理显著降低了基于事件的行人检测模型的性能。这项工作突显了这种模型在物理对抗攻击中的脆弱性。



## **45. 1-Lipschitz Network Initialization for Certifiably Robust Classification Applications: A Decay Problem**

1-Lipschitz网络可证明鲁棒分类应用程序：一个衰变问题 cs.LG

12 pages, 8 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2503.00240v1) [paper-pdf](http://arxiv.org/pdf/2503.00240v1)

**Authors**: Marius F. R. Juston, William R. Norris, Dustin Nottage, Ahmet Soylemezoglu

**Abstract**: This paper discusses the weight parametrization of two standard 1-Lipschitz network structure methodologies, the Almost-Orthogonal-Layers (AOL) and the SDP-based Lipschitz Layers (SLL), and derives their impact on the initialization for deep 1-Lipschitz feedforward networks in addition to discussing underlying issues surrounding this initialization. These networks are mainly used in certifiably robust classification applications to combat adversarial attacks by limiting the effects of perturbations on the output classification result. An exact and an upper bound for the parameterized weight variance was calculated assuming a standard Normal distribution initialization; additionally, an upper bound was computed assuming a Generalized Normal Distribution, generalizing the proof for Uniform, Laplace, and Normal distribution weight initializations. It is demonstrated that the weight variance holds no bearing on the output variance distribution and that only the dimension of the weight matrices matters. Additionally, this paper demonstrates that the weight initialization always causes deep 1-Lipschitz networks to decay to zero.

摘要: 本文讨论了两种标准的1-Lipschitz网络结构方法--几乎正交层(AOL)和基于SDP的Lipschitz层(SLL)的权参数化方法，以及它们对深度1-Lipschitz前馈网络初始化的影响，并讨论了围绕这种初始化的基本问题。这些网络主要用于可证明稳健的分类应用，通过限制扰动对输出分类结果的影响来对抗敌意攻击。在假设标准正态分布初始化的情况下，计算了参数化权重方差的精确上界和上界；此外，在假设广义正态分布的情况下，计算了上界，从而推广了均匀、拉普拉斯和正态分布权重初始化的证明。证明了权方差对输出方差分布没有影响，只有权矩阵的维度才是重要的。此外，本文还证明了权值初始化总是导致深度1-Lipschitz网络衰减到零。



## **46. Safeguarding AI Agents: Developing and Analyzing Safety Architectures**

保护人工智能代理：开发和分析安全架构 cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2409.03793v3) [paper-pdf](http://arxiv.org/pdf/2409.03793v3)

**Authors**: Ishaan Domkundwar, Mukunda N S, Ishaan Bhola, Riddhik Kochhar

**Abstract**: AI agents, specifically powered by large language models, have demonstrated exceptional capabilities in various applications where precision and efficacy are necessary. However, these agents come with inherent risks, including the potential for unsafe or biased actions, vulnerability to adversarial attacks, lack of transparency, and tendency to generate hallucinations. As AI agents become more prevalent in critical sectors of the industry, the implementation of effective safety protocols becomes increasingly important. This paper addresses the critical need for safety measures in AI systems, especially ones that collaborate with human teams. We propose and evaluate three frameworks to enhance safety protocols in AI agent systems: an LLM-powered input-output filter, a safety agent integrated within the system, and a hierarchical delegation-based system with embedded safety checks. Our methodology involves implementing these frameworks and testing them against a set of unsafe agentic use cases, providing a comprehensive evaluation of their effectiveness in mitigating risks associated with AI agent deployment. We conclude that these frameworks can significantly strengthen the safety and security of AI agent systems, minimizing potential harmful actions or outputs. Our work contributes to the ongoing effort to create safe and reliable AI applications, particularly in automated operations, and provides a foundation for developing robust guardrails to ensure the responsible use of AI agents in real-world applications.

摘要: 人工智能代理，特别是由大型语言模型驱动的，在需要精确度和效率的各种应用中展示了非凡的能力。然而，这些代理伴随着固有的风险，包括潜在的不安全或有偏见的行动，易受对手攻击，缺乏透明度，以及产生幻觉的倾向。随着人工智能代理在该行业的关键部门变得越来越普遍，实施有效的安全协议变得越来越重要。本文讨论了人工智能系统中安全措施的迫切需要，特别是与人类团队协作的系统。我们提出并评估了三个框架来增强AI代理系统中的安全协议：LLM驱动的输入输出过滤器、集成在系统中的安全代理以及嵌入安全检查的基于分级委托的系统。我们的方法涉及实现这些框架并针对一组不安全的代理用例对它们进行测试，提供对它们在降低与AI代理部署相关的风险方面的有效性的全面评估。我们的结论是，这些框架可以显著加强AI代理系统的安全性和安全性，将潜在的有害行为或输出降至最低。我们的工作有助于持续努力创建安全可靠的人工智能应用程序，特别是在自动化操作中，并为开发强大的护栏提供基础，以确保在现实世界的应用程序中负责任地使用人工智能代理。



## **47. UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning**

UPora：通过动态劫持LLM代理自己的推理来对抗他们的统一红色团队框架 cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2503.01908v1) [paper-pdf](http://arxiv.org/pdf/2503.01908v1)

**Authors**: Jiawei Zhang, Shuang Yang, Bo Li

**Abstract**: Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for handling complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements also amplify the risks of adversarial attacks, particularly when LLM agents can access sensitive external functionalities. Moreover, because LLM agents engage in extensive reasoning or planning before executing final actions, manipulating them into performing targeted malicious actions or invoking specific tools remains a significant challenge. Consequently, directly embedding adversarial strings in malicious instructions or injecting malicious prompts into tool interactions has become less effective against modern LLM agents. In this work, we present UDora, a unified red teaming framework designed for LLM Agents that dynamically leverages the agent's own reasoning processes to compel it toward malicious behavior. Specifically, UDora first samples the model's reasoning for the given task, then automatically identifies multiple optimal positions within these reasoning traces to insert targeted perturbations. Subsequently, it uses the modified reasoning as the objective to optimize the adversarial strings. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets.

摘要: 配备了外部工具的大型语言模型(LLM)代理在处理网络购物、自动回复电子邮件和金融交易等复杂任务方面变得越来越强大。然而，这些进步也放大了对抗性攻击的风险，特别是当LLM特工可以访问敏感的外部功能时。此外，由于LLM代理在执行最终操作之前会进行广泛的推理或规划，因此操纵它们执行有针对性的恶意操作或调用特定工具仍然是一个重大挑战。因此，直接在恶意指令中嵌入敌意字符串或在工具交互中插入恶意提示已变得对现代LLM代理不那么有效。在这项工作中，我们提出了Udora，一个为LLM代理设计的统一的红色团队框架，它动态地利用代理自己的推理过程来迫使其走向恶意行为。具体地说，Udora首先对给定任务的模型推理进行采样，然后自动在这些推理轨迹中识别多个最佳位置，以插入有针对性的扰动。随后，以改进后的推理为目标，对对抗性字符串进行优化。通过反复应用此过程，LLM代理随后将被诱导执行指定的恶意操作或调用特定的恶意工具。与现有方法相比，我们的方法在三个LLM试剂数据集上表现出了更好的有效性。



## **48. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

引导对话动力学，增强抵御多回合越狱攻击的稳健性 cs.CL

28 pages, 10 figures, 7 tables

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2503.00187v1) [paper-pdf](http://arxiv.org/pdf/2503.00187v1)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are highly vulnerable to jailbreaking attacks, wherein adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off between safety and helpfulness under different multi-turn jailbreak methods. Our code is available at https://github.com/HanjiangHu/NBF-LLM .

摘要: 大型语言模型(LLM)非常容易受到越狱攻击，在越狱攻击中，敌意提示旨在引发有害的响应。虽然现有的防御系统通过检测和过滤不安全的输入有效地缓解了单回合攻击，但它们无法抵御利用多个交互中的上下文漂移的多回合越狱，逐渐导致LLM远离安全行为。为了应对这一挑战，我们提出了一种基于安全控制理论的安全转向框架，以确保多轮对话中的恒定安全。我们的方法使用状态空间表示来模拟与LLMS的对话，并引入了一种新的神经屏障函数(NBF)来主动检测和过滤从不断变化的上下文中出现的有害查询。我们的方法通过学习解释敌意查询的安全预测器，防止潜在的上下文漂移到越狱，在每一轮对话中实现不变的安全。在多个LLM下的广泛实验表明，我们的基于NBF的安全转向性能优于安全对齐基线，提供了更强大的防御多转弯越狱的同时，在不同的多转弯越狱方法下保持了安全和帮助之间的更好权衡。我们的代码可以在https://github.com/HanjiangHu/NBF-LLM上找到。



## **49. Enabling AutoML for Zero-Touch Network Security: Use-Case Driven Analysis**

启用AutoML实现零接触网络安全：案例驱动分析 cs.CR

Published in IEEE Transactions on Network and Service Management  (TNSM); Code is available at Github link:  https://github.com/Western-OC2-Lab/AutoML-and-Adversarial-Attack-Defense-for-Zero-Touch-Network-Security

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.21286v1) [paper-pdf](http://arxiv.org/pdf/2502.21286v1)

**Authors**: Li Yang, Mirna El Rajab, Abdallah Shami, Sami Muhaidat

**Abstract**: Zero-Touch Networks (ZTNs) represent a state-of-the-art paradigm shift towards fully automated and intelligent network management, enabling the automation and intelligence required to manage the complexity, scale, and dynamic nature of next-generation (6G) networks. ZTNs leverage Artificial Intelligence (AI) and Machine Learning (ML) to enhance operational efficiency, support intelligent decision-making, and ensure effective resource allocation. However, the implementation of ZTNs is subject to security challenges that need to be resolved to achieve their full potential. In particular, two critical challenges arise: the need for human expertise in developing AI/ML-based security mechanisms, and the threat of adversarial attacks targeting AI/ML models. In this survey paper, we provide a comprehensive review of current security issues in ZTNs, emphasizing the need for advanced AI/ML-based security mechanisms that require minimal human intervention and protect AI/ML models themselves. Furthermore, we explore the potential of Automated ML (AutoML) technologies in developing robust security solutions for ZTNs. Through case studies, we illustrate practical approaches to securing ZTNs against both conventional and AI/ML-specific threats, including the development of autonomous intrusion detection systems and strategies to combat Adversarial ML (AML) attacks. The paper concludes with a discussion of the future research directions for the development of ZTN security approaches.

摘要: 零接触网络(ZTN)代表着向全自动化和智能网络管理的最先进范式转变，实现了管理下一代(6G)网络的复杂性、可扩展性和动态化所需的自动化和智能。ZTN利用人工智能(AI)和机器学习(ML)来提高运营效率，支持智能决策，并确保有效的资源分配。然而，零增长网络的实施受到安全挑战的制约，需要解决这些挑战才能充分发挥其潜力。特别是，出现了两个关键挑战：在开发基于AI/ML的安全机制方面需要人类专业知识，以及针对AI/ML模型的对抗性攻击的威胁。在这篇调查论文中，我们对ZTN中当前的安全问题进行了全面的回顾，强调了需要先进的基于AI/ML的安全机制，这种机制需要最少的人为干预，并保护AI/ML模型本身。此外，我们还探讨了自动ML(AutoML)技术在为ZTNS开发健壮的安全解决方案方面的潜力。通过案例研究，我们展示了保护ZTN免受传统威胁和AI/ML特定威胁的实用方法，包括自主入侵检测系统的开发和对抗敌意ML(AML)攻击的策略。最后，对ZTN安全方法的未来研究方向进行了讨论。



## **50. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

Logicbreaks：理解基于规则的推理颠覆的框架 cs.AI

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2407.00075v5) [paper-pdf](http://arxiv.org/pdf/2407.00075v5)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.

摘要: 我们研究如何根据预算指定的规则颠覆大型语言模型（LLM）。我们首先将规则遵循形式化为命题Horn逻辑中的推理，这是一个数学系统，其中规则的形式为“如果$P$和$Q$，那么$R$”，对于某些命题$P$、$Q$和$R$。接下来，我们证明，尽管小型变压器可以忠实地遵循这些规则，但恶意制作的提示仍然会误导理论构建和从数据中学习的模型。此外，我们证明了LLM上的流行攻击算法可以找到对抗提示并诱导与我们的理论一致的注意力模式。我们新颖的基于逻辑的框架为在基于规则的环境中研究LLM提供了基础，从而能够对逻辑推理和越狱攻击等任务进行正式分析。



