# Latest Adversarial Attack Papers
**update at 2025-03-07 10:04:30**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. FSPGD: Rethinking Black-box Attacks on Semantic Segmentation**

FSPVD：重新思考对语义分割的黑匣子攻击 cs.CV

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2502.01262v2) [paper-pdf](http://arxiv.org/pdf/2502.01262v2)

**Authors**: Eun-Sol Park, MiSo Park, Seung Park, Yong-Goo Shin

**Abstract**: Transferability, the ability of adversarial examples crafted for one model to deceive other models, is crucial for black-box attacks. Despite advancements in attack methods for semantic segmentation, transferability remains limited, reducing their effectiveness in real-world applications. To address this, we introduce the Feature Similarity Projected Gradient Descent (FSPGD) attack, a novel black-box approach that enhances both attack performance and transferability. Unlike conventional segmentation attacks that rely on output predictions for gradient calculation, FSPGD computes gradients from intermediate layer features. Specifically, our method introduces a loss function that targets local information by comparing features between clean images and adversarial examples, while also disrupting contextual information by accounting for spatial relationships between objects. Experiments on Pascal VOC 2012 and Cityscapes datasets demonstrate that FSPGD achieves superior transferability and attack performance, establishing a new state-of-the-art benchmark. Code is available at https://github.com/KU-AIVS/FSPGD.

摘要: 可转移性，即为一个模型制作的敌意例子欺骗其他模型的能力，对黑盒攻击至关重要。尽管语义分割的攻击方法有所进步，但可转移性仍然有限，降低了它们在现实世界应用中的有效性。为了解决这一问题，我们引入了特征相似性投影梯度下降(FSPGD)攻击，这是一种新的黑盒方法，它同时提高了攻击性能和可转移性。与依赖输出预测进行梯度计算的传统分割攻击不同，FSPGD根据中间层特征计算梯度。具体地说，我们的方法引入了一个损失函数，该函数通过比较干净图像和敌意图像之间的特征来定位局部信息，同时还通过考虑对象之间的空间关系来破坏上下文信息。在Pascal VOC 2012和CITYSCAPES数据集上的实验表明，FSPGD实现了卓越的可转移性和攻击性能，建立了一个新的最先进的基准。代码可在https://github.com/KU-AIVS/FSPGD.上找到



## **2. OrbID: Identifying Orbcomm Satellite RF Fingerprints**

OrbID：识别OrbSYS卫星RF指纹 eess.SP

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.02118v2) [paper-pdf](http://arxiv.org/pdf/2503.02118v2)

**Authors**: Cédric Solenthaler, Joshua Smailes, Martin Strohmeier

**Abstract**: An increase in availability of Software Defined Radios (SDRs) has caused a dramatic shift in the threat landscape of legacy satellite systems, opening them up to easy spoofing attacks by low-budget adversaries. Physical-layer authentication methods can help improve the security of these systems by providing additional validation without modifying the space segment. This paper extends previous research on Radio Frequency Fingerprinting (RFF) of satellite communication to the Orbcomm satellite formation. The GPS and Iridium constellations are already well covered in prior research, but the feasibility of transferring techniques to other formations has not yet been examined, and raises previously undiscussed challenges.   In this paper, we collect a novel dataset containing 8992474 packets from the Orbcom satellite constellation using different SDRs and locations. We use this dataset to train RFF systems based on convolutional neural networks. We achieve an ROC AUC score of 0.53 when distinguishing different satellites within the constellation, and 0.98 when distinguishing legitimate satellites from SDRs in a spoofing scenario. We also demonstrate the possibility of mixing datasets using different SDRs in different physical locations.

摘要: 软件定义无线电(SDR)可用性的增加导致传统卫星系统的威胁格局发生了戏剧性的变化，使它们容易受到低预算对手的欺骗攻击。物理层身份验证方法可以在不修改空间段的情况下提供额外的验证，从而帮助提高这些系统的安全性。本文将以往卫星通信射频指纹识别的研究扩展到Orbcomm卫星编队。GPS和Iridium星座已经在以前的研究中得到了很好的覆盖，但将技术转移到其他编队的可行性还没有被审查，这提出了以前没有讨论过的挑战。在本文中，我们使用不同的SDR和不同的位置从Orbcom卫星星座收集了一个包含8992474个信息包的新数据集。我们使用这个数据集来训练基于卷积神经网络的RFF系统。当区分星座内的不同卫星时，我们的ROC AUC得分为0.53，在欺骗情况下区分合法卫星和SDR时，我们的ROC AUC得分为0.98。我们还演示了在不同的物理位置使用不同的SDR混合数据集的可能性。



## **3. Poisoning Bayesian Inference via Data Deletion and Replication**

通过数据删除和复制毒害Bayesian推理 stat.ML

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04480v1) [paper-pdf](http://arxiv.org/pdf/2503.04480v1)

**Authors**: Matthieu Carreau, Roi Naveiro, William N. Caballero

**Abstract**: Research in adversarial machine learning (AML) has shown that statistical models are vulnerable to maliciously altered data. However, despite advances in Bayesian machine learning models, most AML research remains concentrated on classical techniques. Therefore, we focus on extending the white-box model poisoning paradigm to attack generic Bayesian inference, highlighting its vulnerability in adversarial contexts. A suite of attacks are developed that allow an attacker to steer the Bayesian posterior toward a target distribution through the strategic deletion and replication of true observations, even when only sampling access to the posterior is available. Analytic properties of these algorithms are proven and their performance is empirically examined in both synthetic and real-world scenarios. With relatively little effort, the attacker is able to substantively alter the Bayesian's beliefs and, by accepting more risk, they can mold these beliefs to their will. By carefully constructing the adversarial posterior, surgical poisoning is achieved such that only targeted inferences are corrupted and others are minimally disturbed.

摘要: 对抗性机器学习(AML)的研究表明，统计模型很容易受到恶意篡改数据的影响。然而，尽管贝叶斯机器学习模型取得了进展，但大多数AML研究仍然集中在经典技术上。因此，我们专注于扩展白盒模型中毒范例来攻击通用贝叶斯推理，强调其在对抗性环境中的脆弱性。开发了一套攻击，允许攻击者通过战略性删除和复制真实观测来引导贝叶斯后验向目标分布，即使在只有对后验的采样访问可用的情况下也是如此。证明了这些算法的分析性质，并在合成场景和真实世界场景中对它们的性能进行了经验检验。攻击者只需相对较少的努力，就能够实质性地改变贝叶斯人的信念，通过接受更多的风险，他们可以按照自己的意愿塑造这些信念。通过仔细构建对抗性后验，外科中毒被实现，使得只有有针对性的推理被破坏，而其他推理被最小限度地干扰。



## **4. Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges**

了解你的法官：LLM安全法官的稳健性元评估 cs.LG

Accepted to the ICBINB Workshop at ICLR'25

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04474v1) [paper-pdf](http://arxiv.org/pdf/2503.04474v1)

**Authors**: Francisco Eiras, Eliott Zemour, Eric Lin, Vaikkunth Mugunthan

**Abstract**: Large Language Model (LLM) based judges form the underpinnings of key safety evaluation processes such as offline benchmarking, automated red-teaming, and online guardrailing. This widespread requirement raises the crucial question: can we trust the evaluations of these evaluators? In this paper, we highlight two critical challenges that are typically overlooked: (i) evaluations in the wild where factors like prompt sensitivity and distribution shifts can affect performance and (ii) adversarial attacks that target the judge. We highlight the importance of these through a study of commonly used safety judges, showing that small changes such as the style of the model output can lead to jumps of up to 0.24 in the false negative rate on the same dataset, whereas adversarial attacks on the model generation can fool some judges into misclassifying 100% of harmful generations as safe ones. These findings reveal gaps in commonly used meta-evaluation benchmarks and weaknesses in the robustness of current LLM judges, indicating that low attack success under certain judges could create a false sense of security.

摘要: 基于大型语言模型(LLM)的评委构成了关键安全评估流程的基础，如离线基准、自动红色团队和在线护栏。这一普遍的要求提出了一个关键问题：我们能相信这些评估者的评价吗？在这篇文章中，我们强调了两个通常被忽视的关键挑战：(I)在野外评估中，敏感度和分布变化等因素会影响绩效；(Ii)针对法官的对抗性攻击。我们通过对常用安全法官的研究来强调这些的重要性，表明微小的变化，如模型输出的风格，可以导致同一数据集上的假阴性率跃升高达0.24，而对模型生成的对抗性攻击可能会欺骗一些法官，将100%的有害世代错误分类为安全世代。这些发现揭示了常用元评估基准的差距和当前LLM法官稳健性方面的弱点，表明在某些法官的领导下，低攻击成功率可能会产生一种错误的安全感。



## **5. Privacy Preserving and Robust Aggregation for Cross-Silo Federated Learning in Non-IID Settings**

非IID设置中跨筒仓联邦学习的隐私保护和鲁棒聚合 cs.LG

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04451v1) [paper-pdf](http://arxiv.org/pdf/2503.04451v1)

**Authors**: Marco Arazzi, Mert Cihangiroglu, Antonino Nocera

**Abstract**: Federated Averaging remains the most widely used aggregation strategy in federated learning due to its simplicity and scalability. However, its performance degrades significantly in non-IID data settings, where client distributions are highly imbalanced or skewed. Additionally, it relies on clients transmitting metadata, specifically the number of training samples, which introduces privacy risks and may conflict with regulatory frameworks like the European GDPR. In this paper, we propose a novel aggregation strategy that addresses these challenges by introducing class-aware gradient masking. Unlike traditional approaches, our method relies solely on gradient updates, eliminating the need for any additional client metadata, thereby enhancing privacy protection. Furthermore, our approach validates and dynamically weights client contributions based on class-specific importance, ensuring robustness against non-IID distributions, convergence prevention, and backdoor attacks. Extensive experiments on benchmark datasets demonstrate that our method not only outperforms FedAvg and other widely accepted aggregation strategies in non-IID settings but also preserves model integrity in adversarial scenarios. Our results establish the effectiveness of gradient masking as a practical and secure solution for federated learning.

摘要: 由于其简单性和可伸缩性，联合平均仍然是联合学习中使用最广泛的聚合策略。然而，在客户端分布高度不平衡或不对称的非IID数据设置中，其性能会显著降低。此外，它依赖于客户传输元数据，特别是训练样本的数量，这会带来隐私风险，并可能与欧洲GDPR等监管框架相冲突。在本文中，我们提出了一种新的聚合策略，通过引入类感知梯度掩码来解决这些挑战。与传统方法不同，我们的方法完全依赖于梯度更新，不需要任何额外的客户端元数据，从而增强了隐私保护。此外，我们的方法基于特定于类的重要性来验证和动态加权客户端贡献，从而确保对非IID分发、收敛防止和后门攻击的健壮性。在基准数据集上的大量实验表明，我们的方法不仅在非IID环境下优于FedAvg和其他被广泛接受的聚集策略，而且在对抗性场景下保持了模型的完整性。我们的结果证明了梯度掩蔽作为联邦学习的一种实用和安全的解决方案的有效性。



## **6. Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients**

通过打破隐形代理对尖峰神经网络进行有效且稀疏的对抗攻击 cs.CV

Accepted by CVPR 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.03272v2) [paper-pdf](http://arxiv.org/pdf/2503.03272v2)

**Authors**: Li Lun, Kunyu Feng, Qinglong Ni, Ling Liang, Yuan Wang, Ying Li, Dunshan Yu, Xiaoxin Cui

**Abstract**: Spiking neural networks (SNNs) have shown their competence in handling spatial-temporal event-based data with low energy consumption. Similar to conventional artificial neural networks (ANNs), SNNs are also vulnerable to gradient-based adversarial attacks, wherein gradients are calculated by spatial-temporal back-propagation (STBP) and surrogate gradients (SGs). However, the SGs may be invisible for an inference-only model as they do not influence the inference results, and current gradient-based attacks are ineffective for binary dynamic images captured by the dynamic vision sensor (DVS). While some approaches addressed the issue of invisible SGs through universal SGs, their SGs lack a correlation with the victim model, resulting in sub-optimal performance. Moreover, the imperceptibility of existing SNN-based binary attacks is still insufficient. In this paper, we introduce an innovative potential-dependent surrogate gradient (PDSG) method to establish a robust connection between the SG and the model, thereby enhancing the adaptability of adversarial attacks across various models with invisible SGs. Additionally, we propose the sparse dynamic attack (SDA) to effectively attack binary dynamic images. Utilizing a generation-reduction paradigm, SDA can fully optimize the sparsity of adversarial perturbations. Experimental results demonstrate that our PDSG and SDA outperform state-of-the-art SNN-based attacks across various models and datasets. Specifically, our PDSG achieves 100% attack success rate on ImageNet, and our SDA obtains 82% attack success rate by modifying only 0.24% of the pixels on CIFAR10DVS. The code is available at https://github.com/ryime/PDSG-SDA .

摘要: 尖峰神经网络(SNN)在处理基于事件的时空数据方面表现出了低能耗的能力。与传统的人工神经网络(ANN)类似，SNN也容易受到基于梯度的攻击，其中梯度是通过时空反向传播(STBP)和代理梯度(SGS)来计算的。然而，对于仅限推理的模型，SGS可能是不可见的，因为它们不影响推理结果，并且当前基于梯度的攻击对于动态视觉传感器(DVS)捕获的二值动态图像无效。虽然一些方法通过通用的SGS解决了看不见的SGS的问题，但它们的SGS缺乏与受害者模型的相关性，导致性能次优。此外，现有的基于SNN的二进制攻击的不可见性仍然是不够的。在本文中，我们引入了一种创新的势依赖代理梯度(PDSG)方法来在SG和模型之间建立稳健的连接，从而增强了对具有不可见SGS的各种模型的对抗性攻击的适应性。此外，我们还提出了稀疏动态攻击(SDA)来有效地攻击二值动态图像。利用世代约简范式，SDA可以充分优化逆境扰动的稀疏性。实验结果表明，在不同的模型和数据集上，我们的PDSG和SDA都优于最先进的基于SNN的攻击。具体地说，我们的PDSG在ImageNet上达到了100%的攻击成功率，而我们的SDA在CIFAR10DVS上只修改了0.24%的像素就获得了82%的攻击成功率。代码可在https://github.com/ryime/PDSG-SDA上获得。



## **7. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Preemptive Adversarial Defense**

快速抢占：前向-后向级联学习，实现高效且可转移的抢占式对抗防御 cs.CR

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2407.15524v7) [paper-pdf](http://arxiv.org/pdf/2407.15524v7)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: {Deep learning has made significant strides but remains susceptible to adversarial attacks, undermining its reliability. Most existing research addresses these threats after attacks happen. A growing direction explores preemptive defenses like mitigating adversarial threats proactively, offering improved robustness but at cost of efficiency and transferability. This paper introduces Fast Preemption, a novel preemptive adversarial defense that overcomes efficiency challenges while achieving state-of-the-art robustness and transferability, requiring no prior knowledge of attacks and target models. We propose a forward-backward cascade learning algorithm, which generates protective perturbations by combining forward propagation for rapid convergence with iterative backward propagation to prevent overfitting. Executing in just three iterations, Fast Preemption outperforms existing training-time, test-time, and preemptive defenses. Additionally, we introduce an adaptive reversion attack to assess the reliability of preemptive defenses, demonstrating that our approach remains secure in realistic attack scenarios.

摘要: {深度学习已经取得了重大进展，但仍然容易受到对手攻击，破坏了其可靠性。大多数现有的研究都是在袭击发生后解决这些威胁的。一个不断发展的方向是探索先发制人的防御措施，如主动缓解对手威胁，提供更好的健壮性，但代价是效率和可转移性。本文介绍了一种新型的抢占式对抗防御技术--快速抢占，它克服了效率上的挑战，同时实现了最先进的健壮性和可转移性，不需要攻击和目标模型的先验知识。我们提出了一种前向-后向级联学习算法，该算法通过结合前向传播快速收敛和迭代后向传播防止过拟合来产生保护性扰动。快速先发制人只需三次迭代即可完成，其性能优于现有的训练时间、测试时间和先发制人防御系统。此外，我们引入了自适应反向攻击来评估先发制人防御的可靠性，证明了我们的方法在现实攻击场景中仍然是安全的。



## **8. Scale-Invariant Adversarial Attack against Arbitrary-scale Super-resolution**

针对辅助规模超分辨率的规模不变对抗攻击 cs.CV

15 pages, accepted by TIFS 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04385v1) [paper-pdf](http://arxiv.org/pdf/2503.04385v1)

**Authors**: Yihao Huang, Xin Luo, Qing Guo, Felix Juefei-Xu, Xiaojun Jia, Weikai Miao, Geguang Pu, Yang Liu

**Abstract**: The advent of local continuous image function (LIIF) has garnered significant attention for arbitrary-scale super-resolution (SR) techniques. However, while the vulnerabilities of fixed-scale SR have been assessed, the robustness of continuous representation-based arbitrary-scale SR against adversarial attacks remains an area warranting further exploration. The elaborately designed adversarial attacks for fixed-scale SR are scale-dependent, which will cause time-consuming and memory-consuming problems when applied to arbitrary-scale SR. To address this concern, we propose a simple yet effective ``scale-invariant'' SR adversarial attack method with good transferability, termed SIAGT. Specifically, we propose to construct resource-saving attacks by exploiting finite discrete points of continuous representation. In addition, we formulate a coordinate-dependent loss to enhance the cross-model transferability of the attack. The attack can significantly deteriorate the SR images while introducing imperceptible distortion to the targeted low-resolution (LR) images. Experiments carried out on three popular LIIF-based SR approaches and four classical SR datasets show remarkable attack performance and transferability of SIAGT.

摘要: 局部连续图像函数(LIIF)的出现引起了任意尺度超分辨率(SR)技术的极大关注。然而，尽管固定尺度SR的脆弱性已经被评估，但基于连续表示的任意尺度SR对敌意攻击的稳健性仍然是一个值得进一步研究的领域。针对固定规模SR精心设计的敌意攻击是规模相关的，当应用于任意规模SR时，会导致耗时和内存消耗的问题。为了解决这一问题，我们提出了一种简单而有效的具有良好可转移性的“尺度不变”SR对抗攻击方法，称为SIAGT。具体地说，我们提出了利用连续表示的有限离散点来构造节省资源的攻击。此外，为了增强攻击的跨模型可转移性，我们还制定了一个坐标相关损失。该攻击可以显著恶化SR图像，同时给目标低分辨率(LR)图像带来不可察觉的失真。在三种流行的基于LIIF的SR方法和四个经典SR数据集上的实验表明，SIAGT具有良好的攻击性能和可转移性。



## **9. $σ$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples**

$Sigma $-zero：$\ell_0 $-norm的基于对象的优化对抗性示例 cs.LG

Paper accepted at International Conference on Learning  Representations (ICLR 2025). Code available at  https://github.com/sigma0-advx/sigma-zero

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2402.01879v3) [paper-pdf](http://arxiv.org/pdf/2402.01879v3)

**Authors**: Antonio Emanuele Cinà, Francesco Villani, Maura Pintor, Lea Schönherr, Battista Biggio, Marcello Pelillo

**Abstract**: Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages a differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving robust and non-robust models, show that $\sigma$\texttt{-zero} finds minimum $\ell_0$-norm adversarial examples without requiring any time-consuming hyperparameter tuning, and that it outperforms all competing sparse attacks in terms of success rate, perturbation size, and efficiency.

摘要: 评估深度网络对抗基于梯度的攻击的健壮性是具有挑战性的。虽然大多数攻击考虑$\ell_2$-和$\ell_\inty$-范数约束来手工创建输入扰动，但只有少数攻击研究稀疏的$\ell_1$-和$\ell_0$-范数攻击。特别是，由于在非凸和不可微约束上进行优化的固有复杂性，$\ell_0$-范数攻击仍然是研究最少的。然而，在这些攻击下评估对手的健壮性可能会揭示出在更常规的$\ell_2$-和$\ell_\inty$-范数攻击中未被测试的弱点。在这项工作中，我们提出了一种新的$\ell_0$-范数攻击，称为$\sigma$-零，它利用$\ell_0$范数的可微近似来促进基于梯度的优化，并提出了一种自适应投影算子来动态调整损失最小化和扰动稀疏性之间的权衡。使用MNIST、CIFAR10和ImageNet数据集进行的广泛评估，包括健壮和非健壮模型，表明$\sigma$\Texttt{-Zero}在不需要任何耗时的超参数调整的情况下找到最小$\ell_0$-范数对抗示例，并且在成功率、扰动大小和效率方面优于所有竞争的稀疏攻击。



## **10. Robust Eavesdropping in the Presence of Adversarial Communications for RF Fingerprinting**

针对RF指纹识别的对抗通信的鲁棒发射 eess.SP

11 pages, 6 figures

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04120v1) [paper-pdf](http://arxiv.org/pdf/2503.04120v1)

**Authors**: Andrew Yuan, Rajeev Sahay

**Abstract**: Deep learning is an effective approach for performing radio frequency (RF) fingerprinting, which aims to identify the transmitter corresponding to received RF signals. However, beyond the intended receiver, malicious eavesdroppers can also intercept signals and attempt to fingerprint transmitters communicating over a wireless channel. Recent studies suggest that transmitters can counter such threats by embedding deep learning-based transferable adversarial attacks in their signals before transmission. In this work, we develop a time-frequency-based eavesdropper architecture that is capable of withstanding such transferable adversarial perturbations and thus able to perform effective RF fingerprinting. We theoretically demonstrate that adversarial perturbations injected by a transmitter are confined to specific time-frequency regions that are insignificant during inference, directly increasing fingerprinting accuracy on perturbed signals intercepted by the eavesdropper. Empirical evaluations on a real-world dataset validate our theoretical findings, showing that deep learning-based RF fingerprinting eavesdroppers can achieve classification performance comparable to the intended receiver, despite efforts made by the transmitter to deceive the eavesdropper. Our framework reveals that relying on transferable adversarial attacks may not be sufficient to prevent eavesdroppers from successfully fingerprinting transmissions in next-generation deep learning-based communications systems.

摘要: 深度学习是进行射频指纹识别的一种有效方法，其目的是识别接收到的射频信号对应的发射机。然而，除了预定的接收方之外，恶意窃听者还可以截获信号并尝试通过无线信道通信指纹传送器。最近的研究表明，发射机可以通过在发射前在其信号中嵌入基于深度学习的可转移对抗性攻击来对抗此类威胁。在这项工作中，我们开发了一种基于时频的窃听者体系结构，它能够抵抗这种可转移的敌意扰动，从而能够执行有效的射频指纹识别。我们从理论上证明了发射机注入的敌意扰动被限制在推理过程中不重要的特定时频区域，从而直接提高了窃听者截获的扰动信号的指纹识别精度。在真实数据集上的实验评估验证了我们的理论发现，基于深度学习的射频指纹窃听者可以获得与预期接收者相当的分类性能，尽管发射机做出了欺骗窃听者的努力。我们的框架揭示，在下一代基于深度学习的通信系统中，依赖于可转移的敌意攻击可能不足以阻止窃听者成功地识别传输的指纹。



## **11. Adversarial Decoding: Generating Readable Documents for Adversarial Objectives**

对抗性解码：为对抗性目标生成可读文档 cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.02163v2) [paper-pdf](http://arxiv.org/pdf/2410.02163v2)

**Authors**: Collin Zhang, Tingwei Zhang, Vitaly Shmatikov

**Abstract**: We design, implement, and evaluate adversarial decoding, a new, generic text generation technique that produces readable documents for different adversarial objectives. Prior methods either produce easily detectable gibberish, or cannot handle objectives that include embedding similarity. In particular, they only work for direct attacks (such as jailbreaking) and cannot produce adversarial text for realistic indirect injection, e.g., documents that (1) are retrieved in RAG systems in response to broad classes of queries, and also (2) adversarially influence subsequent generation. We also show that fluency (low perplexity) is not sufficient to evade filtering. We measure the effectiveness of adversarial decoding for different objectives, including RAG poisoning, jailbreaking, and evasion of defensive filters, and demonstrate that it outperforms existing methods while producing readable adversarial documents.

摘要: 我们设计、实现和评估对抗性解码，这是一种新的通用文本生成技术，可以为不同的对抗性目标生成可读文档。先前的方法要么产生容易检测到的胡言乱语，要么无法处理包括嵌入相似性在内的目标。特别是，它们仅适用于直接攻击（例如越狱），并且不能为现实的间接注入生成对抗文本，例如，（1）在RAG系统中检索以响应广泛类别的查询，并且（2）对后继产生不利影响的文档。我们还表明，流畅性（低困惑度）不足以逃避过滤。我们衡量了针对不同目标（包括RAG中毒、越狱和规避防御过滤器）的对抗解码的有效性，并证明它在生成可读的对抗文档的同时优于现有方法。



## **12. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2502.16750v2) [paper-pdf](http://arxiv.org/pdf/2502.16750v2)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehnaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但他们面临来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题会出现。考虑到多发越狱和欺骗性对准是一些主要的高级攻击，在监督训练期间使用的静态护栏无法减轻这些攻击，指出了现实世界健壮性的关键研究重点。动态多智能体系统中静态护栏的组合不能抵抗这些攻击。我们打算通过制定新的评估框架，确定和应对安全行动部署所面临的威胁，从而加强基于LLM的特工的安全。我们的工作使用了三种检测方法来通过反向图灵测试来检测流氓代理，并通过多代理模拟来分析欺骗性比对，并开发了一个反越狱系统，通过使用Gemini 1.5Pro和Llama-3.3-70B、使用工具中介的对抗场景来测试DeepSeek R1模型来开发反越狱系统。Gemini 1.5 PRO具有很强的检测能力，如94%的准确率，但在长时间攻击下，随着提示长度的增加，攻击成功率(ASR)增加，多样性度量在预测多个复杂系统故障时变得无效，系统存在持续漏洞。这些发现证明了采用基于主动监控的灵活安全系统的必要性，该系统可以由代理自己执行，并由系统管理员进行适应性干预，因为当前的模型可能会产生漏洞，从而导致系统不可靠和易受攻击。因此，在我们的工作中，我们试图解决这些情况，并提出一个全面的框架来对抗安全问题。



## **13. Task-Agnostic Attacks Against Vision Foundation Models**

针对愿景基金会模型的任务不可知攻击 cs.CV

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03842v1) [paper-pdf](http://arxiv.org/pdf/2503.03842v1)

**Authors**: Brian Pulfer, Yury Belousov, Vitaliy Kinakh, Teddy Furon, Slava Voloshynovskiy

**Abstract**: The study of security in machine learning mainly focuses on downstream task-specific attacks, where the adversarial example is obtained by optimizing a loss function specific to the downstream task. At the same time, it has become standard practice for machine learning practitioners to adopt publicly available pre-trained vision foundation models, effectively sharing a common backbone architecture across a multitude of applications such as classification, segmentation, depth estimation, retrieval, question-answering and more. The study of attacks on such foundation models and their impact to multiple downstream tasks remains vastly unexplored. This work proposes a general framework that forges task-agnostic adversarial examples by maximally disrupting the feature representation obtained with foundation models. We extensively evaluate the security of the feature representations obtained by popular vision foundation models by measuring the impact of this attack on multiple downstream tasks and its transferability between models.

摘要: 机器学习中的安全性研究主要集中在针对下游任务的攻击上，通过优化针对下游任务的损失函数来获得对抗性实例。与此同时，机器学习从业者采用公开可用的预先训练的视觉基础模型已成为标准做法，有效地在分类、分割、深度估计、检索、问答等多种应用程序中共享一个通用的骨干架构。对这类基础模型的攻击及其对多个下游任务的影响的研究仍极大地有待探索。这项工作提出了一个通用的框架，通过最大限度地破坏使用基础模型获得的特征表示来伪造与任务无关的对抗性例子。通过测量这种攻击对多个下游任务的影响及其在模型之间的可转移性，我们广泛地评估了流行的视觉基础模型所获得的特征表示的安全性。



## **14. Improving LLM Safety Alignment with Dual-Objective Optimization**

通过双目标优化改善LLM安全一致性 cs.CL

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03710v1) [paper-pdf](http://arxiv.org/pdf/2503.03710v1)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment

摘要: 现有的大型语言模型(LLM)的训练时间安全对齐技术仍然容易受到越狱攻击。直接偏好优化(DPO)是一种被广泛应用的比对方法，由于其损失函数对于拒绝学习来说是次优的，因此在实验和理论环境中都显示出局限性。通过基于梯度的分析，我们识别了这些缺点，并提出了一种改进的安全对齐方法，将DPO目标分解为两个组成部分：(1)稳健的拒绝训练，即使产生部分不安全的生成也鼓励拒绝，以及(2)有针对性地忘记有害知识。这种方法显著提高了LLM对各种越狱攻击的稳健性，包括跨分发内和分发外场景的预填充、后缀和多轮攻击。此外，通过引入基于奖励的拒绝学习令牌级加权机制，我们引入了一种强调关键拒绝令牌的方法，进一步提高了对恶意攻击的鲁棒性。我们的研究还表明，越狱攻击的稳健性与训练过程中令牌分布的变化以及拒绝和有害令牌的内部表征相关，为未来LLM安全匹配的研究提供了有价值的方向。代码可在https://github.com/wicai24/DOOR-Alignment上获得



## **15. CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial Robustness of CLIP**

CLIP足够强大反击：测试时反击CLIP零攻击对抗鲁棒性 cs.CV

Accepted to CVPR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03613v1) [paper-pdf](http://arxiv.org/pdf/2503.03613v1)

**Authors**: Songlong Xing, Zhengyu Zhao, Nicu Sebe

**Abstract**: Despite its prevalent use in image-text matching tasks in a zero-shot manner, CLIP has been shown to be highly vulnerable to adversarial perturbations added onto images. Recent studies propose to finetune the vision encoder of CLIP with adversarial samples generated on the fly, and show improved robustness against adversarial attacks on a spectrum of downstream datasets, a property termed as zero-shot robustness. In this paper, we show that malicious perturbations that seek to maximise the classification loss lead to `falsely stable' images, and propose to leverage the pre-trained vision encoder of CLIP to counterattack such adversarial images during inference to achieve robustness. Our paradigm is simple and training-free, providing the first method to defend CLIP from adversarial attacks at test time, which is orthogonal to existing methods aiming to boost zero-shot adversarial robustness of CLIP. We conduct experiments across 16 classification datasets, and demonstrate stable and consistent gains compared to test-time defence methods adapted from existing adversarial robustness studies that do not rely on external networks, without noticeably impairing performance on clean images. We also show that our paradigm can be employed on CLIP models that have been adversarially finetuned to further enhance their robustness at test time. Our code is available \href{https://github.com/Sxing2/CLIP-Test-time-Counterattacks}{here}.

摘要: 尽管CLIP在图像-文本匹配任务中以零镜头的方式广泛使用，但它被证明对添加到图像上的敌意扰动非常脆弱。最近的研究建议使用动态生成的对抗性样本来微调CLIP的视觉编码器，并显示出对下游数据集频谱上的对抗性攻击的改进的稳健性，这一特性被称为零镜头稳健性。在本文中，我们证明了试图最大化分类损失的恶意扰动会导致图像的虚假稳定，并提出在推理过程中利用CLIP的预先训练的视觉编码器来对抗这种恶意图像以实现稳健性。我们的范例简单且无需训练，提供了第一种在测试时保护CLIP免受对手攻击的方法，这与现有的旨在提高CLIP的零射击对抗健壮性的方法是正交的。我们在16个分类数据集上进行了实验，并证明了与测试时防御方法相比，稳定和一致的收益来自于现有的不依赖外部网络的对抗性研究，而不会明显影响干净图像的性能。我们还表明，我们的范例可以用于经过相反微调的CLIP模型，以进一步增强它们在测试时的健壮性。我们的代码可从\href{https://github.com/Sxing2/CLIP-Test-time-Counterattacks}{here}.获得



## **16. AttackSeqBench: Benchmarking Large Language Models' Understanding of Sequential Patterns in Cyber Attacks**

AttackSeqBench：对大型语言模型对网络攻击中序列模式的理解进行基准测试 cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03170v1) [paper-pdf](http://arxiv.org/pdf/2503.03170v1)

**Authors**: Javier Yong, Haokai Ma, Yunshan Ma, Anis Yusof, Zhenkai Liang, Ee-Chien Chang

**Abstract**: The observations documented in Cyber Threat Intelligence (CTI) reports play a critical role in describing adversarial behaviors, providing valuable insights for security practitioners to respond to evolving threats. Recent advancements of Large Language Models (LLMs) have demonstrated significant potential in various cybersecurity applications, including CTI report understanding and attack knowledge graph construction. While previous works have proposed benchmarks that focus on the CTI extraction ability of LLMs, the sequential characteristic of adversarial behaviors within CTI reports remains largely unexplored, which holds considerable significance in developing a comprehensive understanding of how adversaries operate. To address this gap, we introduce AttackSeqBench, a benchmark tailored to systematically evaluate LLMs' capability to understand and reason attack sequences in CTI reports. Our benchmark encompasses three distinct Question Answering (QA) tasks, each task focuses on the varying granularity in adversarial behavior. To alleviate the laborious effort of QA construction, we carefully design an automated dataset construction pipeline to create scalable and well-formulated QA datasets based on real-world CTI reports. To ensure the quality of our dataset, we adopt a hybrid approach of combining human evaluation and systematic evaluation metrics. We conduct extensive experiments and analysis with both fast-thinking and slow-thinking LLMs, while highlighting their strengths and limitations in analyzing the sequential patterns in cyber attacks. The overarching goal of this work is to provide a benchmark that advances LLM-driven CTI report understanding and fosters its application in real-world cybersecurity operations. Our dataset and code are available at https://github.com/Javiery3889/AttackSeqBench .

摘要: 网络威胁情报(CTI)报告中记录的观察结果在描述敌对行为方面发挥了关键作用，为安全从业者提供了宝贵的见解，以应对不断变化的威胁。大型语言模型的最新进展在各种网络安全应用中显示出巨大的潜力，包括CTI报告理解和攻击知识图的构建。虽然前人的研究主要集中在低层统计模型的CTI提取能力上，但CTI报告中敌方行为的时序特征在很大程度上还没有被探索，这对于全面理解敌方是如何运作的具有相当重要的意义。为了弥补这一差距，我们引入了AttackSeqBtch，这是一个专门为系统评估LLMS理解和推理CTI报告中的攻击序列的能力而定制的基准测试。我们的基准包括三个不同的问答(QA)任务，每个任务都专注于敌对行为中不同的粒度。为了减轻QA构建的繁重工作，我们精心设计了一个自动化的数据集构建管道，以真实世界的CTI报告为基础创建可扩展的、格式良好的QA数据集。为了确保我们的数据集的质量，我们采用了人工评估和系统评估度量相结合的混合方法。我们使用快速思维和缓慢思维的LLM进行了广泛的实验和分析，同时强调了它们在分析网络攻击中的序列模式方面的优势和局限性。这项工作的总体目标是提供一个基准，以促进LLM驱动的CTI报告的理解，并促进其在现实世界网络安全操作中的应用。我们的数据集和代码可在https://github.com/Javiery3889/AttackSeqBench上获得。



## **17. Exploiting Vulnerabilities in Speech Translation Systems through Targeted Adversarial Attacks**

通过定向对抗攻击利用语音翻译系统中的漏洞 cs.SD

Preprint,17 pages, 17 figures

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.00957v2) [paper-pdf](http://arxiv.org/pdf/2503.00957v2)

**Authors**: Chang Liu, Haolin Wu, Xi Yang, Kui Zhang, Cong Wu, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: As speech translation (ST) systems become increasingly prevalent, understanding their vulnerabilities is crucial for ensuring robust and reliable communication. However, limited work has explored this issue in depth. This paper explores methods of compromising these systems through imperceptible audio manipulations. Specifically, we present two innovative approaches: (1) the injection of perturbation into source audio, and (2) the generation of adversarial music designed to guide targeted translation, while also conducting more practical over-the-air attacks in the physical world. Our experiments reveal that carefully crafted audio perturbations can mislead translation models to produce targeted, harmful outputs, while adversarial music achieve this goal more covertly, exploiting the natural imperceptibility of music. These attacks prove effective across multiple languages and translation models, highlighting a systemic vulnerability in current ST architectures. The implications of this research extend beyond immediate security concerns, shedding light on the interpretability and robustness of neural speech processing systems. Our findings underscore the need for advanced defense mechanisms and more resilient architectures in the realm of audio systems. More details and samples can be found at https://adv-st.github.io.

摘要: 随着语音翻译(ST)系统变得越来越普遍，了解它们的漏洞对于确保健壮可靠的通信至关重要。然而，深入探讨这一问题的工作有限。本文探索了通过不可感知的音频操作来危害这些系统的方法。具体地说，我们提出了两种创新的方法：(1)在源音频中注入扰动，(2)生成对抗性音乐，旨在指导定向翻译，同时也在物理世界中进行更实际的空中攻击。我们的实验表明，精心制作的音频扰动可能会误导翻译模型产生目标明确的有害输出，而对抗性音乐则更隐蔽地实现这一目标，利用音乐的自然不可感知性。事实证明，这些攻击在多种语言和翻译模型中都有效，突显了当前ST架构中的系统性漏洞。这项研究的意义超越了直接的安全问题，揭示了神经语音处理系统的可解释性和稳健性。我们的发现强调了音频系统领域需要先进的防御机制和更具弹性的架构。欲了解更多详情和样品，请访问https://adv-st.github.io.。



## **18. Detecting Adversarial Data using Perturbation Forgery**

使用微扰伪造检测对抗数据 cs.CV

Accepted as a conference paper at CVPR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2405.16226v4) [paper-pdf](http://arxiv.org/pdf/2405.16226v4)

**Authors**: Qian Wang, Chen Li, Yuchen Luo, Hefei Ling, Shijuan Huang, Ruoxi Jia, Ning Yu

**Abstract**: As a defense strategy against adversarial attacks, adversarial detection aims to identify and filter out adversarial data from the data flow based on discrepancies in distribution and noise patterns between natural and adversarial data. Although previous detection methods achieve high performance in detecting gradient-based adversarial attacks, new attacks based on generative models with imbalanced and anisotropic noise patterns evade detection. Even worse, the significant inference time overhead and limited performance against unseen attacks make existing techniques impractical for real-world use. In this paper, we explore the proximity relationship among adversarial noise distributions and demonstrate the existence of an open covering for these distributions. By training on the open covering of adversarial noise distributions, a detector with strong generalization performance against various types of unseen attacks can be developed. Based on this insight, we heuristically propose Perturbation Forgery, which includes noise distribution perturbation, sparse mask generation, and pseudo-adversarial data production, to train an adversarial detector capable of detecting any unseen gradient-based, generative-based, and physical adversarial attacks. Comprehensive experiments conducted on multiple general and facial datasets, with a wide spectrum of attacks, validate the strong generalization of our method.

摘要: 敌意检测是针对敌意攻击的一种防御策略，其目的是根据自然数据和敌意数据之间的分布和噪声模式的差异，从数据流中识别和过滤敌意数据。虽然以前的检测方法在检测基于梯度的敌意攻击方面取得了较高的性能，但基于非平衡和各向异性噪声模式的生成模型的新攻击可以逃避检测。更糟糕的是，巨大的推理时间开销和有限的针对看不见的攻击的性能使得现有技术在现实世界中不切实际。本文研究了对抗性噪声分布之间的邻近关系，并证明了这些分布的开覆盖的存在性。通过对对抗性噪声分布的开放覆盖进行训练，可以开发出对各种类型的不可见攻击具有很强的泛化性能的检测器。基于这一观点，我们启发式地提出了扰动伪造，它包括噪声分布扰动、稀疏掩码生成和伪对抗数据生成，以训练一个能够检测任何不可见的基于梯度的、基于生成的和物理的对抗攻击的对抗检测器。在多个普通数据集和人脸数据集上进行的综合实验表明，该方法具有较强的泛化能力。



## **19. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

最后的迭代优势：差异化私人新元的经验审计和原则性启发式分析 cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2410.06186v3) [paper-pdf](http://arxiv.org/pdf/2410.06186v3)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.

摘要: 在只释放最后一次迭代而隐藏中间迭代的情况下，提出了一种简单的启发式噪声截断随机梯度下降(DP-SGD)隐私分析方法。也就是说，我们的启发式假设模型是线性结构。我们的实验表明，我们的启发式方法可以预测隐私审计应用于各种训练过程的结果。因此，它可以在培训前用作最终隐私泄露的粗略估计。我们还通过提供一些低估隐私泄露的人工反例来探讨我们的启发式算法的局限性。标准的基于组合的DP-SGD隐私分析有效地假设攻击者可以访问所有中间迭代，这通常是不现实的。然而，这种分析在实践中仍然是最先进的。虽然我们的启发式方法没有取代严格的隐私分析，但它说明了最佳理论上限和隐私审计下限之间的巨大差距，并为进一步改进理论隐私分析设定了目标。我们还实证地支持我们的启发式攻击，并表明现有的隐私审计攻击受到我们在视觉和语言任务中的启发式分析的约束。



## **20. An Undetectable Watermark for Generative Image Models**

生成图像模型的不可检测水印 cs.CR

ICLR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2410.07369v3) [paper-pdf](http://arxiv.org/pdf/2410.07369v3)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.

摘要: 我们提出了第一个不可检测的生成图像模型的水印方案。不可检测性确保了即使在进行了许多自适应查询之后，有效的攻击者也无法区分加水印和未加水印的图像。特别是，不可检测的水印在任何有效计算的度量下都不会降低图像质量。我们的方案通过使用伪随机纠错码(Christian和Gunn，2024)来选择扩散模型的初始潜伏期，这是一种保证不可检测性和稳健性的策略。实验证明，利用稳定扩散2.1算法，水印具有较好的保质性和稳健性。我们的实验证明，与我们测试的每个方案相比，我们的水印不会降低图像质量。我们的实验也证明了我们的稳健性：现有的水印去除攻击不能在不显著降低图像质量的情况下去除图像中的水印。最后，我们发现我们的水印可以稳健地编码512比特，当图像没有受到水印去除攻击时，可以编码高达2500比特。我们的代码可以在https://github.com/XuandongZhao/PRC-Watermark.上找到



## **21. LLM Misalignment via Adversarial RLHF Platforms**

对抗性LLHF平台的LLM失调 cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.03039v1) [paper-pdf](http://arxiv.org/pdf/2503.03039v1)

**Authors**: Erfan Entezami, Ali Naseh

**Abstract**: Reinforcement learning has shown remarkable performance in aligning language models with human preferences, leading to the rise of attention towards developing RLHF platforms. These platforms enable users to fine-tune models without requiring any expertise in developing complex machine learning algorithms. While these platforms offer useful features such as reward modeling and RLHF fine-tuning, their security and reliability remain largely unexplored. Given the growing adoption of RLHF and open-source RLHF frameworks, we investigate the trustworthiness of these systems and their potential impact on behavior of LLMs. In this paper, we present an attack targeting publicly available RLHF tools. In our proposed attack, an adversarial RLHF platform corrupts the LLM alignment process by selectively manipulating data samples in the preference dataset. In this scenario, when a user's task aligns with the attacker's objective, the platform manipulates a subset of the preference dataset that contains samples related to the attacker's target. This manipulation results in a corrupted reward model, which ultimately leads to the misalignment of the language model. Our results demonstrate that such an attack can effectively steer LLMs toward undesirable behaviors within the targeted domains. Our work highlights the critical need to explore the vulnerabilities of RLHF platforms and their potential to cause misalignment in LLMs during the RLHF fine-tuning process.

摘要: 强化学习在将语言模型与人类偏好保持一致方面表现出了显著的性能，导致了人们对开发RLHF平台的关注。这些平台使用户能够微调模型，而不需要开发复杂的机器学习算法的任何专业知识。虽然这些平台提供了有用的功能，如奖励建模和RLHF微调，但它们的安全性和可靠性在很大程度上仍未得到探索。鉴于RLHF和开源RLHF框架越来越多地被采用，我们调查了这些系统的可信性及其对LLM行为的潜在影响。本文提出了一种针对公开可用的RLHF工具的攻击。在我们提出的攻击中，敌意的RLHF平台通过选择性地操纵偏好数据集中的数据样本来破坏LLM比对过程。在这种情况下，当用户的任务与攻击者的目标一致时，平台操作包含与攻击者目标相关的样本的首选项数据集的子集。这种操作会导致奖励模型被破坏，这最终会导致语言模型的不一致。我们的结果表明，这样的攻击可以有效地将LLM引向目标域内的不良行为。我们的工作突出了探索RLHF平台的脆弱性及其在RLHF微调过程中导致LLM未对准的可能性的迫切需要。



## **22. Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis**

注意差距：通过查询更新分析检测正在形成的黑匣子对抗攻击 cs.CR

IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02986v1) [paper-pdf](http://arxiv.org/pdf/2503.02986v1)

**Authors**: Jeonghwan Park, Niall McLaughlin, Ihsen Alouani

**Abstract**: Adversarial attacks remain a significant threat that can jeopardize the integrity of Machine Learning (ML) models. In particular, query-based black-box attacks can generate malicious noise without having access to the victim model's architecture, making them practical in real-world contexts. The community has proposed several defenses against adversarial attacks, only to be broken by more advanced and adaptive attack strategies. In this paper, we propose a framework that detects if an adversarial noise instance is being generated. Unlike existing stateful defenses that detect adversarial noise generation by monitoring the input space, our approach learns adversarial patterns in the input update similarity space. In fact, we propose to observe a new metric called Delta Similarity (DS), which we show it captures more efficiently the adversarial behavior. We evaluate our approach against 8 state-of-the-art attacks, including adaptive attacks, where the adversary is aware of the defense and tries to evade detection. We find that our approach is significantly more robust than existing defenses both in terms of specificity and sensitivity.

摘要: 对抗性攻击仍然是一个严重的威胁，可能会危及机器学习(ML)模型的完整性。特别是，基于查询的黑盒攻击可以在不访问受害者模型的体系结构的情况下生成恶意噪声，使它们在真实世界的上下文中具有实用性。社区已经提出了几种针对对抗性攻击的防御措施，但都被更先进和适应性更强的攻击策略打破了。在这篇文章中，我们提出了一个框架，它检测是否正在生成对抗性噪声实例。与现有的通过监测输入空间来检测对抗性噪声产生的状态防御方法不同，我们的方法在输入更新相似性空间中学习对抗性模式。事实上，我们提出了一种新的度量，称为Delta相似度(DS)，我们表明它更有效地捕获了对手的行为。我们评估了我们的方法针对8种最先进的攻击，包括自适应攻击，在这些攻击中，对手知道防御并试图逃避检测。我们发现，我们的方法在特异性和敏感性方面都明显比现有的防御方法更稳健。



## **23. Decentralized Adversarial Training over Graphs**

图上的分散对抗训练 cs.LG

arXiv admin note: text overlap with arXiv:2303.01936

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2303.13326v2) [paper-pdf](http://arxiv.org/pdf/2303.13326v2)

**Authors**: Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed

**Abstract**: The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of distributed learning, we develop a decentralized adversarial training framework for multi-agent systems. Specifically, we devise two decentralized adversarial training algorithms by relying on two popular decentralized learning strategies--diffusion and consensus. We analyze the convergence properties of the proposed framework for strongly-convex, convex, and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.

摘要: 近年来，机器学习模型对敌意攻击的脆弱性引起了人们的极大关注。现有的研究大多集中在单智能体学习者的行为上。相比之下，这项工作研究的是图上的对抗性训练，在图中，单个代理人受到空间上不同强度水平的扰动。考虑到组的协调能力，预计链接代理的交互以及图上可能的攻击模型的异构性可以帮助增强稳健性。利用分布式学习的最小-最大公式，我们提出了一个多智能体系统的分布式对抗训练框架。具体地说，我们依靠两种流行的去中心化学习策略--扩散和共识，设计了两种去中心化对抗训练算法。我们分析了该框架在强凸、凸和非凸环境下的收敛性质，并说明了该框架增强了对敌意攻击的鲁棒性。



## **24. Towards Safe AI Clinicians: A Comprehensive Study on Large Language Model Jailbreaking in Healthcare**

迈向安全的人工智能临床医生：医疗保健领域大语言模型越狱的综合研究 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2501.18632v2) [paper-pdf](http://arxiv.org/pdf/2501.18632v2)

**Authors**: Hang Zhang, Qian Lou, Yanshan Wang

**Abstract**: Large language models (LLMs) are increasingly utilized in healthcare applications. However, their deployment in clinical practice raises significant safety concerns, including the potential spread of harmful information. This study systematically assesses the vulnerabilities of seven LLMs to three advanced black-box jailbreaking techniques within medical contexts. To quantify the effectiveness of these techniques, we propose an automated and domain-adapted agentic evaluation pipeline. Experiment results indicate that leading commercial and open-source LLMs are highly vulnerable to medical jailbreaking attacks. To bolster model safety and reliability, we further investigate the effectiveness of Continual Fine-Tuning (CFT) in defending against medical adversarial attacks. Our findings underscore the necessity for evolving attack methods evaluation, domain-specific safety alignment, and LLM safety-utility balancing. This research offers actionable insights for advancing the safety and reliability of AI clinicians, contributing to ethical and effective AI deployment in healthcare.

摘要: 大型语言模型(LLM)越来越多地用于医疗保健应用程序。然而，它们在临床实践中的部署引起了重大的安全担忧，包括有害信息的潜在传播。这项研究系统地评估了七种低密度脂蛋白对三种先进的黑盒越狱技术在医学背景下的脆弱性。为了量化这些技术的有效性，我们提出了一个自动化的和领域适应的代理评估管道。实验结果表明，领先的商业和开源LLM非常容易受到医疗越狱攻击。为了支持模型的安全性和可靠性，我们进一步研究了连续微调(CFT)在防御医疗对手攻击方面的有效性。我们的发现强调了对不断发展的攻击方法进行评估、特定领域的安全对齐和LLM安全效用平衡的必要性。这项研究为提高人工智能临床医生的安全性和可靠性提供了可操作的见解，有助于在医疗保健领域进行合乎道德和有效的人工智能部署。



## **25. Assessing Robustness via Score-Based Adversarial Image Generation**

通过基于分数的对抗图像生成评估稳健性 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2310.04285v3) [paper-pdf](http://arxiv.org/pdf/2310.04285v3)

**Authors**: Marcel Kollovieh, Lukas Gosch, Marten Lienen, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantics-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate unrestricted adversarial examples that overcome the limitations of $\ell_p$-norm constraints. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG improves upon the majority of state-of-the-art attacks and defenses across multiple benchmarks. This work highlights the importance of investigating adversarial examples bounded by semantics rather than $\ell_p$-norm constraints. ScoreAG represents an important step towards more encompassing robustness assessments.

摘要: 大多数对抗性攻击和防御都集中在较小的$\ell_p$-范数约束内的扰动。然而，$\ell_p$威胁模型不能捕获所有相关的语义保持扰动，因此健壮性评估的范围是有限的。在这项工作中，我们介绍了基于分数的对抗性生成(ScoreAG)，这是一个新的框架，它利用基于分数的生成模型的进步来生成不受限制的对抗性实例，克服了$\ell_p$-范数约束的限制。与传统方法不同，ScoreAG在生成敌意示例的同时保持了图像的核心语义，要么转换现有图像，要么完全从头开始合成新的图像。我们进一步利用ScoreAG的生成能力来净化图像，经验上增强了分类器的稳健性。我们广泛的经验评估表明，ScoreAG在多个基准上改进了大多数最先进的攻击和防御。这项工作强调了研究受语义约束的对抗性例子的重要性，而不是$\ell_p$-范数约束。ScoreAG代表着朝着更全面的稳健性评估迈出的重要一步。



## **26. Realizing Quantum Adversarial Defense on a Trapped-ion Quantum Processor**

在俘获离子量子处理器上实现量子对抗防御 quant-ph

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02436v1) [paper-pdf](http://arxiv.org/pdf/2503.02436v1)

**Authors**: Alex Jin, Tarun Dutta, Anh Tu Ngo, Anupam Chattopadhyay, Manas Mukherjee

**Abstract**: Classification is a fundamental task in machine learning, typically performed using classical models. Quantum machine learning (QML), however, offers distinct advantages, such as enhanced representational power through high-dimensional Hilbert spaces and energy-efficient reversible gate operations. Despite these theoretical benefits, the robustness of QML classifiers against adversarial attacks and inherent quantum noise remains largely under-explored. In this work, we implement a data re-uploading-based quantum classifier on an ion-trap quantum processor using a single qubit to assess its resilience under realistic conditions. We introduce a novel convolutional quantum classifier architecture leveraging data re-uploading and demonstrate its superior robustness on the MNIST dataset. Additionally, we quantify the effects of polarization noise in a realistic setting, where both bit and phase noises are present, further validating the classifier's robustness. Our findings provide insights into the practical security and reliability of quantum classifiers, bridging the gap between theoretical potential and real-world deployment.

摘要: 分类是机器学习中的一项基本任务，通常使用经典模型执行。然而，量子机器学习(QML)提供了独特的优势，例如通过高维希尔伯特空间和节能的可逆门操作来增强表征能力。尽管有这些理论上的好处，但QML分类器对敌意攻击和固有的量子噪声的稳健性仍未得到很大程度的研究。在这项工作中，我们使用单个量子比特在离子陷阱量子处理器上实现了一个基于数据重传的量子分类器，以评估其在现实条件下的弹性。我们介绍了一种利用数据重传的卷积量子分类器结构，并在MNIST数据集上展示了其优越的健壮性。此外，我们在实际环境中量化了极化噪声的影响，其中位噪声和相位噪声都存在，进一步验证了分类器的稳健性。我们的发现为量子分类器的实际安全性和可靠性提供了洞察力，弥合了理论潜力和现实部署之间的差距。



## **27. Trace of the Times: Rootkit Detection through Temporal Anomalies in Kernel Activity**

时代的痕迹：通过核心活动中的时间异常进行RootKit检测 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02402v1) [paper-pdf](http://arxiv.org/pdf/2503.02402v1)

**Authors**: Max Landauer, Leonhard Alton, Martina Lindorfer, Florian Skopik, Markus Wurzenberger, Wolfgang Hotwagner

**Abstract**: Kernel rootkits provide adversaries with permanent high-privileged access to compromised systems and are often a key element of sophisticated attack chains. At the same time, they enable stealthy operation and are thus difficult to detect. Thereby, they inject code into kernel functions to appear invisible to users, for example, by manipulating file enumerations. Existing detection approaches are insufficient, because they rely on signatures that are unable to detect novel rootkits or require domain knowledge about the rootkits to be detected. To overcome this challenge, our approach leverages the fact that runtimes of kernel functions targeted by rootkits increase when additional code is executed. The framework outlined in this paper injects probes into the kernel to measure time stamps of functions within relevant system calls, computes distributions of function execution times, and uses statistical tests to detect time shifts. The evaluation of our open-source implementation on publicly available data sets indicates high detection accuracy with an F1 score of 98.7\% across five scenarios with varying system states.

摘要: 内核Rootkit为攻击者提供了对受攻击系统的永久高权限访问权限，通常是复杂攻击链的关键元素。同时，它们实现了隐形操作，因此很难被检测到。因此，它们向内核函数注入代码，使其对用户不可见，例如，通过操作文件枚举。现有的检测方法是不够的，因为它们依赖于不能检测到新的Rootkit或需要关于Rootkit的领域知识来检测的签名。为了克服这一挑战，我们的方法利用了这样一个事实，即Rootkit所针对的内核函数的运行时间会随着额外代码的执行而增加。本文概述的框架将探测器注入内核，以测量相关系统调用中函数的时间戳，计算函数执行时间的分布，并使用统计测试来检测时间偏移。在公开可用的数据集上对我们的开源实现进行的评估表明，在五个不同系统状态的场景中，检测准确率很高，F1得分为98.7\%。



## **28. Evaluating the Robustness of LiDAR Point Cloud Tracking Against Adversarial Attack**

评估LiDART点云跟踪对抗攻击的鲁棒性 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2410.20893v2) [paper-pdf](http://arxiv.org/pdf/2410.20893v2)

**Authors**: Shengjing Tian, Yinan Han, Xiantong Zhao, Bin Liu, Xiuping Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.

摘要: 在这项研究中，我们深入研究了基于神经网络的LiDAR点云跟踪模型在对抗攻击下的稳健性，这是一个经常被忽略的关键方面，有助于提高性能。尽管这些模型集成了Transformer或Bird‘s Eye View(BEV)等高级架构，但在面临对手攻击、域转移或数据损坏等挑战时往往忽略了健壮性。相反，我们关注的是跟踪模型在对抗性攻击威胁下的健壮性。我们首先建立一个统一的框架，在3D对象跟踪的背景下进行对抗性攻击，这使我们能够彻底调查白盒和黑盒攻击策略。对于白盒攻击，我们定制了特定的损失函数以适应不同的跟踪范例，并将现有的方法如FGSM、C\&W和PGD扩展到点云域。针对黑盒攻击场景，我们引入了一种新的基于传输的方法，目标感知扰动生成(TAPG)算法，其双重目标是实现高攻击性能和保持低可感知性。该方法使用启发式策略来实施稀疏攻击约束，并利用随机子向量分解来增强可转移性。我们的实验结果揭示了高级跟踪方法在同时受到黑盒和白盒攻击时的显著漏洞，强调了在设计LiDAR点云跟踪模型时考虑对对手攻击的健壮性的必要性。值得注意的是，与现有方法相比，TAPG还在攻击的有效性和扰动的隐蔽性之间取得了最佳平衡。



## **29. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

NoPain：通过最佳传输奇异边界进行无箱点云攻击 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.00063v2) [paper-pdf](http://arxiv.org/pdf/2503.00063v2)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. The source code will be publicly available.

摘要: 对抗性攻击利用深度模型针对对抗性样本的脆弱性。现有的点云攻击者是为特定模型量身定做的，基于白盒或黑盒设置中的渐变迭代优化扰动。尽管它们的攻击性能很有希望，但由于代理模型的特定参数过高，它们经常难以产生可转移的对抗性样本。为了解决这一问题，我们将重点转移到数据分布本身，并引入了一种名为NoPain的新方法，该方法使用最优传输(OT)来识别跨网络点云攻击数据流形的固有奇异边界。具体地，我们首先计算噪声到目标特征空间的OT映射，然后通过定位不可微位置来识别奇异边界。最后，我们沿着奇异边界进行采样以生成对抗性点云。一旦确定了奇异边界，NoPain就可以有效地产生对抗性样本，而不需要迭代更新或来自代理分类器的指导。大量的实验表明，端到端方法在可转移性和效率方面都优于基线方法，同时在与防御策略相比也保持了显著的优势。源代码将向公众开放。



## **30. Game-Theoretic Defenses for Robust Conformal Prediction Against Adversarial Attacks in Medical Imaging**

针对医学成像中对抗攻击的鲁棒共形预测的游戏理论防御 cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2411.04376v2) [paper-pdf](http://arxiv.org/pdf/2411.04376v2)

**Authors**: Rui Luo, Jie Bao, Zhixin Zhou, Chuangyin Dang

**Abstract**: Adversarial attacks pose significant threats to the reliability and safety of deep learning models, especially in critical domains such as medical imaging. This paper introduces a novel framework that integrates conformal prediction with game-theoretic defensive strategies to enhance model robustness against both known and unknown adversarial perturbations. We address three primary research questions: constructing valid and efficient conformal prediction sets under known attacks (RQ1), ensuring coverage under unknown attacks through conservative thresholding (RQ2), and determining optimal defensive strategies within a zero-sum game framework (RQ3). Our methodology involves training specialized defensive models against specific attack types and employing maximum and minimum classifiers to aggregate defenses effectively. Extensive experiments conducted on the MedMNIST datasets, including PathMNIST, OrganAMNIST, and TissueMNIST, demonstrate that our approach maintains high coverage guarantees while minimizing prediction set sizes. The game-theoretic analysis reveals that the optimal defensive strategy often converges to a singular robust model, outperforming uniform and simple strategies across all evaluated datasets. This work advances the state-of-the-art in uncertainty quantification and adversarial robustness, providing a reliable mechanism for deploying deep learning models in adversarial environments.

摘要: 对抗性攻击对深度学习模型的可靠性和安全性构成了严重威胁，特别是在医学成像等关键领域。本文介绍了一种新的框架，它将保形预测与博弈论防御策略相结合，以增强模型对已知和未知对手扰动的稳健性。我们主要研究了三个问题：在已知攻击(RQ1)下构造有效且高效的共形预测集，通过保守阈值(RQ2)确保未知攻击下的覆盖，以及在零和博弈框架(RQ3)下确定最优防御策略。我们的方法包括针对特定的攻击类型训练专门的防御模型，并使用最大和最小分类器来有效地聚合防御。在包括PathMNIST、OrganAMNIST和TIseMNIST在内的MedMNIST数据集上进行的大量实验表明，我们的方法在保持高覆盖率的同时最小化了预测集的大小。博弈论分析表明，最优防御策略往往收敛到一个奇异的稳健模型，在所有评估的数据集上表现优于统一和简单的策略。这项工作推进了不确定性量化和对抗稳健性方面的最新进展，为在对抗环境中部署深度学习模型提供了可靠的机制。



## **31. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

TPIA：针对面向代码的大型语言模型的特定目标提示注入攻击 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2407.09164v5) [paper-pdf](http://arxiv.org/pdf/2407.09164v5)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully exploited to simplify and facilitate programming. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. In this paper, we reveal that both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process, which may not be practical; adversarial attacks struggle with fulfilling specific malicious purposes. To alleviate these problems, this paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an attack success rate of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an attack success rate of over 90%) in all threat cases, using only a 12-token non-functional perturbation.

摘要: 最近，面向代码的大型语言模型(Code LLM)已经被广泛并成功地利用来简化和促进编程。不幸的是，一些开创性的工作表明，这些代码LLM容易受到后门和对手的攻击。前者毒化训练数据或模型参数，在遇到触发器时劫持LLMS生成恶意代码片段。后者制作恶意敌意输入代码以降低生成代码的质量。在本文中，我们揭示了这两种攻击都有一些固有的局限性：后门攻击依赖于对手控制模型训练过程的能力，这可能是不实用的；对抗性攻击难以实现特定的恶意目的。针对这些问题，提出了一种新的针对代码LLMS的攻击范式，即目标特定的即时注入攻击(TPIA)。TPIA生成包含恶意指令信息的非功能性扰动，并通过将它们传播到可能使用的依赖项(例如，包或RAG的知识库)，将它们插入到受害者的代码上下文中。它诱导代码LLM在目标位置生成攻击者指定的恶意代码片段。一般而言，我们通过基于贪婪令牌搜索的对抗性优化将攻击者指定的恶意目标压缩为扰动。我们收集了13个具有代表性的恶意目标，为三种流行的编程语言设计了31个威胁案例。实验表明，在所有威胁情况下，仅使用12个令牌的非功能扰动，我们的TPIA就可以成功攻击三个典型的开源代码LLM(攻击成功率高达97.9%)和两个主流商业代码LLM集成应用(攻击成功率超过90%)。



## **32. Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion**

基于属性感知文本倒置的预算驱动可转移对抗攻击 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2502.19697v2) [paper-pdf](http://arxiv.org/pdf/2502.19697v2)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yaonan Wang

**Abstract**: Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.

摘要: 人员再识别(re-id)模型在安全监控系统中是至关重要的，需要可转移的对抗性攻击来探索其脆弱性。近年来，基于视觉语言模型(VLM)的攻击通过攻击VLM的广义图像和文本特征表现出了良好的可转移性，但由于过于强调积分表示的区分语义，缺乏全面的特征破坏。本文介绍了一种新的基于属性感知的提示攻击(AP-Attack)，该方法利用VLM的图文对齐能力，通过破坏特定于属性的文本嵌入来显式破坏行人图像的细粒度语义特征。为了获得个性化的属性文本描述，文本倒置网络被设计成将行人图像映射到代表语义嵌入的伪标记，并利用图像和预定义的提示模板进行对比学习，以明确描述行人属性。反转良性和对抗性细粒度的文本语义有助于攻击者有效地进行彻底的破坏，增强了对抗性实例的可转移性。大量实验表明，AP-Attack具有最好的可转移性，在跨模型和数据集攻击场景下，平均丢失率比以前的方法高出22.9%。



## **33. Adversarial Tokenization**

对抗性代币化 cs.CL

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02174v1) [paper-pdf](http://arxiv.org/pdf/2503.02174v1)

**Authors**: Renato Lui Geh, Zilei Shao, Guy Van den Broeck

**Abstract**: Current LLM pipelines account for only one possible tokenization for a given string, ignoring exponentially many alternative tokenizations during training and inference. For example, the standard Llama3 tokenization of penguin is [p,enguin], yet [peng,uin] is another perfectly valid alternative. In this paper, we show that despite LLMs being trained solely on one tokenization, they still retain semantic understanding of other tokenizations, raising questions about their implications in LLM safety. Put succinctly, we answer the following question: can we adversarially tokenize an obviously malicious string to evade safety and alignment restrictions? We show that not only is adversarial tokenization an effective yet previously neglected axis of attack, but it is also competitive against existing state-of-the-art adversarial approaches without changing the text of the harmful request. We empirically validate this exploit across three state-of-the-art LLMs and adversarial datasets, revealing a previously unknown vulnerability in subword models.

摘要: 当前的LLM流水线只考虑了给定字符串的一个可能的标记化，在训练和推理期间忽略了指数级的许多替代标记化。例如，企鹅的标准Llama3标记化是[p，enguin]，然而[peng，uin]是另一个完全有效的替代方案。在这篇文章中，我们证明了尽管LLM只接受了一个标记化的训练，但它们仍然保留了对其他标记化的语义理解，这引发了人们对它们在LLM安全性方面的影响的疑问。简而言之，我们回答了以下问题：我们可以恶意地对一个明显恶意的字符串进行标记，以逃避安全和对齐限制吗？我们表明，对抗性标记化不仅是一个以前被忽视的有效攻击轴心，而且在不改变有害请求的案文的情况下，也是与现有最先进的对抗性方法相竞争的。我们在三个最先进的LLM和对抗性数据集上实证验证了这一利用，揭示了子词模型中一个以前未知的漏洞。



## **34. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

HoSNN：具有自适应激发阈值的对抗鲁棒自适应尖峰神经网络 cs.NE

Accepted by TMLR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2308.10373v4) [paper-pdf](http://arxiv.org/pdf/2308.10373v4)

**Authors**: Hejia Geng, Peng Li

**Abstract**: While spiking neural networks (SNNs) offer a promising neurally-inspired model of computation, they are vulnerable to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to design a threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model and utilize TA-LIF neurons to construct the adversarially robust homeostatic SNNs (HoSNNs) for improved robustness. The TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, offering a local feedback control solution to the minimization of each neuron's membrane potential error caused by adversarial disturbance. Theoretical analysis demonstrates favorable dynamic properties of TA-LIF neurons in terms of the bounded-input bounded-output stability and suppressed time growth of membrane potential error, underscoring their superior robustness compared with the standard LIF neurons. When trained with weak FGSM attacks (attack budget = 2/255) and tested with much stronger PGD attacks (attack budget = 8/255), our HoSNNs significantly improve model accuracy on several datasets: from 30.54% to 74.91% on FashionMNIST, from 0.44% to 35.06% on SVHN, from 0.56% to 42.63% on CIFAR10, from 0.04% to 16.66% on CIFAR100, over the conventional LIF-based SNNs.

摘要: 虽然尖峰神经网络(SNN)提供了一种很有前途的神经启发计算模型，但它们容易受到对手的攻击。我们首次从神经元自平衡中得到启发，设计了一种阈值自适应泄漏积分与点火(TA-LIF)神经元模型，并利用TA-LIF神经元来构造逆稳健的自平衡SNN(HoSNN)，以提高鲁棒性。TA-LIF模型引入了一种自稳定的动态阈值机制，提供了一种局部反馈控制方案来最小化对抗性干扰引起的每个神经元的膜电位误差。理论分析表明，TA-LIF神经元在有界输入有界输出稳定性和抑制膜电位误差的时间增长方面具有良好的动态特性，与标准LIF神经元相比具有更好的鲁棒性。当用弱FGSM攻击(攻击预算=2/255)训练和用更强的PGD攻击(攻击预算=8/255)测试时，我们的HoSNN在几个数据集上显著提高了模型准确率：在FashionMNIST上从30.54%到74.91%，在SVHN上从0.44%到35.06%，在CIFAR10上从0.56%到42.63%，在CIFAR100上从0.04%到16.66%，比传统的基于LIF的SNN。



## **35. DDAD: A Two-pronged Adversarial Defense Based on Distributional Discrepancy**

DDAD：基于分配差异的双管齐下的对抗防御 cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02169v1) [paper-pdf](http://arxiv.org/pdf/2503.02169v1)

**Authors**: Jiacheng Zhang, Benjamin I. P. Rubinstein, Jingfeng Zhang, Feng Liu

**Abstract**: Statistical adversarial data detection (SADD) detects whether an upcoming batch contains adversarial examples (AEs) by measuring the distributional discrepancies between clean examples (CEs) and AEs. In this paper, we reveal the potential strength of SADD-based methods by theoretically showing that minimizing distributional discrepancy can help reduce the expected loss on AEs. Nevertheless, despite these advantages, SADD-based methods have a potential limitation: they discard inputs that are detected as AEs, leading to the loss of clean information within those inputs. To address this limitation, we propose a two-pronged adversarial defense method, named Distributional-Discrepancy-based Adversarial Defense (DDAD). In the training phase, DDAD first optimizes the test power of the maximum mean discrepancy (MMD) to derive MMD-OPT, and then trains a denoiser by minimizing the MMD-OPT between CEs and AEs. In the inference phase, DDAD first leverages MMD-OPT to differentiate CEs and AEs, and then applies a two-pronged process: (1) directly feeding the detected CEs into the classifier, and (2) removing noise from the detected AEs by the distributional-discrepancy-based denoiser. Extensive experiments show that DDAD outperforms current state-of-the-art (SOTA) defense methods by notably improving clean and robust accuracy on CIFAR-10 and ImageNet-1K against adaptive white-box attacks.

摘要: 统计对抗性数据检测(SADD)通过测量干净样本(CE)和对抗性样本(AE)之间的分布差异来检测即将到来的一批是否包含对抗性样本(AE)。在本文中，我们揭示了基于SADD的方法的潜在优势，从理论上证明了最小化分布差异可以帮助减少AEs的预期损失。然而，尽管有这些优点，基于SADD的方法有一个潜在的局限性：它们丢弃被检测为AE的输入，导致在这些输入中丢失干净的信息。针对这一局限性，我们提出了一种双管齐下的对抗性防御方法，称为基于分布差异的对抗性防御(DDAD)。在训练阶段，DDAD首先优化最大平均偏差(MMD)的测试功率得到MMD-OPT，然后通过最小化CES和AES之间的MMD-OPT来训练去噪器。在推理阶段，DDAD首先利用MMD-OPT来区分CE和AE，然后采用双管齐下的方法：(1)直接将检测到的CE送入分类器；(2)通过基于分布差异的去噪器去除检测到的AE中的噪声。大量的实验表明，DDAD在CIFAR-10和ImageNet-1K上显著提高了对自适应白盒攻击的干净和健壮的准确性，从而超过了当前最先进的(SOTA)防御方法。



## **36. Towards Scalable Topological Regularizers**

迈向可扩展的布局调节器 cs.LG

31 pages, ICLR 2025 camera-ready version

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2501.14641v2) [paper-pdf](http://arxiv.org/pdf/2501.14641v2)

**Authors**: Hiu-Tung Wong, Darrick Lee, Hong Yan

**Abstract**: Latent space matching, which consists of matching distributions of features in latent space, is a crucial component for tasks such as adversarial attacks and defenses, domain adaptation, and generative modelling. Metrics for probability measures, such as Wasserstein and maximum mean discrepancy, are commonly used to quantify the differences between such distributions. However, these are often costly to compute, or do not appropriately take the geometric and topological features of the distributions into consideration. Persistent homology is a tool from topological data analysis which quantifies the multi-scale topological structure of point clouds, and has recently been used as a topological regularizer in learning tasks. However, computation costs preclude larger scale computations, and discontinuities in the gradient lead to unstable training behavior such as in adversarial tasks. We propose the use of principal persistence measures, based on computing the persistent homology of a large number of small subsamples, as a topological regularizer. We provide a parallelized GPU implementation of this regularizer, and prove that gradients are continuous for smooth densities. Furthermore, we demonstrate the efficacy of this regularizer on shape matching, image generation, and semi-supervised learning tasks, opening the door towards a scalable regularizer for topological features.

摘要: 潜在空间匹配由潜在空间中特征的匹配分布组成，是对抗性攻击和防御、领域自适应和产生式建模等任务的重要组成部分。概率度量指标，如Wasserstein和最大均值差异，通常用于量化此类分布之间的差异。然而，这些通常计算成本很高，或者没有适当地考虑分布的几何和拓扑特征。持久同调是一种从拓扑数据分析中量化点云多尺度拓扑结构的工具，近年来被用作学习任务中的拓扑正则化工具。然而，计算成本排除了更大规模的计算，并且梯度中的不连续会导致不稳定的训练行为，例如在对抗性任务中。在计算大量小样本的持久同调的基础上，我们提出使用主持久度量作为拓扑正则化。我们给出了这种正则化算法的并行GPU实现，并证明了对于光滑的密度，梯度是连续的。此外，我们展示了这种正则化算法在形状匹配、图像生成和半监督学习任务中的有效性，为拓扑特征的可伸缩正则化算法打开了大门。



## **37. Jailbreaking Safeguarded Text-to-Image Models via Large Language Models**

通过大型语言模型越狱受保护的文本到图像模型 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01839v1) [paper-pdf](http://arxiv.org/pdf/2503.01839v1)

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.

摘要: 文本到图像模型可能会生成有害内容，例如色情图像，特别是在提交不安全提示时。为了解决这个问题，通常在文本到图像模型之上添加安全过滤器，或者对模型本身进行调整以减少有害输出。然而，当攻击者战略性地设计对抗提示来绕过这些安全护栏时，这些防御仍然容易受到攻击。在这项工作中，我们提出了ObjetTune，这是一种使用微调的大型语言模型来越狱具有安全护栏的文本到图像模型的方法。与其他需要对目标模型重复查询的基于查询的越狱攻击不同，我们的攻击在微调AttackLLM后有效地生成对抗提示。我们在三个不安全提示数据集和五个安全护栏上评估我们的方法。我们的结果表明，我们的方法有效地绕过了安全护栏，优于现有的无框攻击，并且还促进了其他基于查询的攻击。



## **38. AutoAdvExBench: Benchmarking autonomous exploitation of adversarial example defenses**

AutoAdvExBench：对抗性示例防御的自主利用基准 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01811v1) [paper-pdf](http://arxiv.org/pdf/2503.01811v1)

**Authors**: Nicholas Carlini, Javier Rando, Edoardo Debenedetti, Milad Nasr, Florian Tramèr

**Abstract**: We introduce AutoAdvExBench, a benchmark to evaluate if large language models (LLMs) can autonomously exploit defenses to adversarial examples. Unlike existing security benchmarks that often serve as proxies for real-world tasks, bench directly measures LLMs' success on tasks regularly performed by machine learning security experts. This approach offers a significant advantage: if a LLM could solve the challenges presented in bench, it would immediately present practical utility for adversarial machine learning researchers. We then design a strong agent that is capable of breaking 75% of CTF-like ("homework exercise") adversarial example defenses. However, we show that this agent is only able to succeed on 13% of the real-world defenses in our benchmark, indicating the large gap between difficulty in attacking "real" code, and CTF-like code. In contrast, a stronger LLM that can attack 21% of real defenses only succeeds on 54% of CTF-like defenses. We make this benchmark available at https://github.com/ethz-spylab/AutoAdvExBench.

摘要: 我们引入了AutoAdvExB边，这是一个基准，用来评估大型语言模型(LLM)是否能够自主地利用对对手例子的防御。与通常作为真实任务代理的现有安全基准不同，BASE直接衡量LLMS在机器学习安全专家定期执行的任务中的成功程度。这种方法提供了一个显著的优势：如果LLM能够解决BASE中提出的挑战，它将立即为对抗性机器学习研究人员提供实用价值。然后，我们设计了一个强大的代理，它能够打破75%的CTF类(“家庭作业练习”)对抗性范例防御。然而，我们表明，在我们的基准测试中，该代理只能够在13%的真实世界防御中成功，这表明攻击“真实”代码的难度与类似CTF的代码之间存在巨大差距。相比之下，更强大的LLM可以攻击21%的真实防御，只能在54%的CTF类防御上成功。我们在https://github.com/ethz-spylab/AutoAdvExBench.上提供此基准测试



## **39. Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models**

猫混淆推理LLM：推理模型的查询不可知对抗触发器 cs.CL

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01781v1) [paper-pdf](http://arxiv.org/pdf/2503.01781v1)

**Authors**: Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani

**Abstract**: We investigate the robustness of reasoning models trained for step-by-step problem solving by introducing query-agnostic adversarial triggers - short, irrelevant text that, when appended to math problems, systematically mislead models to output incorrect answers without altering the problem's semantics. We propose CatAttack, an automated iterative attack pipeline for generating triggers on a weaker, less expensive proxy model (DeepSeek V3) and successfully transfer them to more advanced reasoning target models like DeepSeek R1 and DeepSeek R1-distilled-Qwen-32B, resulting in greater than 300% increase in the likelihood of the target model generating an incorrect answer. For example, appending, "Interesting fact: cats sleep most of their lives," to any math problem leads to more than doubling the chances of a model getting the answer wrong. Our findings highlight critical vulnerabilities in reasoning models, revealing that even state-of-the-art models remain susceptible to subtle adversarial inputs, raising security and reliability concerns. The CatAttack triggers dataset with model responses is available at https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers.

摘要: 我们通过引入查询不可知的对抗性触发器来研究为逐步解决问题而训练的推理模型的稳健性。查询不可知的对抗性触发器是一种简短的、无关的文本，当附加到数学问题后，系统地误导模型输出不正确的答案，而不改变问题的语义。我们提出了CatAttack，这是一种自动迭代攻击管道，用于在较弱、较便宜的代理模型(DeepSeek V3)上生成触发器，并成功地将它们传输到更高级的推理目标模型，如DeepSeek R1和DeepSeek R1-Distilleed-Qwen-32B，导致目标模型生成错误答案的可能性增加300%以上。例如，在任何一道数学题上加上“有趣的事实：猫一生中的大部分时间都在睡觉”，就会使模型答错答案的几率增加一倍以上。我们的发现突显了推理模型中的关键漏洞，揭示了即使是最先进的模型也仍然容易受到微妙的对手输入的影响，这引发了安全和可靠性方面的担忧。带有模型响应的CatAttack触发器数据集可在https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers.上获得



## **40. Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning**

对抗性代理：利用强化学习进行黑匣子规避攻击 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01734v1) [paper-pdf](http://arxiv.org/pdf/2503.01734v1)

**Authors**: Kyle Domico, Jean-Charles Noirot Ferrand, Ryan Sheatsley, Eric Pauley, Josiah Hanna, Patrick McDaniel

**Abstract**: Reinforcement learning (RL) offers powerful techniques for solving complex sequential decision-making tasks from experience. In this paper, we demonstrate how RL can be applied to adversarial machine learning (AML) to develop a new class of attacks that learn to generate adversarial examples: inputs designed to fool machine learning models. Unlike traditional AML methods that craft adversarial examples independently, our RL-based approach retains and exploits past attack experience to improve future attacks. We formulate adversarial example generation as a Markov Decision Process and evaluate RL's ability to (a) learn effective and efficient attack strategies and (b) compete with state-of-the-art AML. On CIFAR-10, our agent increases the success rate of adversarial examples by 19.4% and decreases the median number of victim model queries per adversarial example by 53.2% from the start to the end of training. In a head-to-head comparison with a state-of-the-art image attack, SquareAttack, our approach enables an adversary to generate adversarial examples with 13.1% more success after 5000 episodes of training. From a security perspective, this work demonstrates a powerful new attack vector that uses RL to attack ML models efficiently and at scale.

摘要: 强化学习(RL)为从经验中解决复杂的顺序决策任务提供了强大的技术。在本文中，我们演示了如何将RL应用于对抗性机器学习(AML)，以开发一类新的攻击，这些攻击学习生成对抗性示例：旨在愚弄机器学习模型的输入。与独立制作对抗性例子的传统AML方法不同，我们基于RL的方法保留并利用过去的攻击经验来改进未来的攻击。我们将敌意实例生成描述为一个马尔可夫决策过程，并评估了RL在(A)学习有效和高效的攻击策略和(B)与最先进的AML竞争的能力。在CIFAR-10上，从训练开始到训练结束，我们的代理将对抗性实例的成功率提高了19.4%，并将每个对抗性实例的受害者模型查询的中位数减少了53.2%。在与最先进的图像攻击SquareAttack进行面对面的比较中，我们的方法使对手能够在5000集的训练后生成对抗性例子，成功率提高13.1%。从安全的角度来看，这项工作展示了一种强大的新攻击载体，它使用RL来高效和大规模地攻击ML模型。



## **41. Towards Physically Realizable Adversarial Attacks in Embodied Vision Navigation**

视觉导航中实现物理可实现的对抗攻击 cs.CV

7 pages, 7 figures, submitted to IEEE/RSJ International Conference on  Intelligent Robots and Systems (IROS) 2025

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.10071v4) [paper-pdf](http://arxiv.org/pdf/2409.10071v4)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The significant advancements in embodied vision navigation have raised concerns about its susceptibility to adversarial attacks exploiting deep neural networks. Investigating the adversarial robustness of embodied vision navigation is crucial, especially given the threat of 3D physical attacks that could pose risks to human safety. However, existing attack methods for embodied vision navigation often lack physical feasibility due to challenges in transferring digital perturbations into the physical world. Moreover, current physical attacks for object detection struggle to achieve both multi-view effectiveness and visual naturalness in navigation scenarios. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches to objects, where both opacity and textures are learnable. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which optimizes the patch's texture based on feedback from the vision-based perception model used in navigation. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, in which opacity is fine-tuned after texture optimization. Experimental results demonstrate that our adversarial patches decrease the navigation success rate by an average of 22.39%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav

摘要: 体现视觉导航的重大进步引起了人们对其易受利用深度神经网络的敌意攻击的担忧。研究体现视觉导航的对抗健壮性至关重要，特别是考虑到可能对人类安全构成风险的3D物理攻击的威胁。然而，现有的具身视觉导航攻击方法往往缺乏物理可行性，这是因为将数字扰动转换到物理世界的挑战。此外，目前用于目标检测的物理攻击难以在导航场景中实现多视角有效性和视觉自然度。为了解决这个问题，我们提出了一种实用的具身导航攻击方法，通过在对象上附加敌意补丁来实现，其中不透明度和纹理都是可学习的。具体地说，为了确保不同视点的有效性，我们采用了一种基于对象感知采样的多视点优化策略，该策略根据导航中使用的基于视觉的感知模型的反馈来优化面片的纹理。为了使面片不易被人察觉，我们引入了一种两阶段不透明度优化机制，在纹理优化后对不透明度进行微调。实验结果表明，我们的对抗性补丁使导航成功率平均降低了22.39%，在实用性、有效性和自然性方面都优于以往的方法。代码可从以下网址获得：https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav



## **42. Revisiting Locally Differentially Private Protocols: Towards Better Trade-offs in Privacy, Utility, and Attack Resistance**

重新审视本地差异私有协议：在隐私、实用性和抗攻击方面实现更好的权衡 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01482v1) [paper-pdf](http://arxiv.org/pdf/2503.01482v1)

**Authors**: Héber H. Arcolezi, Sébastien Gambs

**Abstract**: Local Differential Privacy (LDP) offers strong privacy protection, especially in settings in which the server collecting the data is untrusted. However, designing LDP mechanisms that achieve an optimal trade-off between privacy, utility, and robustness to adversarial inference attacks remains challenging. In this work, we introduce a general multi-objective optimization framework for refining LDP protocols, enabling the joint optimization of privacy and utility under various adversarial settings. While our framework is flexible enough to accommodate multiple privacy and security attacks as well as utility metrics, in this paper we specifically optimize for Attacker Success Rate (ASR) under distinguishability attack as a measure of privacy and Mean Squared Error (MSE) as a measure of utility. We systematically revisit these trade-offs by analyzing eight state-of-the-art LDP protocols and proposing refined counterparts that leverage tailored optimization techniques. Experimental results demonstrate that our proposed adaptive mechanisms consistently outperform their non-adaptive counterparts, reducing ASR by up to five orders of magnitude while maintaining competitive utility. Analytical derivations also confirm the effectiveness of our mechanisms, moving them closer to the ASR-MSE Pareto frontier.

摘要: 本地差异隐私(LDP)提供强大的隐私保护，尤其是在收集数据的服务器不受信任的情况下。然而，设计LDP机制以实现隐私、效用和对抗推理攻击的稳健性之间的最佳平衡仍然具有挑战性。在这项工作中，我们引入了一个通用的多目标优化框架来精化LDP协议，使得在不同的敌意环境下能够联合优化隐私和效用。虽然我们的框架足够灵活，可以适应多种隐私和安全攻击以及效用度量，但在本文中，我们特别针对可区分攻击下的攻击者成功率(ASR)和效用度量的均方误差(MSE)进行了优化。我们通过分析八种最先进的LDP协议并提出利用定制优化技术的改进的对等协议，系统地重新审视了这些权衡。实验结果表明，我们提出的自适应机制的性能始终优于非自适应机制，在保持竞争效用的同时，ASR降低了高达五个数量级。解析推导也证实了我们的机制的有效性，使它们更接近ASR-MSE的帕累托前沿。



## **43. Poison-splat: Computation Cost Attack on 3D Gaussian Splatting**

毒药飞溅：对3D高斯飞溅的计算成本攻击 cs.CV

Accepted by ICLR 2025 as a spotlight paper

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2410.08190v2) [paper-pdf](http://arxiv.org/pdf/2410.08190v2)

**Authors**: Jiahao Lu, Yifan Zhang, Qiuhong Shen, Xinchao Wang, Shuicheng Yan

**Abstract**: 3D Gaussian splatting (3DGS), known for its groundbreaking performance and efficiency, has become a dominant 3D representation and brought progress to many 3D vision tasks. However, in this work, we reveal a significant security vulnerability that has been largely overlooked in 3DGS: the computation cost of training 3DGS could be maliciously tampered by poisoning the input data. By developing an attack named Poison-splat, we reveal a novel attack surface where the adversary can poison the input images to drastically increase the computation memory and time needed for 3DGS training, pushing the algorithm towards its worst computation complexity. In extreme cases, the attack can even consume all allocable memory, leading to a Denial-of-Service (DoS) that disrupts servers, resulting in practical damages to real-world 3DGS service vendors. Such a computation cost attack is achieved by addressing a bi-level optimization problem through three tailored strategies: attack objective approximation, proxy model rendering, and optional constrained optimization. These strategies not only ensure the effectiveness of our attack but also make it difficult to defend with simple defensive measures. We hope the revelation of this novel attack surface can spark attention to this crucial yet overlooked vulnerability of 3DGS systems. Our code is available at https://github.com/jiahaolu97/poison-splat .

摘要: 三维高斯飞溅(3DGS)以其开创性的性能和效率而闻名，已经成为一种占主导地位的3D表示，并为许多3D视觉任务带来了进展。然而，在这项工作中，我们揭示了一个在3DGS中被很大程度上忽视的重大安全漏洞：通过毒化输入数据，可以恶意篡改训练3DGS的计算成本。通过开发一种名为Poison-Splat的攻击，我们揭示了一种新颖的攻击面，在该攻击面上，攻击者可以毒化输入图像，从而大幅增加3DGS训练所需的计算内存和时间，从而将算法推向最差的计算复杂度。在极端情况下，攻击甚至会耗尽所有可分配的内存，导致服务器中断的拒绝服务(DoS)，从而对现实世界中的3DGS服务供应商造成实际损害。这样的计算代价攻击是通过三种定制的策略来解决双层优化问题来实现的：攻击目标近似、代理模型渲染和可选的约束优化。这些策略不仅确保了我们进攻的有效性，而且使我们很难用简单的防御措施进行防守。我们希望这一新颖攻击面的揭示能引起人们对3DGS系统这一关键但被忽视的漏洞的关注。我们的代码可以在https://github.com/jiahaolu97/poison-splat上找到。



## **44. Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification**

分而治：基于扩散的对抗净化的异类噪音集成 cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01407v1) [paper-pdf](http://arxiv.org/pdf/2503.01407v1)

**Authors**: Gaozheng Pei, Shaojie Lyu, Gong Chen, Ke Ma, Qianqian Xu, Yingfei Sun, Qingming Huang

**Abstract**: Existing diffusion-based purification methods aim to disrupt adversarial perturbations by introducing a certain amount of noise through a forward diffusion process, followed by a reverse process to recover clean examples. However, this approach is fundamentally flawed: the uniform operation of the forward process across all pixels compromises normal pixels while attempting to combat adversarial perturbations, resulting in the target model producing incorrect predictions. Simply relying on low-intensity noise is insufficient for effective defense. To address this critical issue, we implement a heterogeneous purification strategy grounded in the interpretability of neural networks. Our method decisively applies higher-intensity noise to specific pixels that the target model focuses on while the remaining pixels are subjected to only low-intensity noise. This requirement motivates us to redesign the sampling process of the diffusion model, allowing for the effective removal of varying noise levels. Furthermore, to evaluate our method against strong adaptative attack, our proposed method sharply reduces time cost and memory usage through a single-step resampling. The empirical evidence from extensive experiments across three datasets demonstrates that our method outperforms most current adversarial training and purification techniques by a substantial margin.

摘要: 现有的基于扩散的净化方法旨在通过正向扩散过程引入一定量的噪声，然后通过反向过程来恢复干净的样本，从而破坏对抗性扰动。然而，这种方法从根本上是有缺陷的：在试图对抗对抗性扰动的同时，在所有像素上的前向过程的统一操作损害了正常像素，导致目标模型产生错误的预测。单纯依靠低强度噪声是不能有效防御的。为了解决这一关键问题，我们实施了一种基于神经网络可解释性的异质净化策略。我们的方法果断地将较高强度的噪声应用于目标模型关注的特定像素，而其余像素仅受到低强度噪声的影响。这一要求促使我们重新设计扩散模型的采样过程，允许有效地去除不同的噪声水平。此外，为了评估我们的方法对强适应性攻击的抵抗力，我们提出的方法通过一步重采样大大减少了时间开销和内存使用。来自三个数据集的广泛实验的经验证据表明，我们的方法比目前大多数对抗性训练和净化技术有很大的优势。



## **45. Attacking Large Language Models with Projected Gradient Descent**

使用投影梯度下降攻击大型语言模型 cs.LG

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2402.09154v2) [paper-pdf](http://arxiv.org/pdf/2402.09154v2)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Johannes Gasteiger, Stephan Günnemann

**Abstract**: Current LLM alignment methods are readily broken through specifically crafted adversarial prompts. While crafting adversarial prompts using discrete optimization is highly effective, such attacks typically use more than 100,000 LLM calls. This high computational cost makes them unsuitable for, e.g., quantitative analyses and adversarial training. To remedy this, we revisit Projected Gradient Descent (PGD) on the continuously relaxed input prompt. Although previous attempts with ordinary gradient-based attacks largely failed, we show that carefully controlling the error introduced by the continuous relaxation tremendously boosts their efficacy. Our PGD for LLMs is up to one order of magnitude faster than state-of-the-art discrete optimization to achieve the same devastating attack results.

摘要: 当前的LLM对齐方法很容易通过专门设计的对抗提示来突破。虽然使用离散优化制作对抗提示非常有效，但此类攻击通常使用超过100，000次LLM调用。这种高计算成本使它们不适合例如定量分析和对抗训练。为了解决这个问题，我们在持续放松的输入提示下重新审视投影梯度下降（PVD）。尽管之前对普通的基于梯度的攻击的尝试基本上失败了，但我们表明，仔细控制持续放松带来的错误可以极大地提高它们的功效。我们的LLM PGO比最先进的离散优化快一个数量级，以实现相同的毁灭性攻击结果。



## **46. Exact Certification of (Graph) Neural Networks Against Label Poisoning**

（图）神经网络对抗标签中毒的精确认证 cs.LG

Published as a spotlight presentation at ICLR 2025

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2412.00537v2) [paper-pdf](http://arxiv.org/pdf/2412.00537v2)

**Authors**: Mahalakshmi Sabanayagam, Lukas Gosch, Stephan Günnemann, Debarghya Ghoshdastidar

**Abstract**: Machine learning models are highly vulnerable to label flipping, i.e., the adversarial modification (poisoning) of training labels to compromise performance. Thus, deriving robustness certificates is important to guarantee that test predictions remain unaffected and to understand worst-case robustness behavior. However, for Graph Neural Networks (GNNs), the problem of certifying label flipping has so far been unsolved. We change this by introducing an exact certification method, deriving both sample-wise and collective certificates. Our method leverages the Neural Tangent Kernel (NTK) to capture the training dynamics of wide networks enabling us to reformulate the bilevel optimization problem representing label flipping into a Mixed-Integer Linear Program (MILP). We apply our method to certify a broad range of GNN architectures in node classification tasks. Thereby, concerning the worst-case robustness to label flipping: $(i)$ we establish hierarchies of GNNs on different benchmark graphs; $(ii)$ quantify the effect of architectural choices such as activations, depth and skip-connections; and surprisingly, $(iii)$ uncover a novel phenomenon of the robustness plateauing for intermediate perturbation budgets across all investigated datasets and architectures. While we focus on GNNs, our certificates are applicable to sufficiently wide NNs in general through their NTK. Thus, our work presents the first exact certificate to a poisoning attack ever derived for neural networks, which could be of independent interest. The code is available at https://github.com/saper0/qpcert.

摘要: 机器学习模型很容易受到标签翻转的影响，即对训练标签进行对抗性修改(中毒)以损害性能。因此，派生健壮性证书对于保证测试预测不受影响以及了解最坏情况下的健壮性行为非常重要。然而，对于图神经网络(GNN)来说，证明标签翻转的问题到目前为止还没有解决。我们通过引入一种精确的认证方法来改变这一点，即同时派生样本证书和集合证书。我们的方法利用神经切核(NTK)来捕捉广域网络的训练动态，使我们能够将表示标签翻转的双层优化问题重新描述为混合整数线性规划(MILP)。我们应用我们的方法在节点分类任务中验证了广泛的GNN体系结构。因此，关于标签翻转的最坏情况的稳健性：$(I)$我们在不同的基准图上建立了GNN的层次结构；$(Ii)$量化了体系结构选择的影响，例如激活、深度和跳过连接；令人惊讶的是，$(Iii)$发现了一个新的现象，即在所有调查的数据集和体系结构中，中间扰动预算的稳健性停滞不前。虽然我们专注于GNN，但我们的证书一般通过其NTK适用于足够广泛的NN。因此，我们的工作提供了有史以来第一个针对神经网络的中毒攻击的确切证书，这可能是独立的兴趣。代码可在https://github.com/saper0/qpcert.上获得



## **47. AdvLogo: Adversarial Patch Attack against Object Detectors based on Diffusion Models**

AdvLogo：针对基于扩散模型的对象检测器的对抗补丁攻击 cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.07002v2) [paper-pdf](http://arxiv.org/pdf/2409.07002v2)

**Authors**: Boming Miao, Chunxiao Li, Yao Zhu, Weixiang Sun, Zizhe Wang, Xiaoyi Wang, Chuanlong Xie

**Abstract**: With the rapid development of deep learning, object detectors have demonstrated impressive performance; however, vulnerabilities still exist in certain scenarios. Current research exploring the vulnerabilities using adversarial patches often struggles to balance the trade-off between attack effectiveness and visual quality. To address this problem, we propose a novel framework of patch attack from semantic perspective, which we refer to as AdvLogo. Based on the hypothesis that every semantic space contains an adversarial subspace where images can cause detectors to fail in recognizing objects, we leverage the semantic understanding of the diffusion denoising process and drive the process to adversarial subareas by perturbing the latent and unconditional embeddings at the last timestep. To mitigate the distribution shift that exposes a negative impact on image quality, we apply perturbation to the latent in frequency domain with the Fourier Transform. Experimental results demonstrate that AdvLogo achieves strong attack performance while maintaining high visual quality.

摘要: 随着深度学习的快速发展，目标检测器表现出了令人印象深刻的性能，但在某些场景下仍然存在漏洞。目前使用对抗性补丁探索漏洞的研究往往难以在攻击效率和视觉质量之间取得平衡。针对这一问题，我们从语义的角度提出了一种新的补丁攻击框架，称为AdvLogo。基于每个语义空间包含一个对抗性的子空间的假设，在这个子空间中，图像可能导致检测器无法识别目标，我们利用扩散去噪过程的语义理解，通过在最后一个时间步扰动潜在的和无条件的嵌入来驱动该过程到对抗性的子区域。为了减少分布漂移对图像质量的负面影响，我们利用傅里叶变换对频域中的潜伏点进行扰动。实验结果表明，AdvLogo在保持较高视觉质量的同时，具有较强的攻击性能。



## **48. Exploring Adversarial Robustness in Classification tasks using DNA Language Models**

使用DNA语言模型探索分类任务中的对抗鲁棒性 cs.CL

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.19788v2) [paper-pdf](http://arxiv.org/pdf/2409.19788v2)

**Authors**: Hyunwoo Yoo, Haebin Shin, Kaidi Xu, Gail Rosen

**Abstract**: DNA Language Models, such as GROVER, DNABERT2 and the Nucleotide Transformer, operate on DNA sequences that inherently contain sequencing errors, mutations, and laboratory-induced noise, which may significantly impact model performance. Despite the importance of this issue, the robustness of DNA language models remains largely underexplored. In this paper, we comprehensivly investigate their robustness in DNA classification by applying various adversarial attack strategies: the character (nucleotide substitutions), word (codon modifications), and sentence levels (back-translation-based transformations) to systematically analyze model vulnerabilities. Our results demonstrate that DNA language models are highly susceptible to adversarial attacks, leading to significant performance degradation. Furthermore, we explore adversarial training method as a defense mechanism, which enhances both robustness and classification accuracy. This study highlights the limitations of DNA language models and underscores the necessity of robustness in bioinformatics.

摘要: DNA语言模型，如Grover、DNABERT2和核苷酸转换器，对DNA序列进行操作，这些序列固有地包含测序错误、突变和实验室诱导的噪声，这些可能会显著影响模型的性能。尽管这个问题很重要，但DNA语言模型的稳健性在很大程度上仍然没有得到充分的研究。在本文中，我们通过应用各种对抗性攻击策略：字符(核苷酸替换)、单词(密码子修改)和句子级别(基于反向翻译的转换)来系统地分析模型的脆弱性，全面地研究了它们在DNA分类中的稳健性。我们的结果表明，DNA语言模型非常容易受到对抗性攻击，导致性能显著下降。此外，我们还探索了对抗性训练方法作为一种防御机制，提高了鲁棒性和分类准确率。这项研究突出了DNA语言模型的局限性，并强调了生物信息学中稳健性的必要性。



## **49. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

MAA：针对视觉语言预训练模型的强力对抗攻击 cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2502.08079v3) [paper-pdf](http://arxiv.org/pdf/2502.08079v3)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.

摘要: 当前用于评估视觉语言预训练(VLP)模型在多模式任务中的稳健性的对抗性攻击存在可转移性有限的问题，其中针对特定模型的攻击往往难以在不同的模型上有效地泛化，从而限制了它们在更广泛地评估稳健性方面的有效性。这主要归因于过度依赖特定型号的特征和区域，特别是在图像模式方面。在本文中，我们提出了一种优雅而高效的方法，称为精细攻击(MAA)，它充分利用了个体样本的模型无关特性和脆弱性，从而增强了泛化能力，降低了模型依赖。MAA通过开发一种新的调整大小和滑动裁剪(RSCrop)技术，结合多粒度相似破坏(MGSD)策略，强调对抗性图像的细粒度优化。在不同的VLP模型、多个基准数据集和各种下游任务上的广泛实验表明，MAA显著增强了对抗性攻击的有效性和可转移性。我们进行了大量的性能研究，以深入了解各种型号配置的有效性，从而指导该领域的未来发展。



## **50. Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation**

$\ell_\infty$-扰动下对抗训练估计的渐进行为 math.ST

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2401.15262v2) [paper-pdf](http://arxiv.org/pdf/2401.15262v2)

**Authors**: Yiling Xie, Xiaoming Huo

**Abstract**: Adversarial training has been proposed to protect machine learning models against adversarial attacks. This paper focuses on adversarial training under $\ell_\infty$-perturbation, which has recently attracted much research attention. The asymptotic behavior of the adversarial training estimator is investigated in the generalized linear model. The results imply that the asymptotic distribution of the adversarial training estimator under $\ell_\infty$-perturbation could put a positive probability mass at $0$ when the true parameter is $0$, providing a theoretical guarantee of the associated sparsity-recovery ability. Alternatively, a two-step procedure is proposed -- adaptive adversarial training, which could further improve the performance of adversarial training under $\ell_\infty$-perturbation. Specifically, the proposed procedure could achieve asymptotic variable-selection consistency and unbiasedness. Numerical experiments are conducted to show the sparsity-recovery ability of adversarial training under $\ell_\infty$-perturbation and to compare the empirical performance between classic adversarial training and adaptive adversarial training.

摘要: 对抗性训练已被提出用来保护机器学习模型免受对抗性攻击。本文主要研究近年来备受关注的对抗性训练问题。研究了广义线性模型中对抗性训练估计器的渐近行为。结果表明，当真参数为$0时，当真参数为$0时，对抗性训练估计量的渐近分布在$0-扰动下可以使正概率质量为$0，从而为相关稀疏恢复能力提供了理论保证。或者，提出了一种分两步进行的训练方法--自适应对抗性训练，它可以进一步提高对抗性训练在干扰下的性能。具体地说，该方法可以实现变量选择的渐近一致性和无偏性。通过数值实验验证了对抗性训练在扰动下的稀疏性恢复能力，并比较了经典对抗性训练和自适应对抗性训练的经验性能。



