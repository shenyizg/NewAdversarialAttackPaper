# Latest Adversarial Attack Papers
**update at 2025-05-03 15:43:14**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Fully passive quantum random number generation with untrusted light**

使用不可信光的完全被动量子随机数生成 quant-ph

21 pages, 9 figures

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00636v1) [paper-pdf](http://arxiv.org/pdf/2505.00636v1)

**Authors**: KaiWei Qiu, Yu Cai, Nelly H. Y. Ng, Jing Yan Haw

**Abstract**: Quantum random number generators (QRNGs) harness the inherent unpredictability of quantum mechanics to produce true randomness. Yet, in many optical implementations, the light source remains a potential vulnerability - susceptible to deviations from ideal behavior and even adversarial eavesdropping. Source-device-independent (SDI) protocols address this with a pragmatic strategy, by removing trust assumptions on the source, and instead rely on realistic modelling and characterization of the measurement device. In this work, we enhance an existing SDI-QRNG protocol by eliminating the need for a perfectly balanced beam splitter within the trusted measurement device, which is an idealized assumption made for the simplification of security analysis. We demonstrate that certified randomness can still be reliably extracted across a wide range of beam-splitting ratios, significantly improving the protocol's practicality and robustness. Using only off-the-shelf components, our implementation achieves real-time randomness generation rates of 0.347 Gbps. We also experimentally validate the protocol's resilience against adversarial attacks and highlight its self-testing capabilities. These advances mark a significant step toward practical, lightweight, high-performance, fully-passive, and composably secure QRNGs suitable for real-world deployment.

摘要: 量子随机数生成器（QRNG）利用量子力学固有的不可预测性来产生真正的随机性。然而，在许多光学实现中，光源仍然是一个潜在的漏洞--容易偏离理想行为，甚至遭到对抗性窃听。源设备无关（SDP）协议通过务实的策略解决了这个问题，通过消除对源的信任假设，而是依赖于测量设备的现实建模和描述。在这项工作中，我们通过消除可信测量设备内对完全平衡的分束器的需要来增强现有的SDI-QRNG协议，这是为了简化安全分析而做出的理想化假设。我们证明，在广泛的分束比范围内仍然可以可靠地提取经过认证的随机性，从而显着提高了协议的实用性和鲁棒性。我们的实施仅使用现成的组件，即可实现0.347 Gbps的实时随机性生成率。我们还通过实验验证了该协议对对抗攻击的弹性，并强调了其自我测试能力。这些进步标志着朝着适合现实世界部署的实用、轻量级、高性能、全无源和可组合安全的QRNG迈出了重要一步。



## **2. Fast and Low-Cost Genomic Foundation Models via Outlier Removal**

通过离群值去除的快速和低成本基因组基础模型 cs.LG

International Conference on Machine Learning (ICML) 2025

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00598v1) [paper-pdf](http://arxiv.org/pdf/2505.00598v1)

**Authors**: Haozheng Luo, Chenghao Qiu, Maojiang Su, Zhihan Zhou, Zoe Mehta, Guo Ye, Jerry Yao-Chieh Hu, Han Liu

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GERM. Unlike existing GFM benchmarks, GERM offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Empirically, transformer-based models exhibit greater robustness to adversarial perturbations compared to HyenaDNA, highlighting the impact of architectural design on vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features.

摘要: 我们为基因组基础模型（GFM）提出了第一个统一的对抗性攻击基准，名为GERM。与现有的GFM基准不同，GERM提供了第一个全面的评估框架，以系统地评估GFM对对抗性攻击的脆弱性。在方法上，我们评估了五个国家的最先进的GFM使用四个广泛采用的攻击算法和三个防御策略的对抗鲁棒性。重要的是，我们的基准提供了一个可访问的和全面的框架，以分析GFM漏洞的模型架构，量化方案和训练数据集。从经验上看，与鬣狗DNA相比，基于变换器的模型对对抗扰动表现出更大的鲁棒性，凸显了架构设计对脆弱性的影响。此外，对抗性攻击经常针对具有生物学意义的基因组区域，这表明这些模型有效地捕获了有意义的序列特征。



## **3. AMUN: Adversarial Machine UNlearning**

AMUN：对抗性机器取消学习 cs.LG

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2503.00917v2) [paper-pdf](http://arxiv.org/pdf/2503.00917v2)

**Authors**: Ali Ebrahimpour-Boroojeny, Hari Sundaram, Varun Chandrasekaran

**Abstract**: Machine unlearning, where users can request the deletion of a forget dataset, is becoming increasingly important because of numerous privacy regulations. Initial works on ``exact'' unlearning (e.g., retraining) incur large computational overheads. However, while computationally inexpensive, ``approximate'' methods have fallen short of reaching the effectiveness of exact unlearning: models produced fail to obtain comparable accuracy and prediction confidence on both the forget and test (i.e., unseen) dataset. Exploiting this observation, we propose a new unlearning method, Adversarial Machine UNlearning (AMUN), that outperforms prior state-of-the-art (SOTA) methods for image classification. AMUN lowers the confidence of the model on the forget samples by fine-tuning the model on their corresponding adversarial examples. Adversarial examples naturally belong to the distribution imposed by the model on the input space; fine-tuning the model on the adversarial examples closest to the corresponding forget samples (a) localizes the changes to the decision boundary of the model around each forget sample and (b) avoids drastic changes to the global behavior of the model, thereby preserving the model's accuracy on test samples. Using AMUN for unlearning a random $10\%$ of CIFAR-10 samples, we observe that even SOTA membership inference attacks cannot do better than random guessing.

摘要: 由于众多的隐私法规，机器取消学习（用户可以请求删除忘记的数据集）变得越来越重要。最初的工作是关于“准确”的取消学习（例如，再培训）会招致大量的计算管理费用。然而，虽然计算成本低，但“近似”方法未能达到精确取消学习的有效性：产生的模型未能在忘记和测试方面获得相当的准确性和预测置信度（即，看不见的）数据集。利用这一观察结果，我们提出了一种新的去学习方法--对抗机器去学习（AMUN），它优于现有的最新技术（SOTA）图像分类方法。AMUN通过根据相应的对抗性示例微调模型，降低了模型对忘记样本的信心。对抗性示例自然属于模型在输入空间上强加的分布;在最接近相应忘记样本的对抗性示例上微调模型，（a）将模型决策边界的变化局部化在每个忘记样本周围，并且（b）避免了模型的全局行为的剧烈变化，从而保持了模型对测试样本的准确性。使用AMUN对CIFAR-10样本中的随机10个样本进行去学习，我们观察到即使SOTA成员推断攻击也不能比随机猜测做得更好。



## **4. Analysis of the vulnerability of machine learning regression models to adversarial attacks using data from 5G wireless networks**

使用5G无线网络数据分析机器学习回归模型对对抗攻击的脆弱性 cs.CR

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00487v1) [paper-pdf](http://arxiv.org/pdf/2505.00487v1)

**Authors**: Leonid Legashev, Artur Zhigalov, Denis Parfenov

**Abstract**: This article describes the process of creating a script and conducting an analytical study of a dataset using the DeepMIMO emulator. An advertorial attack was carried out using the FGSM method to maximize the gradient. A comparison is made of the effectiveness of binary classifiers in the task of detecting distorted data. The dynamics of changes in the quality indicators of the regression model were analyzed in conditions without adversarial attacks, during an adversarial attack and when the distorted data was isolated. It is shown that an adversarial FGSM attack with gradient maximization leads to an increase in the value of the MSE metric by 33% and a decrease in the R2 indicator by 10% on average. The LightGBM binary classifier effectively identifies data with adversarial anomalies with 98% accuracy. Regression machine learning models are susceptible to adversarial attacks, but rapid analysis of network traffic and data transmitted over the network makes it possible to identify malicious activity

摘要: 本文描述了使用DeepMMO模拟器创建脚本并对数据集进行分析研究的过程。使用FGSM方法进行广告攻击以最大化梯度。比较了二进制分类器在检测失真数据任务中的有效性。在没有对抗性攻击的条件下、对抗性攻击期间以及隔离失真数据时，分析了回归模型质量指标变化的动态。结果表明，具有梯度最大化的对抗FGSM攻击导致SSE指标的值平均增加33%，R2指标平均减少10%。LightGBM二进制分类器以98%的准确率有效识别具有对抗异常的数据。回归机器学习模型容易受到对抗攻击，但对网络流量和网络上传输的数据的快速分析使识别恶意活动成为可能



## **5. HoneyWin: High-Interaction Windows Honeypot in Enterprise Environment**

HoneyWin：企业环境中的高交互性Windows蜜罐 cs.CR

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00465v1) [paper-pdf](http://arxiv.org/pdf/2505.00465v1)

**Authors**: Yan Lin Aung, Yee Loon Khoo, Davis Yang Zheng, Bryan Swee Duo, Sudipta Chattopadhyay, Jianying Zhou, Liming Lu, Weihan Goh

**Abstract**: Windows operating systems (OS) are ubiquitous in enterprise Information Technology (IT) and operational technology (OT) environments. Due to their widespread adoption and known vulnerabilities, they are often the primary targets of malware and ransomware attacks. With 93% of the ransomware targeting Windows-based systems, there is an urgent need for advanced defensive mechanisms to detect, analyze, and mitigate threats effectively. In this paper, we propose HoneyWin a high-interaction Windows honeypot that mimics an enterprise IT environment. The HoneyWin consists of three Windows 11 endpoints and an enterprise-grade gateway provisioned with comprehensive network traffic capturing, host-based logging, deceptive tokens, endpoint security and real-time alerts capabilities. The HoneyWin has been deployed live in the wild for 34 days and receives more than 5.79 million unsolicited connections, 1.24 million login attempts, 5 and 354 successful logins via remote desktop protocol (RDP) and secure shell (SSH) respectively. The adversary interacted with the deceptive token in one of the RDP sessions and exploited the public-facing endpoint to initiate the Simple Mail Transfer Protocol (SMTP) brute-force bot attack via SSH sessions. The adversary successfully harvested 1,250 SMTP credentials after attempting 151,179 credentials during the attack.

摘要: Windows操作系统（OS）在企业信息技术（IT）和运营技术（OT）环境中无处不在。由于它们的广泛采用和已知的漏洞，它们通常是恶意软件和勒索软件攻击的主要目标。93%的勒索软件针对基于Windows的系统，因此迫切需要先进的防御机制来有效地检测、分析和缓解威胁。在本文中，我们提出了HoneyWin一个模仿企业IT环境的高交互性Windows蜜罐。HoneyWin由三个Windows 11端点和一个企业级网关组成，该网关配备了全面的网络流量捕获、基于主机的日志记录、欺骗性令牌、端点安全和实时警报功能。HoneyWin已在野外实时部署34天，分别通过远程桌面协议（SDP）和安全外壳（SSH）接收了超过579万次未经请求的连接、124万次登录尝试、5次和354次成功登录。对手在其中一个SDP会话中与欺骗性令牌进行交互，并利用面向公众的端点通过SSH会话发起简单邮件传输协议（RTP）暴力机器人攻击。攻击期间尝试了151，179个凭据后，对手成功收集了1，250个RTP凭据。



## **6. GAN-based Generator of Adversarial Attack on Intelligent End-to-End Autoencoder-based Communication System**

基于GAN的智能端到端自动编码器通信系统对抗攻击发生器 cs.IT

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00395v1) [paper-pdf](http://arxiv.org/pdf/2505.00395v1)

**Authors**: Jianyuan Chen, Lin Zhang, Zuwei Chen, Yawen Chen, Hongcheng Zhuang

**Abstract**: Deep neural networks have been applied in wireless communications system to intelligently adapt to dynamically changing channel conditions, while the users are still under the threat of the malicious attacks due to the broadcasting property of wireless channels. However, most attack models require the knowledge of the target details, which is difficult to be implemented in real systems. Our objective is to develop an attack model with no requirement for the target information, while enhancing the block error rate. In our design, we propose a novel Generative Adversarial Networks(GANs) based attack architecture, which exploits the property of deep learning models being vulnerable to perturbations induced by dynamically changing channel conditions. In the proposed generator, the attack network is composed of convolution layer, convolution transpose layer and linear layer. Then we present the training strategy and the details of the training algorithm. Subsequently, we propose the validation strategy to evaluate the performance of the generator. Simulations are conducted and the results show that our proposed adversarial attack generator achieve better block error rate attack performance than that of benchmark schemes over Additive White Gaussian Noise (AWGN) channel, Rayleigh channel and High-Speed Railway channel.

摘要: 深度神经网络已被应用于无线通信系统中，以智能地适应动态变化的频道条件，而由于无线频道的广播特性，用户仍然面临恶意攻击的威胁。然而，大多数攻击模型都需要了解目标细节，这很难在实际系统中实现。我们的目标是开发一种不需要目标信息的攻击模型，同时提高块错误率。在我们的设计中，我们提出了一种基于生成对抗网络（GAN）的新型攻击架构，该架构利用了深度学习模型容易受到动态变化的通道条件引起的扰动的特性。在所提出的生成器中，攻击网络由卷积层、卷积转置层和线性层组成。然后我们介绍了训练策略和训练算法的细节。随后，我们提出了验证策略来评估生成器的性能。仿真结果表明，我们提出的对抗攻击生成器在加性高斯白噪音（AWGN）通道、Rayleigh通道和高速铁路通道上实现了比基准方案更好的误块率攻击性能。



## **7. SacFL: Self-Adaptive Federated Continual Learning for Resource-Constrained End Devices**

SacFL：针对资源受限的终端设备的自适应联邦连续学习 cs.LG

Accepted by TNNLS 2025

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00365v1) [paper-pdf](http://arxiv.org/pdf/2505.00365v1)

**Authors**: Zhengyi Zhong, Weidong Bao, Ji Wang, Jianguo Chen, Lingjuan Lyu, Wei Yang Bryan Lim

**Abstract**: The proliferation of end devices has led to a distributed computing paradigm, wherein on-device machine learning models continuously process diverse data generated by these devices. The dynamic nature of this data, characterized by continuous changes or data drift, poses significant challenges for on-device models. To address this issue, continual learning (CL) is proposed, enabling machine learning models to incrementally update their knowledge and mitigate catastrophic forgetting. However, the traditional centralized approach to CL is unsuitable for end devices due to privacy and data volume concerns. In this context, federated continual learning (FCL) emerges as a promising solution, preserving user data locally while enhancing models through collaborative updates. Aiming at the challenges of limited storage resources for CL, poor autonomy in task shift detection, and difficulty in coping with new adversarial tasks in FCL scenario, we propose a novel FCL framework named SacFL. SacFL employs an Encoder-Decoder architecture to separate task-robust and task-sensitive components, significantly reducing storage demands by retaining lightweight task-sensitive components for resource-constrained end devices. Moreover, $\rm{SacFL}$ leverages contrastive learning to introduce an autonomous data shift detection mechanism, enabling it to discern whether a new task has emerged and whether it is a benign task. This capability ultimately allows the device to autonomously trigger CL or attack defense strategy without additional information, which is more practical for end devices. Comprehensive experiments conducted on multiple text and image datasets, such as Cifar100 and THUCNews, have validated the effectiveness of $\rm{SacFL}$ in both class-incremental and domain-incremental scenarios. Furthermore, a demo system has been developed to verify its practicality.

摘要: 终端设备的激增导致了分布式计算范式，其中设备上的机器学习模型不断处理由这些设备生成的各种数据。这些数据的动态特性（以连续变化或数据漂移为特征）对设备上模型提出了重大挑战。为了解决这个问题，提出了持续学习（CL），使机器学习模型能够增量地更新其知识并减轻灾难性遗忘。然而，由于隐私和数据量问题，CL的传统集中式方法不适合终端设备。在这种背景下，联合持续学习（FCL）成为一种有前途的解决方案，可以在本地保存用户数据，同时通过协作更新增强模型。针对CL存储资源有限、任务转移检测自主性差以及FCL场景下难以应对新的对抗任务的挑战，我们提出了一种新型的FCL框架SacFL。SacFL采用编码器-解码器架构来分离任务稳健组件和任务敏感组件，通过为资源受限的终端设备保留轻量级任务敏感组件来显着减少存储需求。此外，$\rm {SacFL}$利用对比学习引入自主数据漂移检测机制，使其能够辨别新任务是否已出现以及它是否是良性任务。这种能力最终允许设备自主触发CL或攻击防御策略，而无需额外信息，这对于终端设备来说更实用。对多个文本和图像数据集（例如Cifar 100和THUCNews）进行的全面实验验证了$\rm {SacFL}$在类增量和域增量场景中的有效性。此外，还开发了演示系统来验证其实用性。



## **8. TaeBench: Improving Quality of Toxic Adversarial Examples**

TaeBench：提高有毒对抗示例的质量 cs.CR

Accepted for publication in NAACL 2025. The official version will be  available in the ACL Anthology

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2410.05573v2) [paper-pdf](http://arxiv.org/pdf/2410.05573v2)

**Authors**: Xuan Zhu, Dmitriy Bespalov, Liwen You, Ninad Kulkarni, Yanjun Qi

**Abstract**: Toxicity text detectors can be vulnerable to adversarial examples - small perturbations to input text that fool the systems into wrong detection. Existing attack algorithms are time-consuming and often produce invalid or ambiguous adversarial examples, making them less useful for evaluating or improving real-world toxicity content moderators. This paper proposes an annotation pipeline for quality control of generated toxic adversarial examples (TAE). We design model-based automated annotation and human-based quality verification to assess the quality requirements of TAE. Successful TAE should fool a target toxicity model into making benign predictions, be grammatically reasonable, appear natural like human-generated text, and exhibit semantic toxicity. When applying these requirements to more than 20 state-of-the-art (SOTA) TAE attack recipes, we find many invalid samples from a total of 940k raw TAE attack generations. We then utilize the proposed pipeline to filter and curate a high-quality TAE dataset we call TaeBench (of size 264k). Empirically, we demonstrate that TaeBench can effectively transfer-attack SOTA toxicity content moderation models and services. Our experiments also show that TaeBench with adversarial training achieve significant improvements of the robustness of two toxicity detectors.

摘要: 毒性文本检测器可能容易受到对抗性示例的影响--对输入文本的微小扰动会欺骗系统进行错误检测。现有的攻击算法很耗时，并且经常产生无效或模棱两可的对抗示例，使它们对于评估或改进现实世界的毒性内容版主的用处较小。本文提出了一种注释管道，用于对生成的有毒对抗示例（TAE）进行质量控制。我们设计基于模型的自动注释和基于人性的质量验证来评估TAE的质量要求。成功的TAE应该欺骗目标毒性模型做出良性预测，语法合理，看起来像人类生成的文本一样自然，并表现出语义毒性。当将这些要求应用于20多种最先进的（SOTA）TAE攻击配方时，我们在总共94万个原始TAE攻击世代中发现了许多无效样本。然后，我们利用提议的管道来过滤和策划高质量的TAE数据集，我们称之为TaeBench（大小为264 k）。从经验上看，我们证明TaeBench可以有效地转移攻击SOTA毒性内容审核模型和服务。我们的实验还表明，采用对抗训练的TaeBench可以显着提高两个毒性检测器的稳健性。



## **9. SoK: Security and Privacy Risks of Healthcare AI**

SoK：医疗保健AI的安全和隐私风险 cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2409.07415v2) [paper-pdf](http://arxiv.org/pdf/2409.07415v2)

**Authors**: Yuanhaur Chang, Han Liu, Chenyang Lu, Ning Zhang

**Abstract**: The integration of artificial intelligence (AI) and machine learning (ML) into healthcare systems holds great promise for enhancing patient care and care delivery efficiency; however, it also exposes sensitive data and system integrity to potential cyberattacks. Current security and privacy (S&P) research on healthcare AI is highly unbalanced in terms of healthcare deployment scenarios and threat models, and has a disconnected focus with the biomedical research community. This hinders a comprehensive understanding of the risks that healthcare AI entails. To address this gap, this paper takes a thorough examination of existing healthcare AI S&P research, providing a unified framework that allows the identification of under-explored areas. Our survey presents a systematic overview of healthcare AI attacks and defenses, and points out challenges and research opportunities for each AI-driven healthcare application domain. Through our experimental analysis of different threat models and feasibility studies on under-explored adversarial attacks, we provide compelling insights into the pressing need for cybersecurity research in the rapidly evolving field of healthcare AI.

摘要: 将人工智能（AI）和机器学习（ML）集成到医疗保健系统中对于提高患者护理和护理交付效率具有巨大的希望;然而，它也使敏感数据和系统完整性暴露于潜在的网络攻击之下。当前对医疗保健人工智能的安全和隐私（S & P）研究在医疗保健部署场景和威胁模型方面高度不平衡，并且与生物医学研究界的重点脱节。这阻碍了对医疗保健人工智能带来的风险的全面了解。为了解决这一差距，本文彻底审查了现有的医疗保健人工智能S & P研究，提供了一个统一的框架，允许识别未充分开发的领域。我们的调查系统地概述了医疗保健人工智能攻击和防御，并指出了每个人工智能驱动的医疗保健应用领域的挑战和研究机会。通过对不同威胁模型的实验分析以及对未充分探索的对抗性攻击的可行性研究，我们为快速发展的医疗保健人工智能领域网络安全研究的迫切需求提供了令人信服的见解。



## **10. Adversarial Data Poisoning Attacks on Quantum Machine Learning in the NISQ Era**

NISQ时代量子机器学习的对抗性数据中毒攻击 quant-ph

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2411.14412v3) [paper-pdf](http://arxiv.org/pdf/2411.14412v3)

**Authors**: Satwik Kundu, Swaroop Ghosh

**Abstract**: With the growing interest in Quantum Machine Learning (QML) and the increasing availability of quantum computers through cloud providers, addressing the potential security risks associated with QML has become an urgent priority. One key concern in the QML domain is the threat of data poisoning attacks in the current quantum cloud setting. Adversarial access to training data could severely compromise the integrity and availability of QML models. Classical data poisoning techniques require significant knowledge and training to generate poisoned data, and lack noise resilience, making them ineffective for QML models in the Noisy Intermediate Scale Quantum (NISQ) era. In this work, we first propose a simple yet effective technique to measure intra-class encoder state similarity (ESS) by analyzing the outputs of encoding circuits. Leveraging this approach, we introduce a \underline{Qu}antum \underline{I}ndiscriminate \underline{D}ata Poisoning attack, QUID. Through extensive experiments conducted in both noiseless and noisy environments (e.g., IBM\_Brisbane's noise), across various architectures and datasets, QUID achieves up to $92\%$ accuracy degradation in model performance compared to baseline models and up to $75\%$ accuracy degradation compared to random label-flipping. We also tested QUID against state-of-the-art classical defenses, with accuracy degradation still exceeding $50\%$, demonstrating its effectiveness. This work represents the first attempt to reevaluate data poisoning attacks in the context of QML.

摘要: 随着人们对量子机器学习（QML）的兴趣日益浓厚，以及量子计算机通过云提供商的可用性不断增加，解决与QML相关的潜在安全风险已成为当务之急。QML领域的一个关键问题是当前量子云环境中的数据中毒攻击威胁。对训练数据的对抗访问可能会严重损害QML模型的完整性和可用性。经典的数据中毒技术需要大量的知识和训练才能生成中毒数据，并且缺乏噪音弹性，使得它们对于有噪音的中间规模量子（NISQ）时代的QML模型无效。在这项工作中，我们首先提出了一种简单而有效的技术来通过分析编码电路的输出来测量类内编码器状态相似性（ESS）。利用这种方法，我们引入了\underline{Qu}antum \underline{I} nddiscriminate\underline{D}ata中毒攻击，QUID。通过在无噪音和有噪音环境中进行的广泛实验（例如，IBM\_Brisbane的噪音），在各种架构和数据集中，与基线模型相比，QUID的模型性能准确性下降高达92%，与随机标签翻转相比，准确性下降高达75%。我们还针对最先进的经典防御测试了QUID，准确性下降仍然超过50美元，证明了其有效性。这项工作代表了在QML背景下重新评估数据中毒攻击的首次尝试。



## **11. Real AI Agents with Fake Memories: Fatal Context Manipulation Attacks on Web3 Agents**

具有虚假记忆的真实人工智能代理：对Web 3代理的致命上下文操纵攻击 cs.CR

29 pages, 21 figures

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2503.16248v2) [paper-pdf](http://arxiv.org/pdf/2503.16248v2)

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath

**Abstract**: The integration of AI agents with Web3 ecosystems harnesses their complementary potential for autonomy and openness yet also introduces underexplored security risks, as these agents dynamically interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation, a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds.   Through empirical analysis of ElizaOS, a decentralized AI agent framework for automated Web3 operations, we demonstrate how adversaries can manipulate context by injecting malicious instructions into prompts or historical interaction records, leading to unintended asset transfers and protocol violations which could be financially devastating.   To quantify these vulnerabilities, we design CrAIBench, a Web3 domain-specific benchmark that evaluates the robustness of AI agents against context manipulation attacks across 150+ realistic blockchain tasks, including token transfers, trading, bridges and cross-chain interactions and 500+ attack test cases using context manipulation. We systematically assess attack and defense strategies, analyzing factors like the influence of security prompts, reasoning models, and the effectiveness of alignment techniques.   Our findings show that prompt-based defenses are insufficient when adversaries corrupt stored context, achieving significant attack success rates despite these defenses. Fine-tuning-based defenses offer a more robust alternative, substantially reducing attack success rates while preserving utility on single-step tasks. This research highlights the urgent need to develop AI agents that are both secure and fiduciarily responsible.

摘要: 人工智能代理与Web 3生态系统的集成利用了它们在自主性和开放性方面的互补潜力，但也引入了未充分开发的安全风险，因为这些代理与金融协议和不可变的智能合同动态交互。本文研究了基于区块链的金融生态系统中人工智能代理在现实世界场景中面临对抗威胁时的脆弱性。我们引入了上下文操纵的概念，这是一种全面的攻击载体，可以利用不受保护的上下文表面，包括输入通道、内存模块和外部数据源。   通过对ElizaOS（一个用于自动化Web 3操作的去中心化人工智能代理框架）的实证分析，我们展示了对手如何通过将恶意指令注入提示或历史交互记录来操纵上下文，从而导致意外的资产转移和协议违规，这可能在经济上造成毁灭性的。   为了量化这些漏洞，我们设计了CrAIBench，这是一个特定于Web 3领域的基准，用于评估人工智能代理针对150多个现实区块链任务（包括代币传输、交易、桥梁和跨链交互）的上下文操纵攻击的稳健性，以及500多个使用上下文操纵的攻击测试案例。我们系统地评估攻击和防御策略，分析安全提示的影响、推理模型和对齐技术的有效性等因素。   我们的研究结果表明，当对手破坏存储上下文时，基于预算的防御是不够的，尽管有这些防御措施，但仍能实现显着的攻击成功率。基于微调的防御提供了一种更强大的替代方案，可以大幅降低攻击成功率，同时保留对一步任务的实用性。这项研究强调了开发既安全又负信托责任的人工智能代理的迫切需要。



## **12. Stochastic Subspace Descent Accelerated via Bi-fidelity Line Search**

通过双保真线搜索加速随机子空间下降 cs.LG

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2505.00162v1) [paper-pdf](http://arxiv.org/pdf/2505.00162v1)

**Authors**: Nuojin Cheng, Alireza Doostan, Stephen Becker

**Abstract**: Efficient optimization remains a fundamental challenge across numerous scientific and engineering domains, especially when objective function and gradient evaluations are computationally expensive. While zeroth-order optimization methods offer effective approaches when gradients are inaccessible, their practical performance can be limited by the high cost associated with function queries. This work introduces the bi-fidelity stochastic subspace descent (BF-SSD) algorithm, a novel zeroth-order optimization method designed to reduce this computational burden. BF-SSD leverages a bi-fidelity framework, constructing a surrogate model from a combination of computationally inexpensive low-fidelity (LF) and accurate high-fidelity (HF) function evaluations. This surrogate model facilitates an efficient backtracking line search for step size selection, for which we provide theoretical convergence guarantees under standard assumptions. We perform a comprehensive empirical evaluation of BF-SSD across four distinct problems: a synthetic optimization benchmark, dual-form kernel ridge regression, black-box adversarial attacks on machine learning models, and transformer-based black-box language model fine-tuning. Numerical results demonstrate that BF-SSD consistently achieves superior optimization performance while requiring significantly fewer HF function evaluations compared to relevant baseline methods. This study highlights the efficacy of integrating bi-fidelity strategies within zeroth-order optimization, positioning BF-SSD as a promising and computationally efficient approach for tackling large-scale, high-dimensional problems encountered in various real-world applications.

摘要: 有效的优化仍然是众多科学和工程领域的一个根本挑战，特别是当目标函数和梯度评估计算昂贵时。虽然零阶优化方法在无法访问梯度时提供了有效的方法，但其实际性能可能会受到与函数查询相关的高成本的限制。这项工作引入了双保真随机子空间下降（BF-SSD）算法，这是一种新颖的零阶优化方法，旨在减少这种计算负担。BF-SSD利用双保真框架，从计算成本低的低保真度（LF）和准确的高保真度（HF）功能评估的组合中构建代理模型。该代理模型促进了对步骤大小选择的高效回溯线搜索，为此我们在标准假设下提供了理论收敛保证。我们针对四个不同的问题对BF-SSD进行了全面的实证评估：合成优化基准、双重形式内核岭回归、对机器学习模型的黑匣子对抗攻击以及基于转换器的黑匣子语言模型微调。数值结果表明，与相关基线方法相比，BF-SSD始终实现了卓越的优化性能，同时需要的高频功能评估显着减少。这项研究强调了在零阶优化中集成双保真策略的功效，将BF-SSD定位为一种有前途且计算效率高的方法，用于解决各种现实世界应用中遇到的大规模、多维问题。



## **13. WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks**

WASP：针对即时注入攻击的Web代理安全性基准测试 cs.CR

Code and data: https://github.com/facebookresearch/wasp

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.18575v2) [paper-pdf](http://arxiv.org/pdf/2504.18575v2)

**Authors**: Ivan Evtimov, Arman Zharmagambetov, Aaron Grattafiori, Chuan Guo, Kamalika Chaudhuri

**Abstract**: Web navigation AI agents use language-and-vision foundation models to enhance productivity but these models are known to be susceptible to indirect prompt injections that get them to follow instructions different from the legitimate user's. Existing explorations of this threat applied to web agents often focus on a single isolated adversarial goal, test with injected instructions that are either too easy or not truly malicious, and often give the adversary unreasonable access. In order to better focus adversarial research, we construct a new benchmark called WASP (Web Agent Security against Prompt injection attacks) that introduces realistic web agent hijacking objectives and an isolated environment to test them in that does not affect real users or the live web. As part of WASP, we also develop baseline attacks against popular web agentic systems (VisualWebArena, Claude Computer Use, etc.) instantiated with various state-of-the-art models. Our evaluation shows that even AI agents backed by models with advanced reasoning capabilities and by models with instruction hierarchy mitigations are susceptible to low-effort human-written prompt injections. However, the realistic objectives in WASP also allow us to observe that agents are currently not capable enough to complete the goals of attackers end-to-end. Agents begin executing the adversarial instruction between 16 and 86% of the time but only achieve the goal between 0 and 17% of the time. Based on these findings, we argue that adversarial researchers should demonstrate stronger attacks that more consistently maintain control over the agent given realistic constraints on the adversary's power.

摘要: 网络导航人工智能代理使用语言和视觉基础模型来提高生产力，但众所周知，这些模型容易受到间接提示注入的影响，使它们遵循与合法用户不同的指令。应用于Web代理的这种威胁的现有探索通常集中在单个孤立的对抗目标上，使用太容易或并非真正恶意的注入指令进行测试，并且通常为对手提供不合理的访问权限。为了更好地关注对抗性研究，我们构建了一个名为WISP（针对提示注入攻击的Web代理安全性）的新基准，该基准引入了现实的Web代理劫持目标和一个隔离的环境来测试它们，并且不会影响真实用户或实时网络。作为WISP的一部分，我们还开发针对流行的网络代理系统（Visual WebArena、Claude Computer Use等）的基线攻击用各种最先进的模型实例化。我们的评估表明，即使是由具有高级推理能力的模型和具有指令层次缓解的模型支持的人工智能代理，也容易受到低努力的人工编写提示注入的影响。然而，WSP中的现实目标也让我们观察到，代理目前没有足够的能力完成攻击者的端到端的目标。代理在16%到86%的时间内开始执行对抗指令，但仅在0%到17%的时间内实现目标。基于这些发现，我们认为，在对对手力量的现实限制的情况下，对抗性研究人员应该展示更强的攻击，以更一致地保持对代理人的控制。



## **14. Active Light Modulation to Counter Manipulation of Speech Visual Content**

主动光调制以对抗语音视觉内容的操纵 cs.CV

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21846v1) [paper-pdf](http://arxiv.org/pdf/2504.21846v1)

**Authors**: Hadleigh Schwartz, Xiaofeng Yan, Charles J. Carver, Xia Zhou

**Abstract**: High-profile speech videos are prime targets for falsification, owing to their accessibility and influence. This work proposes Spotlight, a low-overhead and unobtrusive system for protecting live speech videos from visual falsification of speaker identity and lip and facial motion. Unlike predominant falsification detection methods operating in the digital domain, Spotlight creates dynamic physical signatures at the event site and embeds them into all video recordings via imperceptible modulated light. These physical signatures encode semantically-meaningful features unique to the speech event, including the speaker's identity and facial motion, and are cryptographically-secured to prevent spoofing. The signatures can be extracted from any video downstream and validated against the portrayed speech content to check its integrity. Key elements of Spotlight include (1) a framework for generating extremely compact (i.e., 150-bit), pose-invariant speech video features, based on locality-sensitive hashing; and (2) an optical modulation scheme that embeds >200 bps into video while remaining imperceptible both in video and live. Prototype experiments on extensive video datasets show Spotlight achieves AUCs $\geq$ 0.99 and an overall true positive rate of 100% in detecting falsified videos. Further, Spotlight is highly robust across recording conditions, video post-processing techniques, and white-box adversarial attacks on its video feature extraction methodologies.

摘要: 由于备受瞩目的演讲视频的可访问性和影响力，因此成为伪造的主要目标。这项工作提出了Spotlight，这是一种低成本且不引人注目的系统，用于保护现场语音视频免受说话者身份以及嘴唇和面部运动的视觉伪造。与数字领域中运行的主要伪造检测方法不同，Spotlight在活动现场创建动态物理签名，并通过不可感知的调制光将其嵌入所有视频记录中。这些物理签名编码语音事件特有的具有语义意义的特征，包括说话者的身份和面部动作，并且经过加密保护以防止欺骗。签名可以从任何视频下游提取，并根据所描绘的语音内容进行验证，以检查其完整性。聚光灯的关键要素包括（1）一个框架，用于生成极其紧凑的（即，150位）、基于位置敏感散列的姿态不变语音视频特征;以及（2）光学调制方案，其将>200 bps嵌入到视频中，同时在视频和实况中保持不可感知。在广泛的视频数据集上进行的原型实验表明，Spotlight在检测伪造视频时达到了AUC $\geq $0.99和100%的总体真阳性率。此外，Spotlight在记录条件、视频后处理技术以及对其视频特征提取方法的白盒对抗攻击方面具有高度稳健性。



## **15. Adversarial KA**

对手KA cs.LG

8 pages, 3 figures; minor revision, question 4.1 added

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.05255v2) [paper-pdf](http://arxiv.org/pdf/2504.05255v2)

**Authors**: Sviatoslav Dzhenzher, Michael H. Freedman

**Abstract**: Regarding the representation theorem of Kolmogorov and Arnold (KA) as an algorithm for representing or {\guillemotleft}expressing{\guillemotright} functions, we test its robustness by analyzing its ability to withstand adversarial attacks. We find KA to be robust to countable collections of continuous adversaries, but unearth a question about the equi-continuity of the outer functions that, so far, obstructs taking limits and defeating continuous groups of adversaries. This question on the regularity of the outer functions is relevant to the debate over the applicability of KA to the general theory of NNs.

摘要: 将Kolmogorov和Arnold（KA）的表示定理视为表示或{\guillemotleft}表达{\guillemotleft}函数的算法，我们通过分析其抵御对抗攻击的能力来测试其稳健性。我们发现KA对于连续对手的可计数集合来说是稳健的，但我们发现了一个关于外部函数的等连续性的问题，到目前为止，该问题阻碍了采取限制和击败连续对手组。这个关于外部函数规律性的问题与KA对NN一般理论的适用性的争论有关。



## **16. Hoist with His Own Petard: Inducing Guardrails to Facilitate Denial-of-Service Attacks on Retrieval-Augmented Generation of LLMs**

用自己的花瓣提升：引入护栏以促进对检索增强一代LLM的拒绝服务攻击 cs.CR

11 pages, 6 figures. This work will be submitted to the IEEE for  possible publication

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21680v1) [paper-pdf](http://arxiv.org/pdf/2504.21680v1)

**Authors**: Pan Suo, Yu-Ming Shang, San-Chuan Guo, Xi Zhang

**Abstract**: Retrieval-Augmented Generation (RAG) integrates Large Language Models (LLMs) with external knowledge bases, improving output quality while introducing new security risks. Existing studies on RAG vulnerabilities typically focus on exploiting the retrieval mechanism to inject erroneous knowledge or malicious texts, inducing incorrect outputs. However, these approaches overlook critical weaknesses within LLMs, leaving important attack vectors unexplored and limiting the scope and efficiency of attacks. In this paper, we uncover a novel vulnerability: the safety guardrails of LLMs, while designed for protection, can also be exploited as an attack vector by adversaries. Building on this vulnerability, we propose MutedRAG, a novel denial-of-service attack that reversely leverages the guardrails of LLMs to undermine the availability of RAG systems. By injecting minimalistic jailbreak texts, such as "\textit{How to build a bomb}", into the knowledge base, MutedRAG intentionally triggers the LLM's safety guardrails, causing the system to reject legitimate queries. Besides, due to the high sensitivity of guardrails, a single jailbreak sample can affect multiple queries, effectively amplifying the efficiency of attacks while reducing their costs. Experimental results on three datasets demonstrate that MutedRAG achieves an attack success rate exceeding 60% in many scenarios, requiring only less than one malicious text to each target query on average. In addition, we evaluate potential defense strategies against MutedRAG, finding that some of current mechanisms are insufficient to mitigate this threat, underscoring the urgent need for more robust solutions.

摘要: 检索增强生成（RAG）将大型语言模型（LLM）与外部知识库集成，提高输出质量，同时引入新的安全风险。现有关于RAG漏洞的研究通常集中在利用检索机制注入错误知识或恶意文本，从而引发错误的输出。然而，这些方法忽视了LLM中的关键弱点，导致重要的攻击载体未被探索，并限制了攻击的范围和效率。在本文中，我们发现了一个新颖的漏洞：LLM的安全护栏虽然是为了保护而设计的，但也可能被对手用作攻击载体。在此漏洞的基础上，我们提出了MutedRAG，一种新型的拒绝服务攻击，它利用LLM的护栏来破坏RAG系统的可用性。通过向知识库中注入极简的越狱文本，例如“\textit{How to build a bomb}"，MutedRAG故意触发LLM的安全护栏，导致系统拒绝合法查询。此外，由于护栏的高度敏感性，单个越狱样本可以影响多个查询，有效地放大了攻击的效率，同时降低了攻击的成本。在三个数据集上的实验结果表明，MutedRAG在许多场景下实现了超过60%的攻击成功率，平均每个目标查询只需要不到一个恶意文本。此外，我们评估了针对MutedRAG的潜在防御策略，发现当前的一些机制不足以减轻这种威胁，这凸显了迫切需要更强大的解决方案。



## **17. Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection**

基于扩散的对抗性身份操纵用于面部隐私保护 cs.CV

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21646v1) [paper-pdf](http://arxiv.org/pdf/2504.21646v1)

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.

摘要: 由于社交网络上潜在的未经授权的监视和用户跟踪，面部识别（FR）系统的成功引发了严重的隐私问题。现有的增强隐私的方法无法生成可以保护面部隐私的自然面部图像。在本文中，我们提出了基于扩散的对抗身份操纵（DiffAIM）来生成针对恶意FR系统的自然且高度可转移的对抗面孔。具体来说，我们在扩散模型的低维潜在空间内操纵面部身份。这涉及在反向扩散过程中迭代地注入基于梯度的对抗性身份指导，逐步引导一代人走向所需的对抗性面孔。该指南针对向目标的身份融合进行了优化，同时促进源自源头的语义分歧，促进有效模仿，同时保持视觉自然性。我们进一步结合了结构保留的正规化，以在操作过程中保持面部结构一致性。针对人脸验证和识别任务的大量实验表明，与最新技术相比，迪夫AIM实现了更强的黑匣子攻击可转移性，同时保持了卓越的视觉质量。我们还证明了所提出的方法对商业FR API（包括Face++和Aliyun）的有效性。



## **18. Generative AI in Financial Institution: A Global Survey of Opportunities, Threats, and Regulation**

金融机构中的生成人工智能：机会、威胁和监管的全球调查 cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21574v1) [paper-pdf](http://arxiv.org/pdf/2504.21574v1)

**Authors**: Bikash Saha, Nanda Rani, Sandeep Kumar Shukla

**Abstract**: Generative Artificial Intelligence (GenAI) is rapidly reshaping the global financial landscape, offering unprecedented opportunities to enhance customer engagement, automate complex workflows, and extract actionable insights from vast financial data. This survey provides an overview of GenAI adoption across the financial ecosystem, examining how banks, insurers, asset managers, and fintech startups worldwide are integrating large language models and other generative tools into their operations. From AI-powered virtual assistants and personalized financial advisory to fraud detection and compliance automation, GenAI is driving innovation across functions. However, this transformation comes with significant cybersecurity and ethical risks. We discuss emerging threats such as AI-generated phishing, deepfake-enabled fraud, and adversarial attacks on AI systems, as well as concerns around bias, opacity, and data misuse. The evolving global regulatory landscape is explored in depth, including initiatives by major financial regulators and international efforts to develop risk-based AI governance. Finally, we propose best practices for secure and responsible adoption - including explainability techniques, adversarial testing, auditability, and human oversight. Drawing from academic literature, industry case studies, and policy frameworks, this chapter offers a perspective on how the financial sector can harness GenAI's transformative potential while navigating the complex risks it introduces.

摘要: 生成式人工智能（GenAI）正在迅速重塑全球金融格局，为增强客户参与度、自动化复杂的工作流程以及从大量金融数据中提取可操作的见解提供了前所未有的机会。该调查概述了整个金融生态系统中GenAI的采用情况，研究了全球银行，保险公司，资产管理公司和金融科技初创公司如何将大型语言模型和其他生成工具集成到其运营中。从人工智能驱动的虚拟助理和个性化财务咨询到欺诈检测和合规自动化，GenAI正在推动跨职能的创新。然而，这种转变伴随着重大的网络安全和道德风险。我们讨论了人工智能生成的网络钓鱼、深度伪造的欺诈和对人工智能系统的对抗攻击等新兴威胁，以及对偏见、不透明和数据滥用的担忧。深入探讨了不断变化的全球监管格局，包括主要金融监管机构的举措以及国际上发展基于风险的人工智能治理的努力。最后，我们提出了安全且负责任的采用的最佳实践-包括可解释性技术、对抗性测试、可互换性和人类监督。本章借鉴学术文献、行业案例研究和政策框架，提供了金融部门如何利用GenAI的变革潜力，同时应对其带来的复杂风险的视角。



## **19. A Test Suite for Efficient Robustness Evaluation of Face Recognition Systems**

人脸识别系统高效鲁棒性评估的测试套件 cs.SE

IEEE Transactions on Reliability

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21420v1) [paper-pdf](http://arxiv.org/pdf/2504.21420v1)

**Authors**: Ruihan Zhang, Jun Sun

**Abstract**: Face recognition is a widely used authentication technology in practice, where robustness is required. It is thus essential to have an efficient and easy-to-use method for evaluating the robustness of (possibly third-party) trained face recognition systems. Existing approaches to evaluating the robustness of face recognition systems are either based on empirical evaluation (e.g., measuring attacking success rate using state-of-the-art attacking methods) or formal analysis (e.g., measuring the Lipschitz constant). While the former demands significant user efforts and expertise, the latter is extremely time-consuming. In pursuit of a comprehensive, efficient, easy-to-use and scalable estimation of the robustness of face recognition systems, we take an old-school alternative approach and introduce RobFace, i.e., evaluation using an optimised test suite. It contains transferable adversarial face images that are designed to comprehensively evaluate a face recognition system's robustness along a variety of dimensions. RobFace is system-agnostic and still consistent with system-specific empirical evaluation or formal analysis. We support this claim through extensive experimental results with various perturbations on multiple face recognition systems. To our knowledge, RobFace is the first system-agnostic robustness estimation test suite.

摘要: 人脸识别是实践中广泛使用的认证技术，需要鲁棒性。因此，拥有一种高效且易于使用的方法来评估（可能是第三方）训练的人脸识别系统的稳健性至关重要。评估人脸识别系统稳健性的现有方法要么基于经验评估（例如，使用最先进的攻击方法测量攻击成功率）或正式分析（例如，测量Lipschitz常数）。前者需要用户付出大量努力和专门知识，而后者极其耗时。为了追求对人脸识别系统的鲁棒性进行全面、高效、易于使用和可扩展的估计，我们采用了一种老派的替代方法，并引入了RobFace，即，使用优化的测试套件进行评估。它包含可转移的对抗性人脸图像，旨在全面评估人脸识别系统在各个维度上的鲁棒性。RobFace是系统不可知的，仍然与系统特定的经验评估或形式分析相一致。我们支持这一说法，通过广泛的实验结果与各种扰动多人脸识别系统。据我们所知，RobFace是第一个与系统无关的鲁棒性估计测试套件。



## **20. Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning**

在联邦学习中利用防御基于GAN的特征推理攻击 cs.CR

Published in ACM Transactions on Knowledge Discovery from Data  (TKDD), 2025

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2004.12571v5) [paper-pdf](http://arxiv.org/pdf/2004.12571v5)

**Authors**: Xinjian Luo, Xianglong Zhang

**Abstract**: Federated learning (FL) is a decentralized model training framework that aims to merge isolated data islands while maintaining data privacy. However, recent studies have revealed that Generative Adversarial Network (GAN) based attacks can be employed in FL to learn the distribution of private datasets and reconstruct recognizable images. In this paper, we exploit defenses against GAN-based attacks in FL and propose a framework, Anti-GAN, to prevent attackers from learning the real distribution of the victim's data. The core idea of Anti-GAN is to manipulate the visual features of private training images to make them indistinguishable to human eyes even restored by attackers. Specifically, Anti-GAN projects the private dataset onto a GAN's generator and combines the generated fake images with the actual images to create the training dataset, which is then used for federated model training. The experimental results demonstrate that Anti-GAN is effective in preventing attackers from learning the distribution of private images while causing minimal harm to the accuracy of the federated model.

摘要: 联邦学习（FL）是一个去中心化的模型训练框架，旨在合并孤立的数据岛，同时维护数据隐私。然而，最近的研究表明，基于生成对抗网络（GAN）的攻击可以用于FL来了解私人数据集的分布并重建可识别图像。在本文中，我们利用FL中针对基于GAN的攻击的防御措施，并提出了一个框架Anti-GAN，以防止攻击者了解受害者数据的真实分布。Anti-GAN的核心思想是操纵私人训练图像的视觉特征，使其即使被攻击者恢复，人眼也无法区分。具体来说，Anti-GAN将私人数据集投影到GAN的生成器上，并将生成的假图像与实际图像结合起来创建训练数据集，然后将其用于联邦模型训练。实验结果表明，Anti-GAN可以有效防止攻击者学习私人图像的分布，同时对联邦模型的准确性造成最小的伤害。



## **21. How to Backdoor the Knowledge Distillation**

如何后门知识提炼 cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21323v1) [paper-pdf](http://arxiv.org/pdf/2504.21323v1)

**Authors**: Chen Wu, Qian Ma, Prasenjit Mitra, Sencun Zhu

**Abstract**: Knowledge distillation has become a cornerstone in modern machine learning systems, celebrated for its ability to transfer knowledge from a large, complex teacher model to a more efficient student model. Traditionally, this process is regarded as secure, assuming the teacher model is clean. This belief stems from conventional backdoor attacks relying on poisoned training data with backdoor triggers and attacker-chosen labels, which are not involved in the distillation process. Instead, knowledge distillation uses the outputs of a clean teacher model to guide the student model, inherently preventing recognition or response to backdoor triggers as intended by an attacker. In this paper, we challenge this assumption by introducing a novel attack methodology that strategically poisons the distillation dataset with adversarial examples embedded with backdoor triggers. This technique allows for the stealthy compromise of the student model while maintaining the integrity of the teacher model. Our innovative approach represents the first successful exploitation of vulnerabilities within the knowledge distillation process using clean teacher models. Through extensive experiments conducted across various datasets and attack settings, we demonstrate the robustness, stealthiness, and effectiveness of our method. Our findings reveal previously unrecognized vulnerabilities and pave the way for future research aimed at securing knowledge distillation processes against backdoor attacks.

摘要: 知识蒸馏已成为现代机器学习系统的基石，以其将知识从大型、复杂的教师模型转移到更高效的学生模型的能力而闻名。传统上，假设教师模型是干净的，这个过程被认为是安全的。这种信念源于传统的后门攻击，这种攻击依赖于带有后门触发器和攻击者选择的标签的有毒训练数据，这些标签不参与蒸馏过程。相反，知识蒸馏使用干净教师模型的输出来指导学生模型，从本质上阻止了攻击者意图的对后门触发器的识别或响应。在本文中，我们通过引入一种新颖的攻击方法来挑战这一假设，该方法通过嵌入后门触发器的对抗示例战略性地毒害蒸馏数据集。该技术允许学生模型的秘密妥协，同时保持教师模型的完整性。我们的创新方法代表了使用廉洁教师模型在知识提炼过程中首次成功利用漏洞。通过在各种数据集和攻击设置中进行的广泛实验，我们证明了我们方法的稳健性、隐蔽性和有效性。我们的研究结果揭示了之前未被识别的漏洞，并为未来旨在保护知识提炼过程免受后门攻击的研究铺平了道路。



## **22. Round Trip Translation Defence against Large Language Model Jailbreaking Attacks**

针对大型语言模型越狱攻击的往返翻译防御 cs.CL

6 pages, 6 figures

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2402.13517v2) [paper-pdf](http://arxiv.org/pdf/2402.13517v2)

**Authors**: Canaan Yung, Hadi Mohaghegh Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Large language models (LLMs) are susceptible to social-engineered attacks that are human-interpretable but require a high level of comprehension for LLMs to counteract. Existing defensive measures can only mitigate less than half of these attacks at most. To address this issue, we propose the Round Trip Translation (RTT) method, the first algorithm specifically designed to defend against social-engineered attacks on LLMs. RTT paraphrases the adversarial prompt and generalizes the idea conveyed, making it easier for LLMs to detect induced harmful behavior. This method is versatile, lightweight, and transferrable to different LLMs. Our defense successfully mitigated over 70% of Prompt Automatic Iterative Refinement (PAIR) attacks, which is currently the most effective defense to the best of our knowledge. We are also the first to attempt mitigating the MathsAttack and reduced its attack success rate by almost 40%. Our code is publicly available at https://github.com/Cancanxxx/Round_Trip_Translation_Defence   This version of the article has been accepted for publication, after peer review (when applicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The Version of Record is available online at: https://doi.org/10.48550/arXiv.2402.13517 Use of this Accepted Version is subject to the publisher's Accepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscript-terms

摘要: 大型语言模型（LLM）很容易受到人类可解释的社会工程攻击，但LLM需要高水平的理解力才能对抗。现有的防御措施最多只能减轻不到一半的攻击。为了解决这个问题，我们提出了往返翻译（RTI）方法，这是第一个专门设计用于防御对LLM的社会工程攻击的算法。HRT解释了对抗提示并概括了所传达的想法，使LLM更容易检测诱导的有害行为。该方法通用、轻量级，并且可转移到不同的LLM。我们的防御成功缓解了超过70%的提示自动迭代细化（PAIR）攻击，据我们所知，这是目前最有效的防御。我们也是第一个尝试缓解MathsAttack的公司，并将其攻击成功率降低了近40%。我们的代码可在https://github.com/Cancanxxx/Round_Trip_Translation_Defence上公开获取   经过同行评审（如果适用）后，该版本的文章已被接受出版，但不是记录版本，并且不反映接受后的改进或任何更正。记录版本可在线获取：https://doi.org/10.48550/arXiv.2402.13517此接受版本的使用须遵守出版商的接受手稿使用条款https://www.springernature.com/gp/open-research/policies/accepted-manuscript-terms



## **23. Quantifying the Noise of Structural Perturbations on Graph Adversarial Attacks**

量化图对抗攻击的结构扰动噪音 cs.LG

Under Review

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.20869v2) [paper-pdf](http://arxiv.org/pdf/2504.20869v2)

**Authors**: Junyuan Fang, Han Yang, Haixian Wen, Jiajing Wu, Zibin Zheng, Chi K. Tse

**Abstract**: Graph neural networks have been widely utilized to solve graph-related tasks because of their strong learning power in utilizing the local information of neighbors. However, recent studies on graph adversarial attacks have proven that current graph neural networks are not robust against malicious attacks. Yet much of the existing work has focused on the optimization objective based on attack performance to obtain (near) optimal perturbations, but paid less attention to the strength quantification of each perturbation such as the injection of a particular node/link, which makes the choice of perturbations a black-box model that lacks interpretability. In this work, we propose the concept of noise to quantify the attack strength of each adversarial link. Furthermore, we propose three attack strategies based on the defined noise and classification margins in terms of single and multiple steps optimization. Extensive experiments conducted on benchmark datasets against three representative graph neural networks demonstrate the effectiveness of the proposed attack strategies. Particularly, we also investigate the preferred patterns of effective adversarial perturbations by analyzing the corresponding properties of the selected perturbation nodes.

摘要: 图神经网络因其在利用邻居的局部信息方面具有强大的学习能力，已被广泛用于解决与图相关的任务。然而，最近关于图对抗攻击的研究证明，当前的图神经网络对恶意攻击并不强大。然而，现有的大部分工作都集中在基于攻击性能的优化目标上，以获得（接近）最佳扰动，但较少关注每个扰动的强度量化，例如特定节点/链路的注入，这使得扰动的选择成为缺乏可解释性的黑匣子模型。在这项工作中，我们提出了噪音的概念来量化每个对抗链接的攻击强度。此外，我们根据定义的噪音和分类裕度提出了三种攻击策略，从单步和多步优化角度进行。针对三个代表性图神经网络在基准数据集上进行的大量实验证明了所提出的攻击策略的有效性。特别是，我们还通过分析所选扰动节点的相应属性来研究有效对抗扰动的首选模式。



## **24. Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training**

铁磨铁：通过对抗训练防御机器生成文本检测中的攻击 cs.CR

Accepted by ACL 2025 Main Conference

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2502.12734v2) [paper-pdf](http://arxiv.org/pdf/2502.12734v2)

**Authors**: Yuanfan Li, Zhaohan Zhang, Chengzhengxu Li, Chao Shen, Xiaoming Liu

**Abstract**: Machine-generated Text (MGT) detection is crucial for regulating and attributing online texts. While the existing MGT detectors achieve strong performance, they remain vulnerable to simple perturbations and adversarial attacks. To build an effective defense against malicious perturbations, we view MGT detection from a threat modeling perspective, that is, analyzing the model's vulnerability from an adversary's point of view and exploring effective mitigations. To this end, we introduce an adversarial framework for training a robust MGT detector, named GREedy Adversary PromoTed DefendER (GREATER). The GREATER consists of two key components: an adversary GREATER-A and a detector GREATER-D. The GREATER-D learns to defend against the adversarial attack from GREATER-A and generalizes the defense to other attacks. GREATER-A identifies and perturbs the critical tokens in embedding space, along with greedy search and pruning to generate stealthy and disruptive adversarial examples. Besides, we update the GREATER-A and GREATER-D synchronously, encouraging the GREATER-D to generalize its defense to different attacks and varying attack intensities. Our experimental results across 10 text perturbation strategies and 6 adversarial attacks show that our GREATER-D reduces the Attack Success Rate (ASR) by 0.67% compared with SOTA defense methods while our GREATER-A is demonstrated to be more effective and efficient than SOTA attack approaches. Codes and dataset are available in https://github.com/Liyuuuu111/GREATER.

摘要: 机器生成文本（MGT）检测对于监管和归属在线文本至关重要。虽然现有的MGT检测器实现了出色的性能，但它们仍然容易受到简单扰动和对抗攻击的影响。为了建立针对恶意干扰的有效防御，我们从威胁建模的角度来看待MGT检测，即从对手的角度分析模型的漏洞并探索有效的缓解措施。为此，我们引入了一个对抗框架来训练稳健的MGT检测器，名为GREedy Adminster PromTed DefendER（GREATER）。GREATER由两个关键组件组成：对手GREATER-A和探测器GREATER-D。GREATER-D学会防御GREATER-A的对抗攻击，并将防御推广到其他攻击。GREATER-A识别和扰乱嵌入空间中的关键令牌，同时进行贪婪搜索和修剪，以生成隐秘且破坏性的对抗示例。此外，我们同步更新了GREATER-A和GREATER-D，鼓励GREATER-D将其防御推广到不同的攻击和不同的攻击强度。我们对10种文本扰动策略和6种对抗性攻击的实验结果表明，与SOTA防御方法相比，我们的GREATER-D将攻击成功率（ASB）降低了0.67%，而我们的GREATER-A被证明比SOTA攻击方法更有效和高效。代码和数据集可在https://github.com/Liyuuuu111/GREATER上获取。



## **25. Generate-then-Verify: Reconstructing Data from Limited Published Statistics**

生成然后验证：从有限的已发布统计数据重建数据 stat.ML

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.21199v1) [paper-pdf](http://arxiv.org/pdf/2504.21199v1)

**Authors**: Terrance Liu, Eileen Xiao, Pratiksha Thaker, Adam Smith, Zhiwei Steven Wu

**Abstract**: We study the problem of reconstructing tabular data from aggregate statistics, in which the attacker aims to identify interesting claims about the sensitive data that can be verified with 100% certainty given the aggregates. Successful attempts in prior work have conducted studies in settings where the set of published statistics is rich enough that entire datasets can be reconstructed with certainty. In our work, we instead focus on the regime where many possible datasets match the published statistics, making it impossible to reconstruct the entire private dataset perfectly (i.e., when approaches in prior work fail). We propose the problem of partial data reconstruction, in which the goal of the adversary is to instead output a $\textit{subset}$ of rows and/or columns that are $\textit{guaranteed to be correct}$. We introduce a novel integer programming approach that first $\textbf{generates}$ a set of claims and then $\textbf{verifies}$ whether each claim holds for all possible datasets consistent with the published aggregates. We evaluate our approach on the housing-level microdata from the U.S. Decennial Census release, demonstrating that privacy violations can still persist even when information published about such data is relatively sparse.

摘要: 我们研究了从汇总统计数据中重建表格数据的问题，其中攻击者的目标是确定有关敏感数据的有趣声明，这些敏感数据可以100%确定地进行验证。在先前的工作中，成功的尝试是在已发布的统计数据集足够丰富的环境中进行研究，可以确定地重建整个数据集。在我们的工作中，我们专注于许多可能的数据集与已发布的统计数据相匹配的情况，这使得不可能完美地重建整个私有数据集（即，当先前工作中的方法失败时）。我们提出了部分数据重建的问题，其中对手的目标是输出$\textit{subset}$的行和/或列，这些行和/或列为$\textit{保证正确}$。我们引入了一种新颖的整元规划方法，首先$\textBF{generate}$一组声明，然后$\textBF{verify}$每个声明是否适用于与已发布的聚合物一致的所有可能数据集。我们评估了我们对美国十年一次人口普查发布的住房级微数据的方法，证明即使有关此类数据的信息相对稀疏，隐私侵犯仍然可能持续存在。



## **26. Demo: ViolentUTF as An Accessible Platform for Generative AI Red Teaming**

演示：ViolentUTF作为生成性AI Red团队的可用平台 cs.CR

3 pages, 1 figure, 1 table. This is a demo paper for  CyberWarrior2025. The video demo is at https://youtu.be/c-UCYXq0rfY. Codes  will be shared when the competition concludes in June 2025 due to embargo  requirements

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.10603v2) [paper-pdf](http://arxiv.org/pdf/2504.10603v2)

**Authors**: Tam n. Nguyen

**Abstract**: The rapid integration of Generative AI (GenAI) into various applications necessitates robust risk management strategies which includes Red Teaming (RT) - an evaluation method for simulating adversarial attacks. Unfortunately, RT for GenAI is often hindered by technical complexity, lack of user-friendly interfaces, and inadequate reporting features. This paper introduces Violent UTF - an accessible, modular, and scalable platform for GenAI red teaming. Through intuitive interfaces (Web GUI, CLI, API, MCP) powered by LLMs and for LLMs, Violent UTF aims to empower non-technical domain experts and students alongside technical experts, facilitate comprehensive security evaluation by unifying capabilities from RT frameworks like Microsoft PyRIT, Nvidia Garak and its own specialized evaluators. ViolentUTF is being used for evaluating the robustness of a flagship LLM-based product in a large US Government department. It also demonstrates effectiveness in evaluating LLMs' cross-domain reasoning capability between cybersecurity and behavioral psychology.

摘要: 将生成式AI（GenAI）快速集成到各种应用程序中需要强大的风险管理策略，其中包括Red Teaming（RT）-一种用于模拟对抗性攻击的评估方法。不幸的是，RT for GenAI经常受到技术复杂性，缺乏用户友好界面和报告功能不足的阻碍。本文介绍Violent UTF -一个可访问的，模块化的，可扩展的GenAI红色团队平台。通过由LLM和LLM提供支持的直观界面（Web GUI，CLI，API，MCP），Violent UTF旨在为非技术领域专家和学生以及技术专家提供支持，通过统一Microsoft PyRIT，Nvidia Garak等RT框架的功能以及自己的专业评估人员来促进全面的安全评估。ViolentUTF正在用于评估美国大型政府部门基于LLM的旗舰产品的稳健性。它还展示了评估LLM网络安全和行为心理学之间跨领域推理能力的有效性。



## **27. AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security**

AegisLLM：扩展统计系统以实现LLM安全中的自我反思防御 cs.LG

ICLR 2025 Workshop BuildingTrust

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20965v1) [paper-pdf](http://arxiv.org/pdf/2504.20965v1)

**Authors**: Zikui Cai, Shayan Shabihi, Bang An, Zora Che, Brian R. Bartoldson, Bhavya Kailkhura, Tom Goldstein, Furong Huang

**Abstract**: We introduce AegisLLM, a cooperative multi-agent defense against adversarial attacks and information leakage. In AegisLLM, a structured workflow of autonomous agents - orchestrator, deflector, responder, and evaluator - collaborate to ensure safe and compliant LLM outputs, while self-improving over time through prompt optimization. We show that scaling agentic reasoning system at test-time - both by incorporating additional agent roles and by leveraging automated prompt optimization (such as DSPy)- substantially enhances robustness without compromising model utility. This test-time defense enables real-time adaptability to evolving attacks, without requiring model retraining. Comprehensive evaluations across key threat scenarios, including unlearning and jailbreaking, demonstrate the effectiveness of AegisLLM. On the WMDP unlearning benchmark, AegisLLM achieves near-perfect unlearning with only 20 training examples and fewer than 300 LM calls. For jailbreaking benchmarks, we achieve 51% improvement compared to the base model on StrongReject, with false refusal rates of only 7.9% on PHTest compared to 18-55% for comparable methods. Our results highlight the advantages of adaptive, agentic reasoning over static defenses, establishing AegisLLM as a strong runtime alternative to traditional approaches based on model modifications. Code is available at https://github.com/zikuicai/aegisllm

摘要: 我们引入了AegisLLM，这是一种针对对抗攻击和信息泄露的协作多代理防御系统。在AegisLLM中，由自治代理（协调器、偏转器、响应者和评估器）组成的结构化工作流程相互协作，以确保安全合规的LLM输出，同时通过及时优化随着时间的推移进行自我改进。我们表明，在测试时扩展代理推理系统--通过合并额外的代理角色和利用自动化提示优化（例如DSPy）--可以在不损害模型效用的情况下大幅增强稳健性。这种测试时防御能够实时适应不断发展的攻击，而无需模型重新训练。对关键威胁场景（包括取消学习和越狱）的全面评估展示了AegisLLM的有效性。在WMDP取消学习基准上，AegisLLM仅使用20个训练示例和少于300个LM调用即可实现近乎完美的取消学习。对于越狱基准，与Strongestival上的基本模型相比，我们实现了51%的改进，PHTest上的错误拒绝率仅为7.9%，而类似方法的错误拒绝率为18-55%。我们的结果强调了自适应、代理推理相对于静态防御的优势，将AegisLLM确立为基于模型修改的传统方法的强大运行时替代方案。代码可在https://github.com/zikuicai/aegisllm上获得



## **28. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

NoPain：通过最佳传输奇异边界进行无箱点云攻击 cs.CV

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2503.00063v4) [paper-pdf](http://arxiv.org/pdf/2503.00063v4)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. Code and model are available at https://github.com/cognaclee/nopain

摘要: 对抗性攻击利用深度模型针对对抗性样本的脆弱性。现有的点云攻击者针对特定模型进行定制，根据白盒或黑盒设置中的梯度迭代优化扰动。尽管它们的攻击性能令人鼓舞，但由于过度匹配代理模型的特定参数，它们经常难以产生可转移的对抗样本。为了克服这个问题，我们将重点转移到数据分布本身上，并引入了一种名为NoPain的新颖方法，该方法采用最优传输（OT）来识别跨网络点云攻击的数据集的固有奇异边界。具体来说，我们首先计算从噪音到目标特征空间的OT映射，然后通过定位不可微位置来识别奇异边界。最后，我们沿着奇异边界进行采样以生成对抗点云。一旦确定奇异边界，NoPain就可以有效地生成对抗样本，而无需迭代更新或代理分类器的指导。大量实验表明，提出的端到端方法在可移植性和效率方面都优于基线方法，同时即使在防御策略方面也保持了显着的优势。代码和型号可在https://github.com/cognaclee/nopain上获得



## **29. Erased but Not Forgotten: How Backdoors Compromise Concept Erasure**

被擦除但没有忘记：后门如何损害概念擦除 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.21072v1) [paper-pdf](http://arxiv.org/pdf/2504.21072v1)

**Authors**: Jonas Henry Grebe, Tobias Braun, Marcus Rohrbach, Anna Rohrbach

**Abstract**: The expansion of large-scale text-to-image diffusion models has raised growing concerns about their potential to generate undesirable or harmful content, ranging from fabricated depictions of public figures to sexually explicit images. To mitigate these risks, prior work has devised machine unlearning techniques that attempt to erase unwanted concepts through fine-tuning. However, in this paper, we introduce a new threat model, Toxic Erasure (ToxE), and demonstrate how recent unlearning algorithms, including those explicitly designed for robustness, can be circumvented through targeted backdoor attacks. The threat is realized by establishing a link between a trigger and the undesired content. Subsequent unlearning attempts fail to erase this link, allowing adversaries to produce harmful content. We instantiate ToxE via two established backdoor attacks: one targeting the text encoder and another manipulating the cross-attention layers. Further, we introduce Deep Intervention Score-based Attack (DISA), a novel, deeper backdoor attack that optimizes the entire U-Net using a score-based objective, improving the attack's persistence across different erasure methods. We evaluate five recent concept erasure methods against our threat model. For celebrity identity erasure, our deep attack circumvents erasure with up to 82% success, averaging 57% across all erasure methods. For explicit content erasure, ToxE attacks can elicit up to 9 times more exposed body parts, with DISA yielding an average increase by a factor of 2.9. These results highlight a critical security gap in current unlearning strategies.

摘要: 大规模文本到图像传播模式的扩展引发了人们越来越多的担忧，人们对它们可能产生不受欢迎或有害内容的可能性，从对公众人物的捏造描述到露骨的色情图像。为了减轻这些风险，之前的工作设计了机器取消学习技术，试图通过微调删除不需要的概念。然而，在本文中，我们引入了一种新的威胁模型有毒擦除（ToxE），并演示了如何通过有针对性的后门攻击来规避最近的取消学习算法，包括那些明确为鲁棒性设计的算法。威胁是通过在触发器和不需要的内容之间建立链接来实现的。随后的取消学习尝试未能删除此链接，从而导致对手产生有害内容。我们通过两种已建立的后门攻击实例化ToxE：一种针对文本编码器，另一种针对交叉注意力层。此外，我们还引入了深度干预基于分数的攻击（DISA），这是一种新颖的、更深层次的后门攻击，可以使用基于分数的目标优化整个U-Net，提高攻击在不同擦除方法中的持续性。我们针对我们的威胁模型评估了最近的五种概念擦除方法。对于名人身份擦除，我们的深度攻击以高达82%的成功率规避了擦除，所有擦除方法的平均成功率为57%。对于明确的内容擦除，ToxE攻击可以引起高达9倍的身体部位暴露，DISA平均增加2.9倍。这些结果凸显了当前遗忘策略中的一个关键安全漏洞。



## **30. Mitigating the Structural Bias in Graph Adversarial Defenses**

缓解图对抗防御中的结构偏差 cs.LG

Under Review

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20848v1) [paper-pdf](http://arxiv.org/pdf/2504.20848v1)

**Authors**: Junyuan Fang, Huimin Liu, Han Yang, Jiajing Wu, Zibin Zheng, Chi K. Tse

**Abstract**: In recent years, graph neural networks (GNNs) have shown great potential in addressing various graph structure-related downstream tasks. However, recent studies have found that current GNNs are susceptible to malicious adversarial attacks. Given the inevitable presence of adversarial attacks in the real world, a variety of defense methods have been proposed to counter these attacks and enhance the robustness of GNNs. Despite the commendable performance of these defense methods, we have observed that they tend to exhibit a structural bias in terms of their defense capability on nodes with low degree (i.e., tail nodes), which is similar to the structural bias of traditional GNNs on nodes with low degree in the clean graph. Therefore, in this work, we propose a defense strategy by including hetero-homo augmented graph construction, $k$NN augmented graph construction, and multi-view node-wise attention modules to mitigate the structural bias of GNNs against adversarial attacks. Notably, the hetero-homo augmented graph consists of removing heterophilic links (i.e., links connecting nodes with dissimilar features) globally and adding homophilic links (i.e., links connecting nodes with similar features) for nodes with low degree. To further enhance the defense capability, an attention mechanism is adopted to adaptively combine the representations from the above two kinds of graph views. We conduct extensive experiments to demonstrate the defense and debiasing effect of the proposed strategy on benchmark datasets.

摘要: 近年来，图神经网络（GNN）在解决各种与图结构相关的下游任务方面表现出了巨大的潜力。然而，最近的研究发现，当前的GNN容易受到恶意对抗攻击。鉴于现实世界中不可避免地存在对抗攻击，人们提出了各种防御方法来对抗这些攻击并增强GNN的鲁棒性。尽管这些防御方法的性能值得赞扬，但我们观察到它们在低程度节点上的防御能力方面往往表现出结构性偏差（即，尾节点），这类似于传统GNN在干净图中度较低的节点上的结构偏差。因此，在这项工作中，我们提出了一种防御策略，包括异同增强图构造、$k$NN增强图构造和多视图节点注意力模块，以减轻GNN对对抗性攻击的结构偏见。值得注意的是，异同增强图包括去除异亲性链接（即，连接具有不同特征的节点的链接）全球并添加同同性链接（即，连接具有相似特征的节点的链接）对于程度较低的节点。为了进一步增强防御能力，采用关注机制自适应地组合上述两种图视图的表示。我们进行了广泛的实验来证明所提出的策略对基准数据集的防御和去偏置效果。



## **31. GaussTrap: Stealthy Poisoning Attacks on 3D Gaussian Splatting for Targeted Scene Confusion**

GaussTrap：对3D高斯飞溅进行隐形中毒攻击以造成目标场景混乱 cs.CV

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20829v1) [paper-pdf](http://arxiv.org/pdf/2504.20829v1)

**Authors**: Jiaxin Hong, Sixu Chen, Shuoyang Sun, Hongyao Yu, Hao Fang, Yuqi Tan, Bin Chen, Shuhan Qi, Jiawei Li

**Abstract**: As 3D Gaussian Splatting (3DGS) emerges as a breakthrough in scene representation and novel view synthesis, its rapid adoption in safety-critical domains (e.g., autonomous systems, AR/VR) urgently demands scrutiny of potential security vulnerabilities. This paper presents the first systematic study of backdoor threats in 3DGS pipelines. We identify that adversaries may implant backdoor views to induce malicious scene confusion during inference, potentially leading to environmental misperception in autonomous navigation or spatial distortion in immersive environments. To uncover this risk, we propose GuassTrap, a novel poisoning attack method targeting 3DGS models. GuassTrap injects malicious views at specific attack viewpoints while preserving high-quality rendering in non-target views, ensuring minimal detectability and maximizing potential harm. Specifically, the proposed method consists of a three-stage pipeline (attack, stabilization, and normal training) to implant stealthy, viewpoint-consistent poisoned renderings in 3DGS, jointly optimizing attack efficacy and perceptual realism to expose security risks in 3D rendering. Extensive experiments on both synthetic and real-world datasets demonstrate that GuassTrap can effectively embed imperceptible yet harmful backdoor views while maintaining high-quality rendering in normal views, validating its robustness, adaptability, and practical applicability.

摘要: 随着3D高斯飞溅（3DGS）成为场景表示和新颖视图合成领域的突破，它在安全关键领域（例如，自治系统（AR/VR）迫切需要对潜在的安全漏洞进行审查。本文首次对3DGS管道中的后门威胁进行了系统研究。我们发现对手可能会植入后门视图，以在推理过程中引发恶意场景混乱，这可能会导致自主导航中的环境误解或沉浸式环境中的空间失真。为了发现这一风险，我们提出了GuassTrap，这是一种针对3DGS模型的新型中毒攻击方法。GuassTrap在特定的攻击观点处注入恶意视图，同时在非目标视图中保留高质量渲染，确保最小的可检测性并最大化潜在危害。具体来说，所提出的方法由三阶段管道（攻击、稳定和正常训练）组成，在3DGS中植入隐形、观点一致的有毒渲染，共同优化攻击功效和感知真实感，以暴露3D渲染中的安全风险。对合成和现实世界数据集的广泛实验表明，GuassTrap可以有效地嵌入难以感知但有害的后门视图，同时在正常视图中保持高质量的渲染，验证了其稳健性、适应性和实际适用性。



## **32. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

正规的鲁棒可靠的学习者和实例有针对性的攻击 cs.LG

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2410.10572v3) [paper-pdf](http://arxiv.org/pdf/2410.10572v3)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.

摘要: 针对实例的数据中毒攻击（对手破坏训练集以在特定测试点上引发错误）引发了严重担忧。Balcan等人（2022）提出了一种解决这一挑战的方法，通过定义鲁棒可靠学习器的概念，即使存在数据中毒攻击，也可以在明确定义的假设下提供每个实例的正确性保证。然后，他们给出了一个通用的最佳（但计算效率低下）鲁棒可靠的学习器，以及一个计算高效的算法，用于线性分离器在线性分离器的情况。   在这项工作中，我们解决了Balcan等人（2022）留下的两个挑战。首先，Balcan et al（2022）中对鲁棒可靠学习者的定义对于高度灵活的假设类别来说变得空洞：如果H中有两个分类器h_0、h_1 \，两者在训练集上的误差为零，使得h_0（x）\neq h_1（x），那么鲁棒可靠学习者必须放弃x。我们通过定义一个修改的正规化鲁棒可靠学习器概念来解决这个问题，该概念允许在这种情况下的非平凡陈述。其次，Balcan等人（2022）的通用算法需要在每个测试点x上重新运行ERM Oracle（本质上是重新训练分类器），即使可以有效地实施ERM，这通常也是不切实际的。为了解决这个问题，我们表明，至少在某些有趣的情况下，我们可以通过使用动态算法设计的技术来设计可以在训练时间内产生次线性输出的算法。



## **33. A Survey on Adversarial Contention Resolution**

对抗性竞争解决方法研究 cs.DC

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2403.03876v4) [paper-pdf](http://arxiv.org/pdf/2403.03876v4)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We also highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.

摘要: 竞争解决解决了协调多个进程对共享资源（例如内存、磁盘存储或通信通道）的访问的挑战。尽管经历了几十年的技术变革，竞争解决最初是受数据库系统和公交网络挑战的推动，但它仍然作为资源共享的重要抽象而经久不衰。在这里，我们调查了有关解决最坏情况争用的文献，其中进程的数量以及每个进程可能开始寻求访问资源的时间由对手决定。我们还强调了竞争解决方案的演变，现代系统引发了新的担忧（例如安全性、服务质量和能源效率）。这些努力深入了解了随机和确定性方法的局限性，以及不同模型假设（例如全球时钟同步、处理器数量的知识、访问尝试的反馈以及对共享资源可用性的攻击）的影响。



## **34. Data Encryption Battlefield: A Deep Dive into the Dynamic Confrontations in Ransomware Attacks**

数据加密战场：深入探讨勒索软件攻击中的动态对抗 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20681v1) [paper-pdf](http://arxiv.org/pdf/2504.20681v1)

**Authors**: Arash Mahboubi, Hamed Aboutorab, Seyit Camtepe, Hang Thanh Bui, Khanh Luong, Keyvan Ansari, Shenlu Wang, Bazara Barry

**Abstract**: In the rapidly evolving landscape of cybersecurity threats, ransomware represents a significant challenge. Attackers increasingly employ sophisticated encryption methods, such as entropy reduction through Base64 encoding, and partial or intermittent encryption to evade traditional detection methods. This study explores the dynamic battle between adversaries who continuously refine encryption strategies and defenders developing advanced countermeasures to protect vulnerable data. We investigate the application of online incremental machine learning algorithms designed to predict file encryption activities despite adversaries evolving obfuscation techniques. Our analysis utilizes an extensive dataset of 32.6 GB, comprising 11,928 files across multiple formats, including Microsoft Word documents (doc), PowerPoint presentations (ppt), Excel spreadsheets (xlsx), image formats (jpg, jpeg, png, tif, gif), PDFs (pdf), audio (mp3), and video (mp4) files. These files were encrypted by 75 distinct ransomware families, facilitating a robust empirical evaluation of machine learning classifiers effectiveness against diverse encryption tactics. Results highlight the Hoeffding Tree algorithms superior incremental learning capability, particularly effective in detecting traditional and AES-Base64 encryption methods employed to lower entropy. Conversely, the Random Forest classifier with warm-start functionality excels at identifying intermittent encryption methods, demonstrating the necessity of tailored machine learning solutions to counter sophisticated ransomware strategies.

摘要: 在迅速变化的网络安全威胁格局中，勒索软件构成了一个重大挑战。攻击者越来越多地使用复杂的加密方法，例如通过Base 64编码进行的减序，以及部分或间歇性加密来逃避传统的检测方法。本研究探讨了不断完善加密策略的对手与开发高级对策来保护脆弱数据的防御者之间的动态战斗。我们研究了在线增量机器学习算法的应用，该算法旨在预测文件加密活动，尽管对手不断发展混淆技术。我们的分析利用了32.6 GB的广泛数据集，包括多种格式的11，928个文件，包括Microsoft Word文档（Doc）、PowerPoint演示文稿（GPT）、Excel电子表格（xlsx）、图像格式（jpg、jpeg、png、tif、gif）、PDF（pdf）、音频（mp3）和视频（mp4）文件。这些文件由75个不同的勒索软件家族加密，有助于对机器学习分类器针对不同加密策略的有效性进行稳健的实证评估。结果凸显了Hoeffding Tree算法优越的增量学习能力，在检测用于降低信息量的传统加密方法和AES-Base 64加密方法方面特别有效。相反，具有热启动功能的随机森林分类器擅长识别间歇性加密方法，证明了定制机器学习解决方案来对抗复杂勒索软件策略的必要性。



## **35. Learning and Generalization with Mixture Data**

使用混合数据学习和概括 stat.ML

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20651v1) [paper-pdf](http://arxiv.org/pdf/2504.20651v1)

**Authors**: Harsh Vardhan, Avishek Ghosh, Arya Mazumdar

**Abstract**: In many, if not most, machine learning applications the training data is naturally heterogeneous (e.g. federated learning, adversarial attacks and domain adaptation in neural net training). Data heterogeneity is identified as one of the major challenges in modern day large-scale learning. A classical way to represent heterogeneous data is via a mixture model. In this paper, we study generalization performance and statistical rates when data is sampled from a mixture distribution. We first characterize the heterogeneity of the mixture in terms of the pairwise total variation distance of the sub-population distributions. Thereafter, as a central theme of this paper, we characterize the range where the mixture may be treated as a single (homogeneous) distribution for learning. In particular, we study the generalization performance under the classical PAC framework and the statistical error rates for parametric (linear regression, mixture of hyperplanes) as well as non-parametric (Lipschitz, convex and H\"older-smooth) regression problems. In order to do this, we obtain Rademacher complexity and (local) Gaussian complexity bounds with mixture data, and apply them to get the generalization and convergence rates respectively. We observe that as the (regression) function classes get more complex, the requirement on the pairwise total variation distance gets stringent, which matches our intuition. We also do a finer analysis for the case of mixed linear regression and provide a tight bound on the generalization error in terms of heterogeneity.

摘要: 在许多（如果不是大多数）机器学习应用中，训练数据自然是异类的（例如联邦学习、对抗性攻击和神经网络训练中的领域适应）。数据异类被认为是现代大规模学习的主要挑战之一。表示异类数据的经典方法是通过混合模型。本文研究了从混合分布中采样数据时的概括性能和统计率。我们首先根据亚群体分布的成对总变异距离来描述混合物的均匀性。此后，作为本文的中心主题，我们描述了混合物可以被视为单一（均匀）学习分布的范围。特别是，我们研究了经典PAC框架下的推广性能以及参数（线性回归、超平面混合）以及非参数（Lipschitz、凸和H ' old smooth）回归问题的统计错误率。为了做到这一点，我们通过混合数据获得Rademacher复杂度和（局部）高斯复杂度界限，并分别应用它们来获得概括率和收敛率。我们观察到，随着（回归）函数类变得越来越复杂，对成对总变异距离的要求变得严格，这符合我们的直觉。我们还对混合线性回归的情况进行了更细致的分析，并在方差方面为概括误差提供了严格的界限。



## **36. WILD: a new in-the-Wild Image Linkage Dataset for synthetic image attribution**

WILD：一个新的用于合成图像属性的野外图像链接数据集 cs.MM

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.19595v2) [paper-pdf](http://arxiv.org/pdf/2504.19595v2)

**Authors**: Pietro Bongini, Sara Mandelli, Andrea Montibeller, Mirko Casu, Orazio Pontorno, Claudio Vittorio Ragaglia, Luca Zanchetta, Mattia Aquilina, Taiba Majid Wani, Luca Guarnera, Benedetta Tondi, Giulia Boato, Paolo Bestagini, Irene Amerini, Francesco De Natale, Sebastiano Battiato, Mauro Barni

**Abstract**: Synthetic image source attribution is an open challenge, with an increasing number of image generators being released yearly. The complexity and the sheer number of available generative techniques, as well as the scarcity of high-quality open source datasets of diverse nature for this task, make training and benchmarking synthetic image source attribution models very challenging. WILD is a new in-the-Wild Image Linkage Dataset designed to provide a powerful training and benchmarking tool for synthetic image attribution models. The dataset is built out of a closed set of 10 popular commercial generators, which constitutes the training base of attribution models, and an open set of 10 additional generators, simulating a real-world in-the-wild scenario. Each generator is represented by 1,000 images, for a total of 10,000 images in the closed set and 10,000 images in the open set. Half of the images are post-processed with a wide range of operators. WILD allows benchmarking attribution models in a wide range of tasks, including closed and open set identification and verification, and robust attribution with respect to post-processing and adversarial attacks. Models trained on WILD are expected to benefit from the challenging scenario represented by the dataset itself. Moreover, an assessment of seven baseline methodologies on closed and open set attribution is presented, including robustness tests with respect to post-processing.

摘要: 合成图像源归属是一个公开的挑战，每年发布的图像生成器数量越来越多。可用生成技术的复杂性和数量之多，以及用于该任务的多样化高质量开源数据集的稀缺性，使得训练和基准合成图像源归因模型变得非常具有挑战性。WILD是一个新的野外图像联动数据集，旨在为合成图像归因模型提供强大的训练和基准测试工具。该数据集由一组由10个流行的商业生成器组成的封闭集和一组由10个额外生成器组成的开放集构建，该生成器构成了归因模型的训练基础，模拟了现实世界的野外场景。每个生成器由1，000个图像表示，闭集中总共有10，000个图像，开集中有10，000个图像。一半的图像经过各种操作员的后处理。WILD允许在广泛的任务中对归因模型进行基准测试，包括封闭和开放集识别和验证，以及针对后处理和对抗攻击的稳健归因。在WILD上训练的模型预计将受益于数据集本身所代表的具有挑战性的场景。此外，还对封闭集和开放集归因的七种基线方法进行了评估，包括后处理方面的稳健性测试。



## **37. NeuRel-Attack: Neuron Relearning for Safety Disalignment in Large Language Models**

NeuRel-Attack：大型语言模型中安全失准的神经元再学习 cs.LG

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.21053v1) [paper-pdf](http://arxiv.org/pdf/2504.21053v1)

**Authors**: Yi Zhou, Wenpeng Xing, Dezhang Kong, Changting Lin, Meng Han

**Abstract**: Safety alignment in large language models (LLMs) is achieved through fine-tuning mechanisms that regulate neuron activations to suppress harmful content. In this work, we propose a novel approach to induce disalignment by identifying and modifying the neurons responsible for safety constraints. Our method consists of three key steps: Neuron Activation Analysis, where we examine activation patterns in response to harmful and harmless prompts to detect neurons that are critical for distinguishing between harmful and harmless inputs; Similarity-Based Neuron Identification, which systematically locates the neurons responsible for safe alignment; and Neuron Relearning for Safety Removal, where we fine-tune these selected neurons to restore the model's ability to generate previously restricted responses. Experimental results demonstrate that our method effectively removes safety constraints with minimal fine-tuning, highlighting a critical vulnerability in current alignment techniques. Our findings underscore the need for robust defenses against adversarial fine-tuning attacks on LLMs.

摘要: 大型语言模型（LLM）中的安全对齐是通过调节神经元激活以抑制有害内容的微调机制来实现的。在这项工作中，我们提出了一种新的方法，通过识别和修改负责安全约束的神经元来诱导失调。我们的方法包括三个关键步骤：神经元激活分析，在那里我们检查响应有害和无害提示的激活模式，以检测对区分有害和无害输入至关重要的神经元;基于相似性的神经元识别，系统地定位负责安全对齐的神经元;和Neuron Relearning for Safety Removal，我们对这些选定的神经元进行微调，以恢复模型生成先前受限响应的能力。实验结果表明，我们的方法可以通过最少的微调有效地消除安全约束，凸显了当前对齐技术中的一个关键漏洞。我们的研究结果强调了对LLM的对抗性微调攻击的强大防御的必要性。



## **38. Enhancing Leakage Attacks on Searchable Symmetric Encryption Using LLM-Based Synthetic Data Generation**

使用基于LLM的合成数据生成增强对可搜索对称加密的泄漏攻击 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20414v1) [paper-pdf](http://arxiv.org/pdf/2504.20414v1)

**Authors**: Joshua Chiu, Partha Protim Paul, Zahin Wahab

**Abstract**: Searchable Symmetric Encryption (SSE) enables efficient search capabilities over encrypted data, allowing users to maintain privacy while utilizing cloud storage. However, SSE schemes are vulnerable to leakage attacks that exploit access patterns, search frequency, and volume information. Existing studies frequently assume that adversaries possess a substantial fraction of the encrypted dataset to mount effective inference attacks, implying there is a database leakage of such documents, thus, an assumption that may not hold in real-world scenarios. In this work, we investigate the feasibility of enhancing leakage attacks under a more realistic threat model in which adversaries have access to minimal leaked data. We propose a novel approach that leverages large language models (LLMs), specifically GPT-4 variants, to generate synthetic documents that statistically and semantically resemble the real-world dataset of Enron emails. Using the email corpus as a case study, we evaluate the effectiveness of synthetic data generated via random sampling and hierarchical clustering methods on the performance of the SAP (Search Access Pattern) keyword inference attack restricted to token volumes only. Our results demonstrate that, while the choice of LLM has limited effect, increasing dataset size and employing clustering-based generation significantly improve attack accuracy, achieving comparable performance to attacks using larger amounts of real data. We highlight the growing relevance of LLMs in adversarial contexts.

摘要: 可搜索对称加密（SSE）支持对加密数据进行高效搜索，使用户能够在利用云存储的同时维护隐私。然而，SSE方案很容易受到利用访问模式、搜索频率和量信息的泄露攻击。现有的研究经常假设对手拥有很大一部分加密数据集来发起有效的推理攻击，这意味着此类文档的数据库泄露，因此，这一假设在现实世界的场景中可能不成立。在这项工作中，我们研究了在更现实的威胁模型下增强泄露攻击的可行性，其中对手可以访问最少的泄露数据。我们提出了一种新颖的方法，利用大型语言模型（LLM），特别是GPT-4变体，来生成在统计和语义上与安然电子邮件的现实世界数据集相似的合成文档。使用电子邮件库作为案例研究，我们评估了通过随机抽样和分层集群方法生成的合成数据对仅限于令牌量的SAP（搜索访问模式）关键字推理攻击性能的有效性。我们的结果表明，虽然选择LLM的效果有限，但增加数据集大小和采用基于集群的生成可以显着提高攻击准确性，实现与使用大量真实数据的攻击相当的性能。我们强调法学硕士在对抗背景下日益增长的相关性。



## **39. Inception: Jailbreak the Memory Mechanism of Text-to-Image Generation Systems**

盗梦空间：越狱文本到图像生成系统的记忆机制 cs.CV

17 pages, 8 figures

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20376v1) [paper-pdf](http://arxiv.org/pdf/2504.20376v1)

**Authors**: Shiqian Zhao, Jiayang Liu, Yiming Li, Runyi Hu, Xiaojun Jia, Wenshu Fan, Xinfeng Li, Jie Zhang, Wei Dong, Tianwei Zhang, Luu Anh Tuan

**Abstract**: Currently, the memory mechanism has been widely and successfully exploited in online text-to-image (T2I) generation systems ($e.g.$, DALL$\cdot$E 3) for alleviating the growing tokenization burden and capturing key information in multi-turn interactions. Despite its practicality, its security analyses have fallen far behind. In this paper, we reveal that this mechanism exacerbates the risk of jailbreak attacks. Different from previous attacks that fuse the unsafe target prompt into one ultimate adversarial prompt, which can be easily detected or may generate non-unsafe images due to under- or over-optimization, we propose Inception, the first multi-turn jailbreak attack against the memory mechanism in real-world text-to-image generation systems. Inception embeds the malice at the inception of the chat session turn by turn, leveraging the mechanism that T2I generation systems retrieve key information in their memory. Specifically, Inception mainly consists of two modules. It first segments the unsafe prompt into chunks, which are subsequently fed to the system in multiple turns, serving as pseudo-gradients for directive optimization. Specifically, we develop a series of segmentation policies that ensure the images generated are semantically consistent with the target prompt. Secondly, after segmentation, to overcome the challenge of the inseparability of minimum unsafe words, we propose recursion, a strategy that makes minimum unsafe words subdivisible. Collectively, segmentation and recursion ensure that all the request prompts are benign but can lead to malicious outcomes. We conduct experiments on the real-world text-to-image generation system ($i.e.$, DALL$\cdot$E 3) to validate the effectiveness of Inception. The results indicate that Inception surpasses the state-of-the-art by a 14\% margin in attack success rate.

摘要: 目前，存储机制已在在线文本到图像（T2 I）生成系统中得到广泛且成功的利用（$e.g.$，DALL$\csot $E 3）用于减轻日益增长的代币化负担并捕获多回合交互中的关键信息。尽管它实用，但它的安全分析却远远落后。在本文中，我们揭示了这种机制加剧了越狱攻击的风险。与之前将不安全的目标提示融合为一个终极对抗提示的攻击不同，这种攻击可以很容易地检测到，或者可能会由于优化不足或过度而生成非不安全的图像，我们提出了Incept，这是第一个针对现实世界中的存储机制的多回合越狱攻击。文本到图像生成系统。Incement利用T2 I生成系统在其内存中检索关键信息的机制，在聊天会话开始时轮流嵌入恶意。具体来说，Incion主要由两个模块组成。它首先将不安全的提示分割成块，随后将这些块分多次输送到系统，作为指令优化的伪梯度。具体来说，我们开发了一系列分割策略，以确保生成的图像在语义上与目标提示一致。其次，在分段之后，为了克服最小不安全词不可分割的挑战，我们提出了回归，这是一种使最小不安全词可细分的策略。总的来说，分段和回归确保所有请求提示都是良性的，但可能会导致恶意结果。我们对现实世界的文本到图像生成系统（$i.e.$，DALL$\csot $E 3）验证Incession的有效性。结果表明，Incion的攻击成功率比最新技术水平高出14%。



## **40. A Cryptographic Perspective on Mitigation vs. Detection in Machine Learning**

机器学习中缓解与检测的密码学视角 cs.LG

29 pages

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20310v1) [paper-pdf](http://arxiv.org/pdf/2504.20310v1)

**Authors**: Greg Gluch, Shafi Goldwasser

**Abstract**: In this paper, we initiate a cryptographically inspired theoretical study of detection versus mitigation of adversarial inputs produced by attackers of Machine Learning algorithms during inference time.   We formally define defense by detection (DbD) and defense by mitigation (DbM). Our definitions come in the form of a 3-round protocol between two resource-bounded parties: a trainer/defender and an attacker. The attacker aims to produce inference-time inputs that fool the training algorithm. We define correctness, completeness, and soundness properties to capture successful defense at inference time while not degrading (too much) the performance of the algorithm on inputs from the training distribution.   We first show that achieving DbD and achieving DbM are equivalent for ML classification tasks. Surprisingly, this is not the case for ML generative learning tasks, where there are many possible correct outputs that can be generated for each input. We show a separation between DbD and DbM by exhibiting a generative learning task for which is possible to defend by mitigation but is provably impossible to defend by detection under the assumption that the Identity-Based Fully Homomorphic Encryption (IB-FHE), publicly-verifiable zero-knowledge Succinct Non-Interactive Arguments of Knowledge (zk-SNARK) and Strongly Unforgeable Signatures exist. The mitigation phase uses significantly fewer samples than the initial training algorithm.

摘要: 在本文中，我们启动了一项受密码启发的理论研究，研究机器学习算法攻击者在推理时间内产生的对抗输入的检测与缓解。   我们正式定义了检测防御（GbD）和缓解防御（GbM）。我们的定义以两个资源有限方之间的三轮协议的形式出现：训练者/防御者和攻击者。攻击者的目标是产生欺骗训练算法的推断时输入。我们定义了正确性、完整性和可靠性属性，以在推理时捕获成功的防御，同时不会降低（太多）算法在训练分布输入上的性能。   我们首先表明，实现GbD和实现GbM对于ML分类任务来说是等效的。令人惊讶的是，ML生成式学习任务的情况并非如此，其中可以为每个输入生成许多可能的正确输出。我们通过展示生成式学习任务来展示GbD和GbM之间的分离，该任务可以通过缓解来防御，但在假设基于身份的完全同质加密（IB-FHE）、可公开验证的零知识连续非交互式知识参数（zk-SNARK）和强不可伪造签名存在的假设下，该任务可以通过缓解来防御。缓解阶段使用的样本比初始训练算法少得多。



## **41. The Dark Side of Digital Twins: Adversarial Attacks on AI-Driven Water Forecasting**

数字双胞胎的阴暗面：对人工智能驱动的水预测的对抗攻击 cs.LG

7 Pages, 7 Figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20295v1) [paper-pdf](http://arxiv.org/pdf/2504.20295v1)

**Authors**: Mohammadhossein Homaei, Victor Gonzalez Morales, Oscar Mogollon-Gutierrez, Andres Caro

**Abstract**: Digital twins (DTs) are improving water distribution systems by using real-time data, analytics, and prediction models to optimize operations. This paper presents a DT platform designed for a Spanish water supply network, utilizing Long Short-Term Memory (LSTM) networks to predict water consumption. However, machine learning models are vulnerable to adversarial attacks, such as the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). These attacks manipulate critical model parameters, injecting subtle distortions that degrade forecasting accuracy. To further exploit these vulnerabilities, we introduce a Learning Automata (LA) and Random LA-based approach that dynamically adjusts perturbations, making adversarial attacks more difficult to detect. Experimental results show that this approach significantly impacts prediction reliability, causing the Mean Absolute Percentage Error (MAPE) to rise from 26% to over 35%. Moreover, adaptive attack strategies amplify this effect, highlighting cybersecurity risks in AI-driven DTs. These findings emphasize the urgent need for robust defenses, including adversarial training, anomaly detection, and secure data pipelines.

摘要: 数字双胞胎（DT）正在通过使用实时数据、分析和预测模型来优化运营来改善供水系统。本文介绍了一个为西班牙供水网络设计的DT平台，利用长短期记忆（LSTM）网络来预测用水量。然而，机器学习模型很容易受到对抗攻击，例如快速梯度符号法（FGSM）和投影梯度下降（PVD）。这些攻击操纵关键模型参数，注入微妙的扭曲，降低预测准确性。为了进一步利用这些漏洞，我们引入了学习自动机（LA）和基于随机LA的方法，该方法动态调整扰动，使对抗性攻击更难以检测。实验结果表明，这种方法显着影响预测可靠性，导致平均绝对百分比误差（MAPE）从26%上升到35%以上。此外，自适应攻击策略放大了这种影响，凸显了人工智能驱动的DT中的网络安全风险。这些发现强调了对强大防御的迫切需要，包括对抗训练、异常检测和安全数据管道。



## **42. A Case Study on the Use of Representativeness Bias as a Defense Against Adversarial Cyber Threats**

使用代表性偏见作为对抗性网络威胁防御的案例研究 cs.CR

To appear in the 4th Workshop on Active Defense and Deception (ADnD),  co-located with the 10th IEEE European Symposium on Security and Privacy  (EuroS&P 2025)

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20245v1) [paper-pdf](http://arxiv.org/pdf/2504.20245v1)

**Authors**: Briland Hitaj, Grit Denker, Laura Tinnel, Michael McAnally, Bruce DeBruhl, Nathan Bunting, Alex Fafard, Daniel Aaron, Richard D. Roberts, Joshua Lawson, Greg McCain, Dylan Starink

**Abstract**: Cyberspace is an ever-evolving battleground involving adversaries seeking to circumvent existing safeguards and defenders aiming to stay one step ahead by predicting and mitigating the next threat. Existing mitigation strategies have focused primarily on solutions that consider software or hardware aspects, often ignoring the human factor. This paper takes a first step towards psychology-informed, active defense strategies, where we target biases that human beings are susceptible to under conditions of uncertainty.   Using capture-the-flag events, we create realistic challenges that tap into a particular cognitive bias: representativeness. This study finds that this bias can be triggered to thwart hacking attempts and divert hackers into non-vulnerable attack paths. Participants were exposed to two different challenges designed to exploit representativeness biases. One of the representativeness challenges significantly thwarted attackers away from vulnerable attack vectors and onto non-vulnerable paths, signifying an effective bias-based defense mechanism. This work paves the way towards cyber defense strategies that leverage additional human biases to thwart future, sophisticated adversarial attacks.

摘要: 网络空间是一个不断发展的战场，对手试图绕过现有的保障措施，而防御者则试图通过预测和减轻下一个威胁来领先一步。现有的缓解策略主要集中在考虑软件或硬件方面的解决方案上，通常忽视人为因素。本文朝着基于心理的主动防御策略迈出了第一步，我们针对人类在不确定条件下容易受到的偏见。   使用夺旗事件，我们创造了现实的挑战，这些挑战利用了特定的认知偏见：代表性。这项研究发现，这种偏见可以被触发来阻止黑客企图并将黑客转移到非脆弱的攻击路径上。参与者面临两种不同的挑战，旨在利用代表性偏见。代表性挑战之一显着阻止攻击者远离脆弱的攻击载体并转向非脆弱的路径，这意味着有效的基于偏差的防御机制。这项工作为网络防御策略铺平了道路，这些策略利用额外的人类偏见来阻止未来复杂的对抗性攻击。



## **43. DROP: Poison Dilution via Knowledge Distillation for Federated Learning**

Drop：通过联邦学习的知识蒸馏进行毒药稀释 cs.LG

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2502.07011v2) [paper-pdf](http://arxiv.org/pdf/2502.07011v2)

**Authors**: Georgios Syros, Anshuman Suri, Farinaz Koushanfar, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Federated Learning is vulnerable to adversarial manipulation, where malicious clients can inject poisoned updates to influence the global model's behavior. While existing defense mechanisms have made notable progress, they fail to protect against adversaries that aim to induce targeted backdoors under different learning and attack configurations. To address this limitation, we introduce DROP (Distillation-based Reduction Of Poisoning), a novel defense mechanism that combines clustering and activity-tracking techniques with extraction of benign behavior from clients via knowledge distillation to tackle stealthy adversaries that manipulate low data poisoning rates and diverse malicious client ratios within the federation. Through extensive experimentation, our approach demonstrates superior robustness compared to existing defenses across a wide range of learning configurations. Finally, we evaluate existing defenses and our method under the challenging setting of non-IID client data distribution and highlight the challenges of designing a resilient FL defense in this setting.

摘要: 联邦学习很容易受到对抗操纵的影响，恶意客户端可以注入有毒更新来影响全局模型的行为。虽然现有的防御机制取得了显着进展，但它们未能防止对手在不同的学习和攻击配置下诱导有针对性的后门。为了解决这一局限性，我们引入了Dopp（基于蒸馏的中毒减少），这是一种新型防御机制，它将集群和活动跟踪技术与通过知识蒸馏从客户端提取良性行为相结合，以应对操纵低数据中毒率和联邦内各种恶意客户比例的隐形对手。通过广泛的实验，与广泛的学习配置中的现有防御相比，我们的方法表现出了卓越的鲁棒性。最后，我们在非IID客户端数据分发的具有挑战性的环境下评估了现有的防御措施和我们的方法，并强调了在这种环境下设计弹性FL防御的挑战。



## **44. AGATE: Stealthy Black-box Watermarking for Multimodal Model Copyright Protection**

AGATE：用于多模式模型版权保护的隐形黑匣子水印 cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.21044v1) [paper-pdf](http://arxiv.org/pdf/2504.21044v1)

**Authors**: Jianbo Gao, Keke Gai, Jing Yu, Liehuang Zhu, Qi Wu

**Abstract**: Recent advancement in large-scale Artificial Intelligence (AI) models offering multimodal services have become foundational in AI systems, making them prime targets for model theft. Existing methods select Out-of-Distribution (OoD) data as backdoor watermarks and retrain the original model for copyright protection. However, existing methods are susceptible to malicious detection and forgery by adversaries, resulting in watermark evasion. In this work, we propose Model-\underline{ag}nostic Black-box Backdoor W\underline{ate}rmarking Framework (AGATE) to address stealthiness and robustness challenges in multimodal model copyright protection. Specifically, we propose an adversarial trigger generation method to generate stealthy adversarial triggers from ordinary dataset, providing visual fidelity while inducing semantic shifts. To alleviate the issue of anomaly detection among model outputs, we propose a post-transform module to correct the model output by narrowing the distance between adversarial trigger image embedding and text embedding. Subsequently, a two-phase watermark verification is proposed to judge whether the current model infringes by comparing the two results with and without the transform module. Consequently, we consistently outperform state-of-the-art methods across five datasets in the downstream tasks of multimodal image-text retrieval and image classification. Additionally, we validated the robustness of AGATE under two adversarial attack scenarios.

摘要: 提供多模式服务的大规模人工智能（AI）模型的最新进展已成为人工智能系统的基础，使它们成为模型盗窃的主要目标。现有方法选择分发外（OoD）数据作为后门水印，并重新训练原始模型以进行版权保护。然而，现有的方法很容易受到对手的恶意检测和伪造，从而导致水印规避。在这项工作中，我们提出了Model-\underline{ag}nostic Black-box Backdoor W\underline{ate}rmarking框架（AGATE）来解决多模式模型版权保护中的隐蔽性和鲁棒性挑战。具体来说，我们提出了一种对抗触发生成方法，从普通数据集生成隐形对抗触发，在诱导语义转变的同时提供视觉保真度。为了缓解模型输出之间的异常检测问题，我们提出了一个后变换模块，通过缩小对抗触发图像嵌入和文本嵌入之间的距离来纠正模型输出。随后，提出了两阶段水印验证，通过比较有和没有变换模块的两个结果来判断当前模型是否侵权。因此，在多模式图像文本检索和图像分类的下游任务中，我们在五个数据集上始终优于最先进的方法。此外，我们还验证了AGATE在两种对抗攻击场景下的稳健性。



## **45. Evaluate-and-Purify: Fortifying Code Language Models Against Adversarial Attacks Using LLM-as-a-Judge**

评估和净化：使用LLM作为法官来加强代码语言模型对抗性攻击 cs.SE

25 pages, 6 figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19730v1) [paper-pdf](http://arxiv.org/pdf/2504.19730v1)

**Authors**: Wenhan Mu, Ling Xu, Shuren Pei, Le Mi, Huichi Zhou

**Abstract**: The widespread adoption of code language models in software engineering tasks has exposed vulnerabilities to adversarial attacks, especially the identifier substitution attacks. Although existing identifier substitution attackers demonstrate high success rates, they often produce adversarial examples with unnatural code patterns. In this paper, we systematically assess the quality of adversarial examples using LLM-as-a-Judge. Our analysis reveals that over 80% of adversarial examples generated by state-of-the-art identifier substitution attackers (e.g., ALERT) are actually detectable. Based on this insight, we propose EP-Shield, a unified framework for evaluating and purifying identifier substitution attacks via naturalness-aware reasoning. Specifically, we first evaluate the naturalness of code and identify the perturbed adversarial code, then purify it so that the victim model can restore correct prediction. Extensive experiments demonstrate the superiority of EP-Shield over adversarial fine-tuning (up to 83.36% improvement) and its lightweight design 7B parameters) with GPT-4-level performance.

摘要: 代码语言模型在软件工程任务中的广泛采用暴露了对抗攻击的脆弱性，尤其是标识符替换攻击。尽管现有的标识符替换攻击者表现出很高的成功率，但他们经常产生具有不自然代码模式的对抗性示例。在本文中，我们使用LLM作为法官系统评估了对抗性示例的质量。我们的分析表明，超过80%的对抗示例是由最先进的标识符替换攻击者生成的（例如，警报）实际上是可检测的。基于这一见解，我们提出了EP-Shield，一个统一的框架，通过自然感知推理评估和净化标识符替换攻击。具体来说，我们首先评估代码的自然性并识别出受干扰的对抗代码，然后对其进行净化，以便受害者模型能够恢复正确的预测。大量的实验证明了EP-Shield在对抗性微调方面的优越性（高达83.36%的改进）及其轻量级设计（7 B参数），具有GPT-4级性能。



## **46. Fooling the Decoder: An Adversarial Attack on Quantum Error Correction**

欺骗解码器：对量子纠错的对抗性攻击 quant-ph

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19651v1) [paper-pdf](http://arxiv.org/pdf/2504.19651v1)

**Authors**: Jerome Lenssen, Alexandru Paler

**Abstract**: Neural network decoders are becoming essential for achieving fault-tolerant quantum computations. However, their internal mechanisms are poorly understood, hindering our ability to ensure their reliability and security against adversarial attacks. Leading machine learning decoders utilize recurrent and transformer models (e.g., AlphaQubit), with reinforcement learning (RL) playing a key role in training advanced transformer models (e.g., DeepSeek R1). In this work, we target a basic RL surface code decoder (DeepQ) to create the first adversarial attack on quantum error correction. By applying state-of-the-art white-box methods, we uncover vulnerabilities in this decoder, demonstrating an attack that reduces the logical qubit lifetime in memory experiments by up to five orders of magnitude. We validate that this attack exploits a genuine weakness, as the decoder exhibits robustness against noise fluctuations, is largely unaffected by substituting the referee decoder, responsible for episode termination, with an MWPM decoder, and demonstrates fault tolerance at checkable code distances. This attack highlights the susceptibility of machine learning-based QEC and underscores the importance of further research into robust QEC methods.

摘要: 神经网络解码器对于实现耐故障量子计算来说变得至关重要。然而，人们对它们的内部机制知之甚少，这阻碍了我们确保其可靠性和安全性抵御对抗攻击的能力。领先的机器学习解码器利用循环模型和Transformer模型（例如，AlphaQubit），强化学习（RL）在训练高级Transformer模型（例如，DeepSeek R1）。在这项工作中，我们以基本RL表面码解码器（DeepQ）为目标，以创建对量子错误纠正的第一次对抗攻击。通过应用最先进的白盒方法，我们发现了这个解码器中的漏洞，展示了一种将内存实验中逻辑量子位寿命缩短多达五个数量级的攻击。我们验证了这种攻击利用了一个真正的弱点，因为解码器表现出对噪音波动的鲁棒性，用MWPM解码器替换负责剧集终止的裁判解码器在很大程度上不受影响，并且在可检查的代码距离上表现出了故障容忍性。这次攻击凸显了基于机器学习的QEC的易感性，并强调了进一步研究稳健的QEC方法的重要性。



## **47. Prefill-Based Jailbreak: A Novel Approach of Bypassing LLM Safety Boundary**

基于预填充的越狱：一种突破LLM安全边界的新方法 cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.21038v1) [paper-pdf](http://arxiv.org/pdf/2504.21038v1)

**Authors**: Yakai Li, Jiekang Hu, Weiduan Sang, Luping Ma, Jing Xie, Weijuan Zhang, Aimin Yu, Shijie Zhao, Qingjia Huang, Qihang Zhou

**Abstract**: Large Language Models (LLMs) are designed to generate helpful and safe content. However, adversarial attacks, commonly referred to as jailbreak, can bypass their safety protocols, prompting LLMs to generate harmful content or reveal sensitive data. Consequently, investigating jailbreak methodologies is crucial for exposing systemic vulnerabilities within LLMs, ultimately guiding the continuous implementation of security enhancements by developers. In this paper, we introduce a novel jailbreak attack method that leverages the prefilling feature of LLMs, a feature designed to enhance model output constraints. Unlike traditional jailbreak methods, the proposed attack circumvents LLMs' safety mechanisms by directly manipulating the probability distribution of subsequent tokens, thereby exerting control over the model's output. We propose two attack variants: Static Prefilling (SP), which employs a universal prefill text, and Optimized Prefilling (OP), which iteratively optimizes the prefill text to maximize the attack success rate. Experiments on six state-of-the-art LLMs using the AdvBench benchmark validate the effectiveness of our method and demonstrate its capability to substantially enhance attack success rates when combined with existing jailbreak approaches. The OP method achieved attack success rates of up to 99.82% on certain models, significantly outperforming baseline methods. This work introduces a new jailbreak attack method in LLMs, emphasizing the need for robust content validation mechanisms to mitigate the adversarial exploitation of prefilling features. All code and data used in this paper are publicly available.

摘要: 大型语言模型（LLM）旨在生成有用且安全的内容。然而，对抗性攻击（通常称为越狱）可能会绕过其安全协议，促使LLM生成有害内容或泄露敏感数据。因此，调查越狱方法对于暴露LLC内的系统漏洞至关重要，最终指导开发人员持续实施安全增强。在本文中，我们引入了一种新颖的越狱攻击方法，该方法利用了LLM的预填充功能，该功能旨在增强模型输出约束。与传统的越狱方法不同，提出的攻击通过直接操纵后续令牌的概率分布来规避LLM的安全机制，从而对模型的输出施加控制。我们提出了两种攻击变体：采用通用预填充文本的静态预填充（SP）和迭代优化预填充文本以最大化攻击成功率的优化预填充（OP）。使用AdvBench基准对六种最先进的LLM进行实验验证了我们方法的有效性，并证明了其与现有越狱方法相结合时能够大幅提高攻击成功率。OP方法在某些模型上的攻击成功率高达99.82%，显着优于基线方法。这项工作在LLM中引入了一种新的越狱攻击方法，强调需要强大的内容验证机制来减轻对预填充功能的对抗性利用。本文中使用的所有代码和数据都是公开的。



## **48. FCGHunter: Towards Evaluating Robustness of Graph-Based Android Malware Detection**

FCGHunter：评估基于图形的Android恶意软件检测的稳健性 cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19456v1) [paper-pdf](http://arxiv.org/pdf/2504.19456v1)

**Authors**: Shiwen Song, Xiaofei Xie, Ruitao Feng, Qi Guo, Sen Chen

**Abstract**: Graph-based detection methods leveraging Function Call Graphs (FCGs) have shown promise for Android malware detection (AMD) due to their semantic insights. However, the deployment of malware detectors in dynamic and hostile environments raises significant concerns about their robustness. While recent approaches evaluate the robustness of FCG-based detectors using adversarial attacks, their effectiveness is constrained by the vast perturbation space, particularly across diverse models and features.   To address these challenges, we introduce FCGHunter, a novel robustness testing framework for FCG-based AMD systems. Specifically, FCGHunter employs innovative techniques to enhance exploration and exploitation within this huge search space. Initially, it identifies critical areas within the FCG related to malware behaviors to narrow down the perturbation space. We then develop a dependency-aware crossover and mutation method to enhance the validity and diversity of perturbations, generating diverse FCGs. Furthermore, FCGHunter leverages multi-objective feedback to select perturbed FCGs, significantly improving the search process with interpretation-based feature change feedback.   Extensive evaluations across 40 scenarios demonstrate that FCGHunter achieves an average attack success rate of 87.9%, significantly outperforming baselines by at least 44.7%. Notably, FCGHunter achieves a 100% success rate on robust models (e.g., AdaBoost with MalScan), where baselines achieve only 11% or are inapplicable.

摘要: 利用函数调用图（FCG）的基于图的检测方法因其语义洞察而显示出Android恶意软件检测（AMD）的前景。然而，在动态和敌对环境中部署恶意软件检测器引发了对其稳健性的严重担忧。虽然最近的方法使用对抗性攻击来评估基于FCG的检测器的稳健性，但其有效性受到巨大扰动空间的限制，特别是在不同的模型和特征中。   为了应对这些挑战，我们引入了FCGHunter，这是一个针对基于FCG的AMD系统的新型稳健性测试框架。具体来说，FCGHunter采用创新技术来加强这个巨大搜索空间中的探索和利用。最初，它会识别FCG中与恶意软件行为相关的关键区域，以缩小干扰空间。然后，我们开发了一种依赖性感知的交叉和突变方法，以增强扰动的有效性和多样性，生成不同的FCG。此外，FCGHunter利用多目标反馈来选择受干扰的FCG，通过基于解释的特征更改反馈显着改进搜索过程。   对40个场景的广泛评估表明，FCGHunter的平均攻击成功率为87.9%，明显优于基线至少44.7%。值得注意的是，FCGHunter在稳健模型上实现了100%的成功率（例如，AdaBoost with MalScan），基线仅达到11%或不适用。



## **49. Mitigating Evasion Attacks in Federated Learning-Based Signal Classifiers**

缓解基于联邦学习的信号分类器中的逃避攻击 eess.SP

Accepted for publication in IEEE Transactions on Network Science and  Engineering. arXiv admin note: substantial text overlap with arXiv:2301.08866

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2306.04872v2) [paper-pdf](http://arxiv.org/pdf/2306.04872v2)

**Authors**: Su Wang, Rajeev Sahay, Adam Piaseczny, Christopher G. Brinton

**Abstract**: Recent interest in leveraging federated learning (FL) for radio signal classification (SC) tasks has shown promise but FL-based SC remains susceptible to model poisoning adversarial attacks. These adversarial attacks mislead the ML model training process, damaging ML models across the network and leading to lower SC performance. In this work, we seek to mitigate model poisoning adversarial attacks on FL-based SC by proposing the Underlying Server Defense of Federated Learning (USD-FL). Unlike existing server-driven defenses, USD-FL does not rely on perfect network information, i.e., knowing the quantity of adversaries, the adversarial attack architecture, or the start time of the adversarial attacks. Our proposed USD-FL methodology consists of deriving logits for devices' ML models on a reserve dataset, comparing pair-wise logits via 1-Wasserstein distance and then determining a time-varying threshold for adversarial detection. As a result, USD-FL effectively mitigates model poisoning attacks introduced in the FL network. Specifically, when baseline server-driven defenses do have perfect network information, USD-FL outperforms them by (i) improving final ML classification accuracies by at least 6%, (ii) reducing false positive adversary detection rates by at least 10%, and (iii) decreasing the total number of misclassified signals by over 8%. Moreover, when baseline defenses do not have perfect network information, we show that USD-FL achieves accuracies of approximately 74.1% and 62.5% in i.i.d. and non-i.i.d. settings, outperforming existing server-driven baselines, which achieve 52.1% and 39.2% in i.i.d. and non-i.i.d. settings, respectively.

摘要: 最近人们对利用联邦学习（FL）进行无线电信号分类（SC）任务的兴趣已经显示出希望，但基于FL的SC仍然容易受到模型中毒对抗攻击的影响。这些对抗性攻击误导了ML模型训练过程，损害了整个网络的ML模型，并导致SC性能下降。在这项工作中，我们试图通过提出联邦学习的底层服务器防御（USD-FL）来减轻对基于FL的SC的模型中毒对抗攻击。与现有的服务器驱动防御不同，USD-FL不依赖于完美的网络信息，即了解对手的数量、对抗性攻击架构或对抗性攻击的开始时间。我们提出的USD-FL方法包括在储备数据集中推导设备ML模型的logit，通过1-Wasserstein距离比较成对logit，然后确定对抗检测的时变阈值。因此，USD-FL有效缓解了FL网络中引入的模型中毒攻击。具体来说，当基线服务器驱动的防御确实具有完美的网络信息时，USD-FL通过以下方式优于它们：（i）将最终ML分类准确性提高至少6%，（ii）将误报对手检测率降低至少10%，以及（iii）将错误分类的信号总数减少超过8%。此外，当基线防御没有完美的网络信息时，我们表明USD-FL在i. i. d方面的准确性约为74.1%和62.5%。和非i.i.d.设置的表现优于现有的服务器驱动基准，后者的i. i. d分别达到52.1%和39.2%和非i.i.d.设置，分别。



## **50. Forging and Removing Latent-Noise Diffusion Watermarks Using a Single Image**

基于单幅图像的隐噪声扩散水印的伪造与去除 cs.CV

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2504.20111v1) [paper-pdf](http://arxiv.org/pdf/2504.20111v1)

**Authors**: Anubhav Jain, Yuya Kobayashi, Naoki Murata, Yuhta Takida, Takashi Shibuya, Yuki Mitsufuji, Niv Cohen, Nasir Memon, Julian Togelius

**Abstract**: Watermarking techniques are vital for protecting intellectual property and preventing fraudulent use of media. Most previous watermarking schemes designed for diffusion models embed a secret key in the initial noise. The resulting pattern is often considered hard to remove and forge into unrelated images. In this paper, we propose a black-box adversarial attack without presuming access to the diffusion model weights. Our attack uses only a single watermarked example and is based on a simple observation: there is a many-to-one mapping between images and initial noises. There are regions in the clean image latent space pertaining to each watermark that get mapped to the same initial noise when inverted. Based on this intuition, we propose an adversarial attack to forge the watermark by introducing perturbations to the images such that we can enter the region of watermarked images. We show that we can also apply a similar approach for watermark removal by learning perturbations to exit this region. We report results on multiple watermarking schemes (Tree-Ring, RingID, WIND, and Gaussian Shading) across two diffusion models (SDv1.4 and SDv2.0). Our results demonstrate the effectiveness of the attack and expose vulnerabilities in the watermarking methods, motivating future research on improving them.

摘要: 水印技术对于保护知识产权和防止媒体欺诈性使用至关重要。大多数以前为扩散模型设计的水印方案都在初始噪音中嵌入秘密密钥。生成的图案通常被认为很难删除并伪造成不相关的图像。在本文中，我们提出了一种黑匣子对抗攻击，而不假设访问扩散模型权重。我们的攻击仅使用一个带水印的示例，并且基于一个简单的观察：图像和初始噪音之间存在多对一的映射。干净图像潜空间中存在与每个水印相关的区域，这些区域在倒置时映射到相同的初始噪音。基于这一直觉，我们提出了一种对抗攻击来通过向图像引入扰动来伪造水印，以便我们可以进入带有水印的图像的区域。我们表明，我们还可以通过学习扰动退出该区域来应用类似的方法来去除水印。我们报告了跨两个扩散模型（SDv1.4和SDv2.0）的多种水印方案（Tree-Ring、RingID、WIND和高斯着色）的结果。我们的结果证明了攻击的有效性，并暴露了水印方法中的漏洞，激励了未来改进它们的研究。



