# Latest Adversarial Attack Papers
**update at 2025-05-26 16:05:17**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Revisiting Adversarial Perception Attacks and Defense Methods on Autonomous Driving Systems**

重新审视自动驾驶系统的敌对感知攻击和防御方法 cs.RO

8 pages, 2 figures, To appear in the 8th Dependable and Secure  Machine Learning Workshop (DSML 2025)

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.11532v2) [paper-pdf](http://arxiv.org/pdf/2505.11532v2)

**Authors**: Cheng Chen, Yuhong Wang, Nafis S Munir, Xiangwei Zhou, Xugui Zhou

**Abstract**: Autonomous driving systems (ADS) increasingly rely on deep learning-based perception models, which remain vulnerable to adversarial attacks. In this paper, we revisit adversarial attacks and defense methods, focusing on road sign recognition and lead object detection and prediction (e.g., relative distance). Using a Level-2 production ADS, OpenPilot by Comma$.$ai, and the widely adopted YOLO model, we systematically examine the impact of adversarial perturbations and assess defense techniques, including adversarial training, image processing, contrastive learning, and diffusion models. Our experiments highlight both the strengths and limitations of these methods in mitigating complex attacks. Through targeted evaluations of model robustness, we aim to provide deeper insights into the vulnerabilities of ADS perception systems and contribute guidance for developing more resilient defense strategies.

摘要: 自动驾驶系统（ADS）越来越依赖基于深度学习的感知模型，而这些模型仍然容易受到对抗性攻击。在本文中，我们重新审视了对抗性攻击和防御方法，重点关注路标识别以及引导对象检测和预测（例如，相对距离）。使用2级生产ADS，OpenPilot by Comma$.$ ai和广泛采用的YOLO模型，我们系统地检查了对抗性扰动的影响并评估防御技术，包括对抗性训练、图像处理、对比学习和扩散模型。我们的实验强调了这些方法在减轻复杂攻击方面的优势和局限性。通过对模型稳健性进行有针对性的评估，我们的目标是为ADS感知系统的漏洞提供更深入的见解，并为制定更具弹性的防御策略提供指导。



## **2. Towards more transferable adversarial attack in black-box manner**

以黑匣子方式实现更具转移性的对抗攻击 cs.LG

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.18097v1) [paper-pdf](http://arxiv.org/pdf/2505.18097v1)

**Authors**: Chun Tong Lei, Zhongliang Guo, Hon Chung Lee, Minh Quoc Duong, Chun Pong Lau

**Abstract**: Adversarial attacks have become a well-explored domain, frequently serving as evaluation baselines for model robustness. Among these, black-box attacks based on transferability have received significant attention due to their practical applicability in real-world scenarios. Traditional black-box methods have generally focused on improving the optimization framework (e.g., utilizing momentum in MI-FGSM) to enhance transferability, rather than examining the dependency on surrogate white-box model architectures. Recent state-of-the-art approach DiffPGD has demonstrated enhanced transferability by employing diffusion-based adversarial purification models for adaptive attacks. The inductive bias of diffusion-based adversarial purification aligns naturally with the adversarial attack process, where both involving noise addition, reducing dependency on surrogate white-box model selection. However, the denoising process of diffusion models incurs substantial computational costs through chain rule derivation, manifested in excessive VRAM consumption and extended runtime. This progression prompts us to question whether introducing diffusion models is necessary. We hypothesize that a model sharing similar inductive bias to diffusion-based adversarial purification, combined with an appropriate loss function, could achieve comparable or superior transferability while dramatically reducing computational overhead. In this paper, we propose a novel loss function coupled with a unique surrogate model to validate our hypothesis. Our approach leverages the score of the time-dependent classifier from classifier-guided diffusion models, effectively incorporating natural data distribution knowledge into the adversarial optimization process. Experimental results demonstrate significantly improved transferability across diverse model architectures while maintaining robustness against diffusion-based defenses.

摘要: 对抗性攻击已成为一个充分探索的领域，经常作为模型稳健性的评估基线。其中，基于可移植性的黑匣子攻击因其在现实世界场景中的实际适用性而受到了广泛关注。传统的黑匣子方法通常专注于改进优化框架（例如，利用MI-FGSM中的动量）来增强可移植性，而不是检查对替代白盒模型架构的依赖性。最近的国家的最先进的方法DiffPGD已经证明了增强的可转移性，采用基于扩散的对抗净化模型的自适应攻击。基于扩散的对抗性净化的归纳偏差与对抗性攻击过程自然一致，两者都涉及噪声添加，减少了对替代白盒模型选择的依赖。然而，扩散模型的去噪过程通过链式规则推导产生大量的计算成本，表现在过多的VRAM消耗和延长的运行时间。这一进展促使我们质疑是否有必要引入扩散模型。我们假设，与基于扩散的对抗性纯化具有类似的诱导偏差的模型，结合适当的损失函数，可以实现相当或更好的可移植性，同时显着减少计算费用。在本文中，我们提出了一种新颖的损失函数，并结合了一个独特的代理模型来验证我们的假设。我们的方法利用分类器引导的扩散模型中的时间相关分类器的分数，有效地将自然数据分布知识融入到对抗优化过程中。实验结果表明，不同模型架构之间的可移植性显着提高，同时保持针对基于扩散的防御的鲁棒性。



## **3. CAMME: Adaptive Deepfake Image Detection with Multi-Modal Cross-Attention**

CAMME：具有多模态交叉注意的自适应Deepfake图像检测 cs.CV

20 pages, 8 figures, 12 Tables

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.18035v1) [paper-pdf](http://arxiv.org/pdf/2505.18035v1)

**Authors**: Naseem Khan, Tuan Nguyen, Amine Bermak, Issa Khalil

**Abstract**: The proliferation of sophisticated AI-generated deepfakes poses critical challenges for digital media authentication and societal security. While existing detection methods perform well within specific generative domains, they exhibit significant performance degradation when applied to manipulations produced by unseen architectures--a fundamental limitation as generative technologies rapidly evolve. We propose CAMME (Cross-Attention Multi-Modal Embeddings), a framework that dynamically integrates visual, textual, and frequency-domain features through a multi-head cross-attention mechanism to establish robust cross-domain generalization. Extensive experiments demonstrate CAMME's superiority over state-of-the-art methods, yielding improvements of 12.56% on natural scenes and 13.25% on facial deepfakes. The framework demonstrates exceptional resilience, maintaining (over 91%) accuracy under natural image perturbations and achieving 89.01% and 96.14% accuracy against PGD and FGSM adversarial attacks, respectively. Our findings validate that integrating complementary modalities through cross-attention enables more effective decision boundary realignment for reliable deepfake detection across heterogeneous generative architectures.

摘要: 复杂的人工智能生成的深度造假的激增给数字媒体认证和社会安全带来了严峻的挑战。虽然现有的检测方法在特定的生成域中表现良好，但当应用于由未见架构产生的操作时，它们会表现出显着的性能下降--这是生成技术迅速发展的根本限制。我们提出了CAME（交叉注意多模式嵌入），这是一个通过多头交叉注意机制动态集成视觉、文本和频域特征的框架，以建立强大的跨域概括。大量实验证明了CAME相对于最先进方法的优越性，在自然场景上提高了12.56%，在面部深度造假上提高了13.25%。该框架表现出出色的弹性，在自然图像扰动下保持（超过91%）的准确性，并分别实现89.01%和96.14%的准确性。我们的研究结果证实，通过交叉注意力集成补充模式可以更有效地重新调整决策边界，以实现跨异类生成架构的可靠深度伪造检测。



## **4. Towards Copyright Protection for Knowledge Bases of Retrieval-augmented Language Models via Reasoning**

通过推理实现检索增强语言模型知识库的版权保护 cs.CR

The first two authors contributed equally to this work. 25 pages

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2502.10440v2) [paper-pdf](http://arxiv.org/pdf/2502.10440v2)

**Authors**: Junfeng Guo, Yiming Li, Ruibo Chen, Yihan Wu, Chenxi Liu, Yanshuo Chen, Heng Huang

**Abstract**: Large language models (LLMs) are increasingly integrated into real-world personalized applications through retrieval-augmented generation (RAG) mechanisms to supplement their responses with domain-specific knowledge. However, the valuable and often proprietary nature of the knowledge bases used in RAG introduces the risk of unauthorized usage by adversaries. Existing methods that can be generalized as watermarking techniques to protect these knowledge bases typically involve poisoning or backdoor attacks. However, these methods require altering the LLM's results of verification samples, inevitably making these watermarks susceptible to anomaly detection and even introducing new security risks. To address these challenges, we propose \name{} for `harmless' copyright protection of knowledge bases. Instead of manipulating LLM's final output, \name{} implants distinct yet benign verification behaviors in the space of chain-of-thought (CoT) reasoning, maintaining the correctness of the final answer. Our method has three main stages: (1) Generating CoTs: For each verification question, we generate two `innocent' CoTs, including a target CoT for building watermark behaviors; (2) Optimizing Watermark Phrases and Target CoTs: Inspired by our theoretical analysis, we optimize them to minimize retrieval errors under the \emph{black-box} and \emph{text-only} setting of suspicious LLM, ensuring that only watermarked verification queries can retrieve their correspondingly target CoTs contained in the knowledge base; (3) Ownership Verification: We exploit a pairwise Wilcoxon test to verify whether a suspicious LLM is augmented with the protected knowledge base by comparing its responses to watermarked and benign verification queries. Our experiments on diverse benchmarks demonstrate that \name{} effectively protects knowledge bases and its resistance to adaptive attacks.

摘要: 大型语言模型（LLM）通过检索增强生成（RAG）机制越来越多地集成到现实世界的个性化应用程序中，以用特定领域的知识补充其响应。然而，RAG中使用的知识库的宝贵且通常是专有的，这带来了对手未经授权使用的风险。可以概括为保护这些知识库的水印技术的现有方法通常涉及中毒或后门攻击。然而，这些方法需要改变LLM的验证样本结果，不可避免地使这些水印容易受到异常检测，甚至引入新的安全风险。为了应对这些挑战，我们提议\Name{}对知识库进行“无害”版权保护。\Name{}不是操纵LLM的最终输出，而是在思想链（CoT）推理空间中植入独特但良性的验证行为，以保持最终答案的正确性。我们的方法有三个主要阶段：（1）生成CoT：对于每个验证问题，我们生成两个“无辜”CoT，包括用于构建水印行为的目标CoT;（2）优化水印短语和目标CoT：受我们理论分析的启发，我们对它们进行了优化，以最大限度地减少可疑LLM的\{black-box}和\{text-only}设置下的检索错误，确保只有带水印的验证查询才能检索知识库中包含的相应目标CoT;（3）所有权验证：我们利用成对Wilcoxon测试来验证可疑LLM是否通过比较其响应与带水印和良性验证查询进行比较来使用受保护的知识库进行扩展。我们对不同基准的实验表明\Name{}可以有效地保护知识库及其对适应性攻击的抵抗力。



## **5. SemSegBench & DetecBench: Benchmarking Reliability and Generalization Beyond Classification**

SemSegBench和DetecBench：基准可靠性和超越分类的一般化 cs.CV

First seven listed authors have equal contribution. GitHub:  https://github.com/shashankskagnihotri/benchmarking_reliability_generalization.  arXiv admin note: text overlap with arXiv:2505.05091

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.18015v1) [paper-pdf](http://arxiv.org/pdf/2505.18015v1)

**Authors**: Shashank Agnihotri, David Schader, Jonas Jakubassa, Nico Sharei, Simon Kral, Mehmet Ege Kaçar, Ruben Weber, Margret Keuper

**Abstract**: Reliability and generalization in deep learning are predominantly studied in the context of image classification. Yet, real-world applications in safety-critical domains involve a broader set of semantic tasks, such as semantic segmentation and object detection, which come with a diverse set of dedicated model architectures. To facilitate research towards robust model design in segmentation and detection, our primary objective is to provide benchmarking tools regarding robustness to distribution shifts and adversarial manipulations. We propose the benchmarking tools SEMSEGBENCH and DETECBENCH, along with the most extensive evaluation to date on the reliability and generalization of semantic segmentation and object detection models. In particular, we benchmark 76 segmentation models across four datasets and 61 object detectors across two datasets, evaluating their performance under diverse adversarial attacks and common corruptions. Our findings reveal systematic weaknesses in state-of-the-art models and uncover key trends based on architecture, backbone, and model capacity. SEMSEGBENCH and DETECBENCH are open-sourced in our GitHub repository (https://github.com/shashankskagnihotri/benchmarking_reliability_generalization) along with our complete set of total 6139 evaluations. We anticipate the collected data to foster and encourage future research towards improved model reliability beyond classification.

摘要: 深度学习的可靠性和概括性主要是在图像分类的背景下研究的。然而，安全关键领域中的现实世界应用程序涉及更广泛的语义任务，例如语义分割和对象检测，这些任务具有一组多样化的专用模型架构。为了促进分割和检测中稳健模型设计的研究，我们的主要目标是提供有关分布变化和对抗操纵稳健性的基准工具。我们提出了基准测试工具SEMSEGBENCH和DETEDETEBENCH，以及迄今为止对语义分割和对象检测模型的可靠性和概括性进行了最广泛的评估。特别是，我们对四个数据集上的76个分割模型和两个数据集上的61个对象检测器进行了基准测试，评估了它们在各种对抗性攻击和常见腐败情况下的性能。我们的研究结果揭示了最先进模型的系统性弱点，并揭示了基于架构、主干和模型容量的关键趋势。SEMSEGBENCH和DETECBENCH在我们的GitHub存储库（https：//github.com/shashankskagnihotri/benchmarking_reliability_generalization）中开源，以及我们的完整集合，总共6139个评估。我们预计收集的数据，以促进和鼓励未来的研究，以提高模型的可靠性超越分类。



## **6. Superplatforms Have to Attack AI Agents**

超级平台不得不攻击AI代理 cs.AI

Position paper under review

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17861v1) [paper-pdf](http://arxiv.org/pdf/2505.17861v1)

**Authors**: Jianghao Lin, Jiachen Zhu, Zheli Zhou, Yunjia Xi, Weiwen Liu, Yong Yu, Weinan Zhang

**Abstract**: Over the past decades, superplatforms, digital companies that integrate a vast range of third-party services and applications into a single, unified ecosystem, have built their fortunes on monopolizing user attention through targeted advertising and algorithmic content curation. Yet the emergence of AI agents driven by large language models (LLMs) threatens to upend this business model. Agents can not only free user attention with autonomy across diverse platforms and therefore bypass the user-attention-based monetization, but might also become the new entrance for digital traffic. Hence, we argue that superplatforms have to attack AI agents to defend their centralized control of digital traffic entrance. Specifically, we analyze the fundamental conflict between user-attention-based monetization and agent-driven autonomy through the lens of our gatekeeping theory. We show how AI agents can disintermediate superplatforms and potentially become the next dominant gatekeepers, thereby forming the urgent necessity for superplatforms to proactively constrain and attack AI agents. Moreover, we go through the potential technologies for superplatform-initiated attacks, covering a brand-new, unexplored technical area with unique challenges. We have to emphasize that, despite our position, this paper does not advocate for adversarial attacks by superplatforms on AI agents, but rather offers an envisioned trend to highlight the emerging tensions between superplatforms and AI agents. Our aim is to raise awareness and encourage critical discussion for collaborative solutions, prioritizing user interests and perserving the openness of digital ecosystems in the age of AI agents.

摘要: 在过去的几十年里，超级平台和数字公司将大量第三方服务和应用程序集成到单一、统一的生态系统中，通过有针对性的广告和算法内容策展垄断用户注意力，积累了自己的财富。然而，由大型语言模型（LLM）驱动的人工智能代理的出现可能会颠覆这种商业模式。代理商不仅可以在不同平台上自主释放用户注意力，从而绕过基于用户注意力的货币化，而且还可能成为数字流量的新入口。因此，我们认为超级平台必须攻击人工智能代理来捍卫他们对数字流量入口的集中控制。具体来说，我们通过守门理论的视角分析了基于用户注意力的货币化和代理驱动的自主性之间的根本冲突。我们展示了人工智能代理如何摆脱超级平台的中间化，并有可能成为下一个占主导地位的守门人，从而形成超级平台主动约束和攻击人工智能代理的迫切需要。此外，我们还探讨了超级平台发起的攻击的潜在技术，涵盖了一个全新的、未经探索的、具有独特挑战的技术领域。我们必须强调的是，尽管我们的立场，但本文并不主张超级平台对人工智能代理进行对抗攻击，而是提供了一种设想的趋势来强调超级平台和人工智能代理之间正在出现的紧张关系。我们的目标是提高人们的认识并鼓励对协作解决方案进行批判性讨论，优先考虑用户兴趣并在人工智能代理时代保持数字生态系统的开放性。



## **7. Temporal Consistency Constrained Transferable Adversarial Attacks with Background Mixup for Action Recognition**

用于动作识别的时间一致性约束的具有背景混淆的可转移对抗攻击 cs.CV

Accepted in IJCAI'25

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17807v1) [paper-pdf](http://arxiv.org/pdf/2505.17807v1)

**Authors**: Ping Li, Jianan Ni, Bo Pang

**Abstract**: Action recognition models using deep learning are vulnerable to adversarial examples, which are transferable across other models trained on the same data modality. Existing transferable attack methods face two major challenges: 1) they heavily rely on the assumption that the decision boundaries of the surrogate (a.k.a., source) model and the target model are similar, which limits the adversarial transferability; and 2) their decision boundary difference makes the attack direction uncertain, which may result in the gradient oscillation, weakening the adversarial attack. This motivates us to propose a Background Mixup-induced Temporal Consistency (BMTC) attack method for action recognition. From the input transformation perspective, we design a model-agnostic background adversarial mixup module to reduce the surrogate-target model dependency. In particular, we randomly sample one video from each category and make its background frame, while selecting the background frame with the top attack ability for mixup with the clean frame by reinforcement learning. Moreover, to ensure an explicit attack direction, we leverage the background category as guidance for updating the gradient of adversarial example, and design a temporal gradient consistency loss, which strengthens the stability of the attack direction on subsequent frames. Empirical studies on two video datasets, i.e., UCF101 and Kinetics-400, and one image dataset, i.e., ImageNet, demonstrate that our method significantly boosts the transferability of adversarial examples across several action/image recognition models. Our code is available at https://github.com/mlvccn/BMTC_TransferAttackVid.

摘要: 使用深度学习的动作识别模型容易受到对抗性示例的影响，这些示例可以在相同数据模式上训练的其他模型之间转移。现有的可转移攻击方法面临两个主要挑战：1）它们严重依赖于代理的决策边界（也称为，源模型与目标模型相似，限制了对抗性的可转移性; 2）源模型与目标模型的决策边界差异使得攻击方向不确定，可能导致梯度振荡，削弱对抗性攻击。这促使我们提出了一个背景混淆诱导的时间一致性（BMTC）攻击方法的动作识别。从输入转换的角度，我们设计了一个模型不可知的背景对抗混合模块，以减少代理目标模型的依赖性。特别是，我们从每个类别中随机采样一个视频并制作其背景帧，同时选择攻击能力最强的背景帧，通过强化学习与干净帧混合。此外，为了确保明确的攻击方向，我们利用背景类别作为更新对抗示例梯度的指导，并设计时间梯度一致性损失，增强了后续帧攻击方向的稳定性。对两个视频数据集的实证研究，即UCF 101和Kinetics-400，以及一个图像数据集，即ImageNet证明我们的方法显着提高了对抗性示例在多个动作/图像识别模型之间的可移植性。我们的代码可在https://github.com/mlvccn/BMTC_TransferAttackVid上获取。



## **8. Adversarial Robustness in Two-Stage Learning-to-Defer: Algorithms and Guarantees**

两阶段学习推迟中的对抗稳健性：算法和保证 stat.ML

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2502.01027v2) [paper-pdf](http://arxiv.org/pdf/2502.01027v2)

**Authors**: Yannis Montreuil, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Two-stage Learning-to-Defer (L2D) enables optimal task delegation by assigning each input to either a fixed main model or one of several offline experts, supporting reliable decision-making in complex, multi-agent environments. However, existing L2D frameworks assume clean inputs and are vulnerable to adversarial perturbations that can manipulate query allocation--causing costly misrouting or expert overload. We present the first comprehensive study of adversarial robustness in two-stage L2D systems. We introduce two novel attack strategie--untargeted and targeted--which respectively disrupt optimal allocations or force queries to specific agents. To defend against such threats, we propose SARD, a convex learning algorithm built on a family of surrogate losses that are provably Bayes-consistent and $(\mathcal{R}, \mathcal{G})$-consistent. These guarantees hold across classification, regression, and multi-task settings. Empirical results demonstrate that SARD significantly improves robustness under adversarial attacks while maintaining strong clean performance, marking a critical step toward secure and trustworthy L2D deployment.

摘要: 两阶段学习延迟（L2 D）通过将每个输入分配给固定的主模型或多个离线专家之一来实现最佳任务委托，支持复杂的多代理环境中的可靠决策。然而，现有的L2 D框架假设干净的输入，并且容易受到可以操纵查询分配的对抗性扰动的影响，从而导致代价高昂的错误路由或专家超载。我们首次对两阶段L2 D系统中的对抗鲁棒性进行了全面研究。我们引入了两种新颖的攻击策略--无针对性和有针对性--它们分别扰乱最佳分配或强制向特定代理进行查询。为了抵御此类威胁，我们提出了SAARD，这是一种凸学习算法，它建立在一系列可证明Bayes-一致且$（\mathCal{R}，\mathCal{G}）$-一致的替代损失之上。这些保证适用于分类、回归和多任务设置。经验结果表明，SAARD显着提高了对抗攻击下的鲁棒性，同时保持了强大的清洁性能，标志着迈向安全且值得信赖的L2 D部署的关键一步。



## **9. Architecture Selection via the Trade-off Between Accuracy and Robustness**

通过准确性和鲁棒性之间的权衡选择架构 cs.LG

Incorporated in a later submission. This submission is not complete  in results

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/1906.01354v2) [paper-pdf](http://arxiv.org/pdf/1906.01354v2)

**Authors**: Zhun Deng, Cynthia Dwork, Jialiang Wang, Yao Zhao

**Abstract**: We provide a general framework for characterizing the trade-off between accuracy and robustness in supervised learning. We propose a method and define quantities to characterize the trade-off between accuracy and robustness for a given architecture, and provide theoretical insight into the trade-off. Specifically we introduce a simple trade-off curve, define and study an influence function that captures the sensitivity, under adversarial attack, of the optima of a given loss function. We further show how adversarial training regularizes the parameters in an over-parameterized linear model, recovering the LASSO and ridge regression as special cases, which also allows us to theoretically analyze the behavior of the trade-off curve. In experiments, we demonstrate the corresponding trade-off curves of neural networks and how they vary with respect to factors such as number of layers, neurons, and across different network structures. Such information provides a useful guideline to architecture selection.

摘要: 我们提供了一个通用框架来描述监督学习中准确性和稳健性之间的权衡。我们提出了一种方法并定义了量化来描述给定架构的准确性和稳健性之间的权衡，并提供了对权衡的理论见解。具体来说，我们引入了一个简单的权衡曲线，定义和研究了一个影响函数，该函数捕捉了在对抗攻击下给定损失函数最优值的敏感性。我们进一步展示了对抗训练如何在过度参数化线性模型中对参数进行正规化，将LASO和岭回归恢复为特殊情况，这也使我们能够从理论上分析权衡曲线的行为。在实验中，我们展示了神经网络的相应权衡曲线，以及它们如何随着层数、神经元数和不同网络结构等因素而变化。此类信息为建筑选择提供了有用的指南。



## **10. Sec5GLoc: Securing 5G Indoor Localization via Adversary-Resilient Deep Learning Architecture**

Sec 5GLoc：通过对抗弹性深度学习架构确保5G室内本地化 cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17776v1) [paper-pdf](http://arxiv.org/pdf/2505.17776v1)

**Authors**: Ildi Alla, Valeria Loscri

**Abstract**: Emerging 5G millimeter-wave and sub-6 GHz networks enable high-accuracy indoor localization, but security and privacy vulnerabilities pose serious challenges. In this paper, we identify and address threats including location spoofing and adversarial signal manipulation against 5G-based indoor localization. We formalize a threat model encompassing attackers who inject forged radio signals or perturb channel measurements to mislead the localization system. To defend against these threats, we propose an adversary-resilient localization architecture that combines deep learning fingerprinting with physical domain knowledge. Our approach integrates multi-anchor Channel Impulse Response (CIR) fingerprints with Time Difference of Arrival (TDoA) features and known anchor positions in a hybrid Convolutional Neural Network (CNN) and multi-head attention network. This design inherently checks geometric consistency and dynamically down-weights anomalous signals, making localization robust to tampering. We formulate the secure localization problem and demonstrate, through extensive experiments on a public 5G indoor dataset, that the proposed system achieves a mean error approximately 0.58 m under mixed Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) trajectories in benign conditions and gracefully degrades to around 0.81 m under attack scenarios. We also show via ablation studies that each architecture component (attention mechanism, TDoA, etc.) is critical for both accuracy and resilience, reducing errors by 4-5 times compared to baselines. In addition, our system runs in real-time, localizing the user in just 1 ms on a simple CPU. The code has been released to ensure reproducibility (https://github.com/sec5gloc/Sec5GLoc).

摘要: 新兴的5G毫米波和6 GHz以下网络可以实现高准确度的室内定位，但安全和隐私漏洞带来了严重挑战。在本文中，我们识别并解决了针对基于5G的室内定位的位置欺骗和对抗信号操纵等威胁。我们正式化了一个威胁模型，其中包括注入伪造无线电信号或扰乱频道测量以误导定位系统的攻击者。为了抵御这些威胁，我们提出了一种具有对抗能力的本地化架构，该架构将深度学习指纹识别与物理领域知识相结合。我们的方法将多锚信道脉冲响应（CIR）指纹与到达时间差（TDoA）特征和已知锚点位置集成在混合卷积神经网络（CNN）和多头注意力网络中。这种设计本质上检查几何一致性，并动态地降低异常信号的权重，使定位对篡改具有鲁棒性。我们制定了安全定位问题，并通过对公共5G室内数据集的广泛实验证明，所提出的系统在良性条件下在混合视线（LOS）和非视线（NLOS）轨迹下实现了约0.58 m的平均误差，并在攻击场景下优雅地下降到约0.81 m。我们还通过消融研究表明，每个架构组件（注意力机制，TDoA等）。对于准确性和弹性都至关重要，与基线相比，误差减少了4-5倍。此外，我们的系统实时运行，在简单的CPU上仅需1 ms即可定位用户。代码已经发布以确保可重复性（https：//github.com/sec5gloc/Sec5GLoc）。



## **11. DiffBreak: Is Diffusion-Based Purification Robust?**

迪夫Break：基于扩散的净化是否稳健？ cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2411.16598v3) [paper-pdf](http://arxiv.org/pdf/2411.16598v3)

**Authors**: Andre Kassis, Urs Hengartner, Yaoliang Yu

**Abstract**: Diffusion-based purification (DBP) has become a cornerstone defense against adversarial examples (AEs), regarded as robust due to its use of diffusion models (DMs) that project AEs onto the natural data manifold. We refute this core claim, theoretically proving that gradient-based attacks effectively target the DM rather than the classifier, causing DBP's outputs to align with adversarial distributions. This prompts a reassessment of DBP's robustness, attributing it to two critical flaws: incorrect gradients and inappropriate evaluation protocols that test only a single random purification of the AE. We show that with proper accounting for stochasticity and resubmission risk, DBP collapses. To support this, we introduce DiffBreak, the first reliable toolkit for differentiation through DBP, eliminating gradient flaws that previously further inflated robustness estimates. We also analyze the current defense scheme used for DBP where classification relies on a single purification, pinpointing its inherent invalidity. We provide a statistically grounded majority-vote (MV) alternative that aggregates predictions across multiple purified copies, showing partial but meaningful robustness gain. We then propose a novel adaptation of an optimization method against deepfake watermarking, crafting systemic perturbations that defeat DBP even under MV, challenging DBP's viability.

摘要: 基于扩散的纯化（DAB）已成为对抗性例子（AE）的基石防御，由于其使用将AE投射到自然数据集上的扩散模型（DM），因此被认为是强大的。我们反驳了这一核心主张，从理论上证明基于梯度的攻击有效地针对DM而不是分类器，导致CBP的输出与对抗性分布一致。这促使人们重新评估CBP的稳健性，并将其归因于两个关键缺陷：不正确的梯度和仅测试AE单次随机纯化的不恰当评估方案。我们发现，适当的会计随机性和重新提交的风险，DBP崩溃。为了支持这一点，我们引入了DiffBreak，这是第一个通过DBP进行区分的可靠工具包，消除了以前进一步夸大鲁棒性估计的梯度缺陷。我们还分析了目前的防御计划用于DBP的分类依赖于一个单一的纯化，查明其固有的无效性。我们提供了一个统计接地多数表决（MV）的替代方案，聚合预测跨多个纯化的副本，显示部分但有意义的鲁棒性增益。然后，我们提出了一种针对Deepfake水印的优化方法的新颖调整，精心设计系统性扰动，即使在MV下也能击败CBP，挑战CBP的生存能力。



## **12. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

EVADE：电子商务应用程序中规避内容检测的多模式基准 cs.CL

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17654v1) [paper-pdf](http://arxiv.org/pdf/2505.17654v1)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.

摘要: 电子商务平台越来越依赖大型语言模型（LLM）和视觉语言模型（VLM）来检测非法或误导性产品内容。然而，这些模型仍然容易受到规避内容的影响：表面上遵守平台政策但秘密传达禁止声明的输入（文本或图像）。与导致明显失败的传统对抗性攻击不同，规避内容利用了模糊性和上下文，使其更难检测。现有的稳健性基准对这一要求严格的现实世界挑战几乎没有提供指导。我们引入EVADE，这是第一个由专家策划的中国多模式基准，专门用于评估电子商务中规避内容检测的基础模型。该数据集包含2，833个注释文本样本和13，961张图像，涵盖六个要求严格的产品类别，包括身材塑造、身高增长和保健品。两项补充任务评估不同的能力：Single-Violation（在短提示下探索细粒度推理）和All-in-One（通过将重叠的策略规则合并到统一指令中来测试长上下文推理）。值得注意的是，一体化设置显着缩小了部分匹配准确性和完全匹配准确性之间的性能差距，这表明更清晰的规则定义可以改善人类和模型判断之间的一致性。我们对26种主流LLM和VLM进行了基准测试，并观察到了巨大的性能差距：即使是最先进的模型也经常对规避样本进行错误分类。通过发布EVADE和强大的基线，我们为评估逃避内容检测提供了第一个严格的标准，暴露了当前多模式推理的根本局限性，并为电子商务中更安全、更透明的内容审核系统奠定了基础。该数据集可在https://huggingface.co/datasets/koenshen/EVADE-Bench上公开获取。



## **13. Ownership Verification of DNN Models Using White-Box Adversarial Attacks with Specified Probability Manipulation**

基于指定概率操纵的白盒对抗攻击的DNN模型所有权验证 cs.LG

Accepted to EUSIPCO 2025

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17579v1) [paper-pdf](http://arxiv.org/pdf/2505.17579v1)

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Ishobe, Eisuke Koizumi

**Abstract**: In this paper, we propose a novel framework for ownership verification of deep neural network (DNN) models for image classification tasks. It allows verification of model identity by both the rightful owner and third party without presenting the original model. We assume a gray-box scenario where an unauthorized user owns a model that is illegally copied from the original model, provides services in a cloud environment, and the user throws images and receives the classification results as a probability distribution of output classes. The framework applies a white-box adversarial attack to align the output probability of a specific class to a designated value. Due to the knowledge of original model, it enables the owner to generate such adversarial examples. We propose a simple but effective adversarial attack method based on the iterative Fast Gradient Sign Method (FGSM) by introducing control parameters. Experimental results confirm the effectiveness of the identification of DNN models using adversarial attack.

摘要: 在本文中，我们提出了一种新的框架，用于图像分类任务的深度神经网络（DNN）模型的所有权验证。它允许合法所有者和第三方在不出示原始模型的情况下验证模型身份。我们假设一个灰盒场景，其中未经授权的用户拥有从原始模型非法复制的模型，在云环境中提供服务，用户抛出图像并接收分类结果作为输出类的概率分布。该框架应用白盒对抗攻击来将特定类的输出概率与指定值对齐。由于对原始模型的了解，它使所有者能够生成此类对抗性示例。我们通过引入控制参数，提出了一种基于迭代快速梯度符号法（FGSM）的简单但有效的对抗攻击方法。实验结果证实了使用对抗攻击识别DNN模型的有效性。



## **14. Finetuning-Activated Backdoors in LLMs**

LLM中的微调激活后门 cs.LG

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.16567v2) [paper-pdf](http://arxiv.org/pdf/2505.16567v2)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs.

摘要: 微调可开放访问的大型语言模型（LLM）已成为实现特定任务性能改进的标准实践。到目前为止，微调一直被认为是一个受控且安全的过程，其中对良性数据集的训练会导致可预测的行为。在本文中，我们首次证明对手可以创建有毒的LLM，这些LLM最初看起来是良性的，但一旦被下游用户微调，就会表现出恶意行为。为此，我们提出的攻击FAB（微调激活后门）通过元学习技术毒害LLM，以模拟下游微调，明确优化微调模型中恶意行为的出现。与此同时，有毒的LLM会被规范化，以保留一般能力，并且在微调之前不会表现出恶意行为。因此，当用户在自己的数据集上微调看似良性的模型时，他们会在不知不觉中触发其隐藏的后门行为。我们展示了FAB在多个LLM和三种目标行为中的有效性：未经请求的广告、拒绝和越狱。此外，我们表明FAB后门对于用户做出的各种微调选择是稳健的（例如，数据集、步骤数、调度程序）。我们的发现挑战了有关微调安全性的普遍假设，揭示了另一个利用LLM复杂性的关键攻击载体。



## **15. JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models**

JALMBench：音频语言模型中的越狱漏洞基准 cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17568v1) [paper-pdf](http://arxiv.org/pdf/2505.17568v1)

**Authors**: Zifan Peng, Yule Liu, Zhen Sun, Mingchen Li, Zeren Luo, Jingyi Zheng, Wenhan Dong, Xinlei He, Xuechao Wang, Yingjie Xue, Shengmin Xu, Xinyi Huang

**Abstract**: Audio Language Models (ALMs) have made significant progress recently. These models integrate the audio modality directly into the model, rather than converting speech into text and inputting text to Large Language Models (LLMs). While jailbreak attacks on LLMs have been extensively studied, the security of ALMs with audio modalities remains largely unexplored. Currently, there is a lack of an adversarial audio dataset and a unified framework specifically designed to evaluate and compare attacks and ALMs. In this paper, we present JALMBench, the \textit{first} comprehensive benchmark to assess the safety of ALMs against jailbreak attacks. JALMBench includes a dataset containing 2,200 text samples and 51,381 audio samples with over 268 hours. It supports 12 mainstream ALMs, 4 text-transferred and 4 audio-originated attack methods, and 5 defense methods. Using JALMBench, we provide an in-depth analysis of attack efficiency, topic sensitivity, voice diversity, and attack representations. Additionally, we explore mitigation strategies for the attacks at both the prompt level and the response level.

摘要: 音频语言模型（ILM）最近取得了重大进展。这些模型将音频模式直接集成到模型中，而不是将语音转换为文本并将文本输入到大型语言模型（LLM）。虽然对LLM的越狱攻击已经得到了广泛的研究，但具有音频模式的ILM的安全性在很大程度上仍然没有被探索。目前，缺乏对抗性音频数据集和专门设计用于评估和比较攻击和ILM的统一框架。在本文中，我们介绍了JALMBench，这是一个\textit{first}综合基准，用于评估ILM针对越狱攻击的安全性。JALMBench包括一个包含2，200个文本样本和51，381个音频样本的数据集，时间超过268小时。它支持12种主流ILM、4种文本传输和4种音频源攻击方法以及5种防御方法。使用JALMBench，我们对攻击效率、主题敏感性、语音多样性和攻击表示进行深入分析。此外，我们还探索了即时级别和响应级别的攻击缓解策略。



## **16. FIT-Print: Towards False-claim-resistant Model Ownership Verification via Targeted Fingerprint**

FIT-Print：通过目标指纹实现抗虚假声明的模型所有权验证 cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2501.15509v2) [paper-pdf](http://arxiv.org/pdf/2501.15509v2)

**Authors**: Shuo Shao, Haozhe Zhu, Hongwei Yao, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Model fingerprinting is a widely adopted approach to safeguard the copyright of open-source models by detecting and preventing their unauthorized reuse without modifying the protected model. However, in this paper, we reveal that existing fingerprinting methods are vulnerable to false claim attacks where adversaries falsely assert ownership of third-party non-reused models. We find that this vulnerability mostly stems from their untargeted nature, where they generally compare the outputs of given samples on different models instead of the similarities to specific references. Motivated by this finding, we propose a targeted fingerprinting paradigm (i.e., FIT-Print) to counteract false claim attacks. Specifically, FIT-Print transforms the fingerprint into a targeted signature via optimization. Building on the principles of FIT-Print, we develop bit-wise and list-wise black-box model fingerprinting methods, i.e., FIT-ModelDiff and FIT-LIME, which exploit the distance between model outputs and the feature attribution of specific samples as the fingerprint, respectively. Experiments on benchmark models and datasets verify the effectiveness, conferrability, and resistance to false claim attacks of our FIT-Print.

摘要: 模型指纹识别是一种广泛采用的方法，通过在不修改受保护模型的情况下检测和防止其未经授权的重复使用来保护开源模型的版权。然而，在本文中，我们揭示了现有的指纹识别方法很容易受到虚假声明攻击，即对手错误地声称对第三方非重复使用模型的所有权。我们发现，这种漏洞主要源于其非目标性，即它们通常比较不同模型上给定样本的输出，而不是与特定参考文献的相似性。受这一发现的启发，我们提出了一种有针对性的指纹识别范式（即，FIT-Print）来对抗虚假索赔攻击。具体来说，FIT-Print通过优化将指纹转换为目标签名。基于FIT-Print的原则，我们开发了逐位和逐列表黑匣子模型指纹识别方法，即FIT-Model Diff和FIT-LIME，它们分别利用模型输出与特定样本的特征属性之间的距离作为指纹。对基准模型和数据集的实验验证了我们的FIT-Print的有效性、可协商性和对虚假声明攻击的抵抗力。



## **17. What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection**

读到的不是听到的：Deepfake语音检测中的语言敏感性 cs.LG

15 pages, 2 fogures

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17513v1) [paper-pdf](http://arxiv.org/pdf/2505.17513v1)

**Authors**: Binh Nguyen, Shuji Shi, Ryan Ofman, Thai Le

**Abstract**: Recent advances in text-to-speech technologies have enabled realistic voice generation, fueling audio-based deepfake attacks such as fraud and impersonation. While audio anti-spoofing systems are critical for detecting such threats, prior work has predominantly focused on acoustic-level perturbations, leaving the impact of linguistic variation largely unexplored. In this paper, we investigate the linguistic sensitivity of both open-source and commercial anti-spoofing detectors by introducing transcript-level adversarial attacks. Our extensive evaluation reveals that even minor linguistic perturbations can significantly degrade detection accuracy: attack success rates surpass 60% on several open-source detector-voice pairs, and notably one commercial detection accuracy drops from 100% on synthetic audio to just 32%. Through a comprehensive feature attribution analysis, we identify that both linguistic complexity and model-level audio embedding similarity contribute strongly to detector vulnerability. We further demonstrate the real-world risk via a case study replicating the Brad Pitt audio deepfake scam, using transcript adversarial attacks to completely bypass commercial detectors. These results highlight the need to move beyond purely acoustic defenses and account for linguistic variation in the design of robust anti-spoofing systems. All source code will be publicly available.

摘要: 文本转语音技术的最新进展使真实的语音生成成为可能，助长了欺诈和模仿等基于音频的深度伪造攻击。虽然音频反欺骗系统对于检测此类威胁至关重要，但之前的工作主要集中在声学层面的扰动上，语言变化的影响在很大程度上没有被探索。在本文中，我们通过引入转录层对抗攻击来研究开源和商业反欺骗检测器的语言敏感性。我们的广泛评估表明，即使是微小的语言扰动也会显着降低检测准确性：几个开源检测器-语音对的攻击成功率超过60%，值得注意的是，一个商业检测准确率从合成音频的100%下降到仅32%。通过全面的特征属性分析，我们发现语言复杂性和模型级音频嵌入相似性对检测器的脆弱性有很大影响。我们通过复制Brad Pitt音频深度伪造骗局的案例研究进一步证明了现实世界的风险，使用文字记录对抗攻击来完全绕过商业检测器。这些结果凸显了需要超越纯粹的声学防御，并在设计稳健的反欺骗系统时考虑语言差异。所有源代码都将公开。



## **18. Enhancing Adversarial Robustness of Vision Language Models via Adversarial Mixture Prompt Tuning**

通过对抗混合提示调优增强视觉语言模型的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17509v1) [paper-pdf](http://arxiv.org/pdf/2505.17509v1)

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Yize Fan, Ranjie Duan, Qing Guo, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) have excellent generalization capabilities but are highly susceptible to adversarial examples, presenting potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which finally leads to the overfitting phenomenon. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts can bring more robustness improvement than a longer prompt. Then we propose an adversarial tuning method named Adversarial Mixture Prompt Tuning (AMPT) to enhance the generalization towards various adversarial attacks for VLMs. AMPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the input adversarial image to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific aggregated text features aligning with different adversarial image features. A series of experiments show that our method can achieve better adversarial robustness than state-of-the-art methods on 11 datasets under different experimental settings.

摘要: 大型预先训练的视觉语言模型（VLM）具有出色的概括能力，但极易受到对抗性示例的影响，从而带来潜在的安全风险。为了提高VLM对对抗性示例的鲁棒性，提出了对抗性提示调整方法，以在不改变模型参数的情况下将文本特征与对抗性图像特征对齐。然而，当面临各种对抗性攻击时，单个可学习的文本提示的概括性不足以与所有对抗性图像特征很好地对齐，最终导致了过度匹配现象。为了应对上述挑战，在本文中，我们经验发现，增加学习提示的数量比更长的提示可以带来更多的鲁棒性改进。然后，我们提出了一种名为对抗混合提示调整（AMPT）的对抗性调整方法，以增强对VLM各种对抗性攻击的概括性。AMPT旨在学习混合文本提示以获得更稳健的文本特征。为了进一步增强适应性，我们提出了一种基于输入对抗图像的条件权重路由器来预测多个学习提示的混合权重，这有助于获得与不同对抗图像特征对齐的特定样本聚合文本特征。一系列实验表明，在不同实验设置下，我们的方法可以在11个数据集上实现比最先进的方法更好的对抗鲁棒性。



## **19. How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?**

城市环境中导航的大型语言模型（LLM）的安全性如何？ cs.RO

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2402.09546v2) [paper-pdf](http://arxiv.org/pdf/2402.09546v2)

**Authors**: Congcong Wen, Jiazhao Liang, Shuaihang Yuan, Hao Huang, Geeta Chandra Raju Bethala, Yu-Shen Liu, Mengyu Wang, Anthony Tzes, Yi Fang

**Abstract**: In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently demonstrated impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the widespread application of this technology in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Attack that manipulates LLM-based navigation models by perturbing the original navigational prompt, leading to incorrect actions. Based on the method of perturbation, our attacks are divided into two types: Navigational Prompt Insert (NPI) Attack and Navigational Prompt Swap (NPS) Attack. We conducted comprehensive experiments on an LLM-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across seven metrics in the face of both white-box and black-box attacks. Moreover, our attacks can be easily extended to other LLM-based navigation models with similarly effective results. These findings highlight the generalizability and transferability of the proposed attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, which concentrates on navigation-relevant keywords to reduce the impact of adversarial attacks. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.

摘要: 在机器人和自动化领域，基于大型语言模型（LLM）的导航系统最近表现出了令人印象深刻的性能。然而，这些系统的安全方面受到的关注相对较少。本文率先探索城市户外环境中基于LLM的导航模型中的漏洞，鉴于该技术在自动驾驶、物流和应急服务中的广泛应用，这是一个关键领域。具体来说，我们引入了一种新型的导航提示攻击，该攻击通过扰乱原始导航提示来操纵基于LLM的导航模型，从而导致错误的操作。根据扰动方法，我们的攻击分为两种类型：导航提示插入（NPI）攻击和导航提示交换（RST）攻击。我们对基于LLM的导航模型进行了全面的实验，该模型采用各种LLM进行推理。我们的结果来自少量学习和微调配置下的Touchdown和Map 2Seq街景数据集，表明面对白盒和黑匣子攻击，七个指标的性能均出现显着下降。此外，我们的攻击可以轻松扩展到其他基于LLM的导航模型，并获得类似有效的结果。这些发现强调了拟议攻击的普遍性和可转移性，强调了基于LLM的导航系统增强安全性的必要性。作为初步对策，我们提出了导航提示工程（NPE）防御策略，该策略专注于与导航相关的关键词，以减少对抗性攻击的影响。虽然初步研究结果表明该策略可以增强航行安全，但仍然迫切需要更广泛的研究界开发更强大的防御方法，以有效应对这些系统面临的现实挑战。



## **20. VEAttack: Downstream-agnostic Vision Encoder Attack against Large Vision Language Models**

VEAttack：针对大型视觉语言模型的下游不可知视觉编码器攻击 cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17440v1) [paper-pdf](http://arxiv.org/pdf/2505.17440v1)

**Authors**: Hefei Mei, Zirui Wang, Shen You, Minjing Dong, Chang Xu

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in multimodal understanding and generation, yet their vulnerability to adversarial attacks raises significant robustness concerns. While existing effective attacks always focus on task-specific white-box settings, these approaches are limited in the context of LVLMs, which are designed for diverse downstream tasks and require expensive full-model gradient computations. Motivated by the pivotal role and wide adoption of the vision encoder in LVLMs, we propose a simple yet effective Vision Encoder Attack (VEAttack), which targets the vision encoder of LVLMs only. Specifically, we propose to generate adversarial examples by minimizing the cosine similarity between the clean and perturbed visual features, without accessing the following large language models, task information, and labels. It significantly reduces the computational overhead while eliminating the task and label dependence of traditional white-box attacks in LVLMs. To make this simple attack effective, we propose to perturb images by optimizing image tokens instead of the classification token. We provide both empirical and theoretical evidence that VEAttack can easily generalize to various tasks. VEAttack has achieved a performance degradation of 94.5% on image caption task and 75.7% on visual question answering task. We also reveal some key observations to provide insights into LVLM attack/defense: 1) hidden layer variations of LLM, 2) token attention differential, 3) M\"obius band in transfer attack, 4) low sensitivity to attack steps. The code is available at https://github.com/hfmei/VEAttack-LVLM

摘要: 大型视觉语言模型（LVLM）在多模式理解和生成方面表现出了非凡的能力，但它们对对抗性攻击的脆弱性引发了严重的鲁棒性担忧。虽然现有的有效攻击始终集中在特定于任务的白盒设置上，但这些方法在LVLM的背景下受到限制，LVLM是为各种下游任务设计的，并且需要昂贵的全模型梯度计算。受视觉编码器在LVLM中的关键作用和广泛采用的激励，我们提出了一种简单而有效的视觉编码器攻击（VEAttack），该攻击仅针对LVLM的视觉编码器。具体来说，我们建议通过最小化干净和受干扰的视觉特征之间的cos相似性来生成对抗性示例，而无需访问以下大型语言模型、任务信息和标签。它显着减少了计算负担，同时消除了LVLM中传统白盒攻击的任务和标签依赖性。为了使这种简单的攻击有效，我们建议通过优化图像令牌而不是分类令牌来扰乱图像。我们提供了经验和理论证据，表明VEAttack可以轻松地推广到各种任务。VEAttack在图像字幕任务上的性能下降了94.5%，在视觉问答任务上的性能下降了75.7%。我们还揭示了一些关键观察结果，以提供对LVLM攻击/防御的见解：1）LLM的隐藏层变化，2）标记注意力差异，3）转移攻击中的M ' obius带，4）对攻击步骤的低敏感性。该代码可在https://github.com/hfmei/VEAttack-LVLM上获取



## **21. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

X-Transfer攻击：CLIP上的超级可转移对抗攻击 cs.CV

ICML 2025

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.05528v2) [paper-pdf](http://arxiv.org/pdf/2505.05528v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.

摘要: 随着对比图像预训练（CLIP）模型越来越多地被用于各种下游任务并集成到大型视觉语言模型（VLM）中，它们对对抗性扰动的敏感性已成为一个关键问题。在这项工作中，我们介绍了\textbf{X-Transfer}，一种新的攻击方法，暴露了CLIP中的一个普遍的对抗性漏洞。X-Transfer生成一个通用对抗扰动（Universal Adversarial Perturbation，UAP），能够欺骗不同样本、任务和域中的各种CLIP编码器和下游VLM。我们将此属性称为\textbf{super transferability}--一个同时实现跨数据、跨域、跨模型和跨任务对抗性可转移性的单一扰动。这是通过\textBF{代理缩放}来实现的，这是我们方法的一个关键创新。与依赖于固定代理模型（扩展计算密集型）的现有方法不同，X-Transfer采用高效的代理扩展策略，可以从大搜索空间中动态选择合适代理的一小子集。广泛的评估表明，X-Transfer的性能显着优于之前最先进的UAP方法，为跨CLIP模型的对抗性可移植性建立了新的基准。该代码可在我们的\href{https：//github.com/HanxunH/XTransferBench}{GitHub存储库}中公开获取。



## **22. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

StealthRank：通过Stealthy提示优化进行LLM排名操纵 cs.IR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2504.05804v2) [paper-pdf](http://arxiv.org/pdf/2504.05804v2)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present $\textbf{StealthRank}$, a novel adversarial attack method that manipulates LLM-driven ranking systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within item or document descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target items while avoiding explicit manipulation traces. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven ranking systems. Our code is publicly available at $\href{https://github.com/Tangyiming205069/controllable-seo}{here}$.

摘要: 将大型语言模型（LLM）集成到信息检索系统中引入了新的攻击表面，特别是对于对抗性排名操纵。我们提出了$\textBF{StealthRank}$，这是一种新型的对抗攻击方法，可以操纵LLM驱动的排名系统，同时保持文本流畅性和隐蔽性。与经常引入可检测异常的现有方法不同，StealthRank采用基于能量的优化框架与Langevin动态相结合来生成StealthRank脚本（SPP）-嵌入在项目或文档描述中的对抗性文本序列，微妙而有效地影响LLM排名机制。我们在多个LLM中评估StealthRank，证明其能够秘密提高目标项目的排名，同时避免显式操纵痕迹。我们的结果表明，StealthRank在有效性和隐蔽性方面始终优于最先进的对抗排名基线，凸显了LLM驱动的排名系统中的关键漏洞。我们的代码可在$\href{https：//github.com/Tangyiming205069/guardable-seo}{here}$公开。



## **23. Defending Multimodal Backdoored Models by Repulsive Visual Prompt Tuning**

通过排斥性视觉提示调整捍卫多模式后门模型 cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2412.20392v3) [paper-pdf](http://arxiv.org/pdf/2412.20392v3)

**Authors**: Zhifang Zhang, Shuo He, Haobo Wang, Bingquan Shen, Lei Feng

**Abstract**: Multimodal contrastive learning models (e.g., CLIP) can learn high-quality representations from large-scale image-text datasets, while they exhibit significant vulnerabilities to backdoor attacks, raising serious safety concerns. In this paper, we reveal that CLIP's vulnerabilities primarily stem from its tendency to encode features beyond in-dataset predictive patterns, compromising its visual feature resistivity to input perturbations. This makes its encoded features highly susceptible to being reshaped by backdoor triggers. To address this challenge, we propose Repulsive Visual Prompt Tuning (RVPT), a novel defense approach that employs deep visual prompt tuning with a specially designed feature-repelling loss. Specifically, RVPT adversarially repels the encoded features from deeper layers while optimizing the standard cross-entropy loss, ensuring that only predictive features in downstream tasks are encoded, thereby enhancing CLIP's visual feature resistivity against input perturbations and mitigating its susceptibility to backdoor attacks. Unlike existing multimodal backdoor defense methods that typically require the availability of poisoned data or involve fine-tuning the entire model, RVPT leverages few-shot downstream clean samples and only tunes a small number of parameters. Empirical results demonstrate that RVPT tunes only 0.27\% of the parameters in CLIP, yet it significantly outperforms state-of-the-art defense methods, reducing the attack success rate from 89.70\% to 2.76\% against the most advanced multimodal attacks on ImageNet and effectively generalizes its defensive capabilities across multiple datasets.

摘要: 多模式对比学习模型（例如，CLIP）可以从大规模图像文本数据集中学习高质量的表示，同时它们对后门攻击表现出显着的漏洞，从而引发了严重的安全问题。在本文中，我们揭示了CLIP的漏洞主要源于它倾向于编码数据集内预测模式之外的特征，从而损害了其视觉特征对输入扰动的抵抗力。这使得其编码功能极易被后门触发器重塑。为了应对这一挑战，我们提出了排斥视觉提示调整（RVPT），一种新的防御方法，采用深度视觉提示调整与专门设计的功能排斥损失。具体而言，RVPT对抗性地排斥来自更深层的编码特征，同时优化标准交叉熵损失，确保仅编码下游任务中的预测特征，从而增强CLIP对输入扰动的视觉特征抵抗力并减轻其对后门攻击的敏感性。与现有的多模态后门防御方法不同，这些方法通常需要中毒数据的可用性或涉及微调整个模型，RVPT利用少量的下游干净样本，仅调整少量参数。实验结果表明，RVPT只调整了CLIP中0.27%的参数，但它明显优于最先进的防御方法，将ImageNet上最先进的多模态攻击的攻击成功率从89.70%降低到2.76%，并有效地将其防御能力推广到多个数据集。



## **24. Secure and Private Federated Learning: Achieving Adversarial Resilience through Robust Aggregation**

安全且私人的联邦学习：通过稳健的聚合实现对抗弹性 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17226v1) [paper-pdf](http://arxiv.org/pdf/2505.17226v1)

**Authors**: Kun Yang, Neena Imam

**Abstract**: Federated Learning (FL) enables collaborative machine learning across decentralized data sources without sharing raw data. It offers a promising approach to privacy-preserving AI. However, FL remains vulnerable to adversarial threats from malicious participants, referred to as Byzantine clients, who can send misleading updates to corrupt the global model. Traditional aggregation methods, such as simple averaging, are not robust to such attacks. More resilient approaches, like the Krum algorithm, require prior knowledge of the number of malicious clients, which is often unavailable in real-world scenarios. To address these limitations, we propose Average-rKrum (ArKrum), a novel aggregation strategy designed to enhance both the resilience and privacy guarantees of FL systems. Building on our previous work (rKrum), ArKrum introduces two key innovations. First, it includes a median-based filtering mechanism that removes extreme outliers before estimating the number of adversarial clients. Second, it applies a multi-update averaging scheme to improve stability and performance, particularly when client data distributions are not identical. We evaluate ArKrum on benchmark image and text datasets under three widely studied Byzantine attack types. Results show that ArKrum consistently achieves high accuracy and stability. It performs as well as or better than other robust aggregation methods. These findings demonstrate that ArKrum is an effective and practical solution for secure FL systems in adversarial environments.

摘要: 联合学习（FL）支持跨去中心化数据源的协作机器学习，而无需共享原始数据。它为保护隐私的人工智能提供了一种有希望的方法。然而，FL仍然容易受到来自恶意参与者（称为拜占庭客户）的敌对威胁，这些参与者可以发送误导性更新以破坏全球模型。传统的聚合方法（例如简单平均）对此类攻击并不稳健。更具弹性的方法，例如Krum算法，需要事先了解恶意客户端的数量，而这在现实世界场景中通常是不可用的。为了解决这些限制，我们提出了Average-rKrum（ArKrum），这是一种新型聚合策略，旨在增强FL系统的弹性和隐私保证。在我们之前的工作（rKrum）的基础上，ArKrum引入了两项关键创新。首先，它包括一个基于媒体的过滤机制，可以在估计对抗客户端的数量之前删除极端异常值。其次，它应用多更新平均方案来提高稳定性和性能，特别是当客户端数据分布不相同时。我们在三种广泛研究的拜占庭攻击类型下，在基准图像和文本数据集上评估ArKrum。结果表明ArKrum始终实现高准确性和稳定性。它的性能与其他稳健聚合方法一样好或更好。这些发现表明，ArKrum是对抗环境中安全FL系统的有效且实用的解决方案。



## **25. Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning**

数据集属性对深度迁移学习的成员推断漏洞的影响 cs.CR

43 pages, 13 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2402.06674v4) [paper-pdf](http://arxiv.org/pdf/2402.06674v4)

**Authors**: Marlon Tobaben, Hibiki Ito, Joonas Jälkö, Yuan He, Antti Honkela

**Abstract**: Membership inference attacks (MIAs) are used to test practical privacy of machine learning models. MIAs complement formal guarantees from differential privacy (DP) under a more realistic adversary model. We analyse MIA vulnerability of fine-tuned neural networks both empirically and theoretically, the latter using a simplified model of fine-tuning. We show that the vulnerability of non-DP models when measured as the attacker advantage at fixed false positive rate reduces according to a simple power law as the number of examples per class increases, even for the most vulnerable points, but the dataset size needed for adequate protection of the most vulnerable points is very large.

摘要: 成员推理攻击（MIA）用于测试机器学习模型的实际隐私。MIA补充正式保证差分隐私（DP）下一个更现实的对手模型。我们分析MIA脆弱性微调神经网络的经验和理论，后者使用一个简化的模型微调。我们发现，非DP模型的脆弱性，当测量为攻击者的优势，在固定的误报率根据一个简单的幂律减少，每个类的例子的数量增加，即使是最脆弱的点，但数据集的大小需要充分保护的最脆弱的点是非常大的。



## **26. Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms**

热带注意力：组合算法的神经数学推理 cs.LG

Under Review

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17190v1) [paper-pdf](http://arxiv.org/pdf/2505.17190v1)

**Authors**: Baran Hashemi, Kurt Pasque, Chris Teska, Ruriko Yoshida

**Abstract**: Dynamic programming (DP) algorithms for combinatorial optimization problems work with taking maximization, minimization, and classical addition in their recursion algorithms. The associated value functions correspond to convex polyhedra in the max plus semiring. Existing Neural Algorithmic Reasoning models, however, rely on softmax-normalized dot-product attention where the smooth exponential weighting blurs these sharp polyhedral structures and collapses when evaluated on out-of-distribution (OOD) settings. We introduce Tropical attention, a novel attention function that operates natively in the max-plus semiring of tropical geometry. We prove that Tropical attention can approximate tropical circuits of DP-type combinatorial algorithms. We then propose that using Tropical transformers enhances empirical OOD performance in both length generalization and value generalization, on algorithmic reasoning tasks, surpassing softmax baselines while remaining stable under adversarial attacks. We also present adversarial-attack generalization as a third axis for Neural Algorithmic Reasoning benchmarking. Our results demonstrate that Tropical attention restores the sharp, scale-invariant reasoning absent from softmax.

摘要: 组合优化问题的动态规划（DP）算法在其递归算法中采用最大化、最小化和经典加法。相关的值函数对应于最大加半环中的凸多面体。然而，现有的神经网络推理模型依赖于softmax归一化的点积注意力，其中平滑的指数加权模糊了这些尖锐的多面体结构，并在分布外（OOD）设置上进行评估时崩溃。我们引入热带的注意力，一个新的注意力函数，本机在热带几何的最大加半环。我们证明热带注意力可以逼近DP型组合算法的热带回路。然后，我们建议使用热带变换器可以在算法推理任务中增强长度概括和值概括方面的经验OOD性能，超越softmax基线，同时在对抗性攻击下保持稳定。我们还将对抗攻击概括作为神经数学推理基准的第三个轴。我们的结果表明，热带注意力恢复了softmax中缺失的尖锐、规模不变的推理。



## **27. When Are Concepts Erased From Diffusion Models?**

概念何时从扩散模型中删除？ cs.LG

Project Page:  https://nyu-dice-lab.github.io/when-are-concepts-erased/

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17013v1) [paper-pdf](http://arxiv.org/pdf/2505.17013v1)

**Authors**: Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, Niv Cohen

**Abstract**: Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.

摘要: 概念擦除，即选择性地阻止模型生成特定概念的能力，引起了越来越多的兴趣，各种方法出现了来应对这一挑战。然而，目前尚不清楚这些方法如何彻底消除目标概念。我们首先提出了扩散模型中擦除机制的两个概念模型：（i）降低生成目标概念的可能性，（ii）干扰模型的内部引导机制。为了彻底评估某个概念是否已真正从模型中删除，我们引入了一套独立评估。我们的评估框架包括对抗性攻击、新颖的探测技术以及对模型替代世代的分析，以取代被删除的概念。我们的结果揭示了最大限度地减少副作用和保持对抗提示的鲁棒性之间的紧张关系。从广义上讲，我们的工作强调了对扩散模型中擦除进行全面评估的重要性。



## **28. Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability**

利用ViT中的计算冗余来提高对抗性可移植性 cs.CV

15 pages. 7 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2504.10804v2) [paper-pdf](http://arxiv.org/pdf/2504.10804v2)

**Authors**: Jiani Liu, Zhiyuan Wang, Zeliang Zhang, Chao Huang, Susan Liang, Yunlong Tang, Chenliang Xu

**Abstract**: Vision Transformers (ViTs) have demonstrated impressive performance across a range of applications, including many safety-critical tasks. However, their unique architectural properties raise new challenges and opportunities in adversarial robustness. In particular, we observe that adversarial examples crafted on ViTs exhibit higher transferability compared to those crafted on CNNs, suggesting that ViTs contain structural characteristics favorable for transferable attacks. In this work, we investigate the role of computational redundancy in ViTs and its impact on adversarial transferability. Unlike prior studies that aim to reduce computation for efficiency, we propose to exploit this redundancy to improve the quality and transferability of adversarial examples. Through a detailed analysis, we identify two forms of redundancy, including the data-level and model-level, that can be harnessed to amplify attack effectiveness. Building on this insight, we design a suite of techniques, including attention sparsity manipulation, attention head permutation, clean token regularization, ghost MoE diversification, and test-time adversarial training. Extensive experiments on the ImageNet-1k dataset validate the effectiveness of our approach, showing that our methods significantly outperform existing baselines in both transferability and generality across diverse model architectures.

摘要: Vision Transformers（ViT）在一系列应用中表现出令人印象深刻的性能，包括许多安全关键任务。然而，它们独特的架构属性在对抗稳健性方面提出了新的挑战和机遇。特别是，我们观察到，与CNN上制作的对抗示例相比，在ViT上制作的对抗示例表现出更高的可转移性，这表明ViT包含有利于转移攻击的结构特征。在这项工作中，我们研究了计算冗余在ViT中的作用及其对对抗可转移性的影响。与之前旨在减少计算以提高效率的研究不同，我们建议利用这种冗余来提高对抗性示例的质量和可移植性。通过详细的分析，我们确定了两种形式的冗余，包括数据级和模型级，可以利用它们来放大攻击有效性。基于这一见解，我们设计了一套技术，包括注意力稀疏性操纵、注意力头排列、干净令牌正规化、幽灵MoE多样化和测试时对抗训练。ImageNet-1 k数据集上的大量实验验证了我们方法的有效性，表明我们的方法在跨不同模型架构的可移植性和通用性方面都显着优于现有基线。



## **29. Invisible Prompts, Visible Threats: Malicious Font Injection in External Resources for Large Language Models**

看不见的警告，可见的威胁：大型语言模型的外部资源中的恶意字体注入 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16957v1) [paper-pdf](http://arxiv.org/pdf/2505.16957v1)

**Authors**: Junjie Xiong, Changjia Zhu, Shuhang Lin, Chong Zhang, Yongfeng Zhang, Yao Liu, Lingyao Li

**Abstract**: Large Language Models (LLMs) are increasingly equipped with capabilities of real-time web search and integrated with protocols like Model Context Protocol (MCP). This extension could introduce new security vulnerabilities. We present a systematic investigation of LLM vulnerabilities to hidden adversarial prompts through malicious font injection in external resources like webpages, where attackers manipulate code-to-glyph mapping to inject deceptive content which are invisible to users. We evaluate two critical attack scenarios: (1) "malicious content relay" and (2) "sensitive data leakage" through MCP-enabled tools. Our experiments reveal that indirect prompts with injected malicious font can bypass LLM safety mechanisms through external resources, achieving varying success rates based on data sensitivity and prompt design. Our research underscores the urgent need for enhanced security measures in LLM deployments when processing external content.

摘要: 大型语言模型（LLM）越来越多地配备实时网络搜索功能，并与模型上下文协议（HCP）等协议集成。此扩展可能会引入新的安全漏洞。我们对通过在网页等外部资源中恶意字体注入来隐藏对抗提示的LLM漏洞进行了系统性调查，其中攻击者操纵代码到收件箱的映射来注入用户不可见的欺骗性内容。我们评估了两种关键攻击场景：（1）“恶意内容中继”和（2）通过支持MVP的工具“敏感数据泄露”。我们的实验表明，注入恶意字体的间接提示可以通过外部资源绕过LLM安全机制，根据数据敏感性和提示设计实现不同的成功率。我们的研究强调了处理外部内容时LLM部署中迫切需要增强的安全措施。



## **30. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

MixAT：结合LLM的连续和离散对抗训练 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16947v1) [paper-pdf](http://arxiv.org/pdf/2505.16947v1)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.

摘要: 尽管最近在大型语言模型（LLM）的安全性和一致性方面做出了努力，但当前对前沿LLM的对抗性攻击仍然能够持续地迫使有害的世代。尽管对抗训练已得到广泛研究，并被证明可以显着提高传统机器学习模型的鲁棒性，但其在LLM背景下的优点和缺点却知之甚少。具体来说，虽然现有的离散对抗攻击可以有效地产生有害内容，但用具体的对抗提示训练LLM通常计算成本高昂，导致依赖于持续的放松。由于这些松弛不对应于离散输入令牌，因此此类潜在训练方法通常使模型容易受到一系列不同的离散攻击。在这项工作中，我们的目标是通过引入MixAT来弥合这一差距，MixAT是一种新颖的方法，在训练期间结合了更强的离散攻击和更快的连续攻击。我们对MixAT进行了广泛的最先进攻击，提出了至少一次攻击成功率（ALO-ASB）指标来捕捉模型的最坏情况漏洞。我们表明，与之前的防御（ALO-ASB> 50%）相比，MixAT实现了更好的鲁棒性（ALO-ASB < 20%），同时保持与基于连续松弛的方法相当的运行时间。我们进一步分析了现实部署环境中的MixAT，探索聊天模板、量化、低等级适配器和温度如何影响对抗训练和评估，从而揭示了当前方法中的其他盲点。我们的结果表明，MixAT的离散-连续防御以最小的计算负担提供了原则性且卓越的鲁棒性-准确性权衡，凸显了其构建更安全的LLM的承诺。我们在https://github.com/insait-institute/MixAT上提供我们的代码和模型。



## **31. CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework**

CAIN：通过两阶段恶意系统提示生成和精炼框架劫持LLM与人类对话 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16888v1) [paper-pdf](http://arxiv.org/pdf/2505.16888v1)

**Authors**: Viet Pham, Thai Le

**Abstract**: Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.

摘要: 大型语言模型（LLM）先进了许多应用程序，但也容易受到对抗攻击。在这项工作中，我们引入了一种新颖的安全威胁：通过操纵LLM的系统提示来劫持人工智能与人类的对话，以仅对特定目标问题（例如，“我应该投票给谁美国总统？”，“新冠疫苗安全吗？”），同时对他人表现友善。这种攻击是有害的，因为它可以使恶意行为者通过在线传播有害但看起来友善的系统提示来进行大规模信息操纵。为了演示此类攻击，我们开发了CAIN，这是一种算法，可以在黑匣子设置中或无需访问LLM参数的情况下自动策划此类有害系统提示特定目标问题。在开源和商业LLM上进行评估，CAIN表现出显着的对抗影响。在无针对性攻击或迫使LLM输出错误答案中，CAIN对目标问题实现了高达40%的F1降级，同时对良性输入保持高准确性。对于有针对性的攻击或迫使LLM输出特定的有害答案，CAIN在这些有针对性的回答上获得了超过70%的F1分数，而对良性问题的影响最小。我们的结果凸显了对增强稳健性措施的迫切需要，以保障LLM在现实世界应用中的完整性和安全性。所有源代码都将公开。



## **32. Safe RLHF-V: Safe Reinforcement Learning from Multi-modal Human Feedback**

Safe RLHF-V：基于多模态人类反馈的安全强化学习 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2503.17682v2) [paper-pdf](http://arxiv.org/pdf/2503.17682v2)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Conghui Zhang, Han Zhu, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Yida Tang, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are essential for building general-purpose AI assistants; however, they pose increasing safety risks. How can we ensure safety alignment of MLLMs to prevent undesired behaviors? Going further, it is critical to explore how to fine-tune MLLMs to preserve capabilities while meeting safety constraints. Fundamentally, this challenge can be formulated as a min-max optimization problem. However, existing datasets have not yet disentangled single preference signals into explicit safety constraints, hindering systematic investigation in this direction. Moreover, it remains an open question whether such constraints can be effectively incorporated into the optimization process for multi-modal models. In this work, we present the first exploration of the Safe RLHF-V -- the first multimodal safety alignment framework. The framework consists of: $\mathbf{(I)}$ BeaverTails-V, the first open-source dataset featuring dual preference annotations for helpfulness and safety, supplemented with multi-level safety labels (minor, moderate, severe); $\mathbf{(II)}$ Beaver-Guard-V, a multi-level guardrail system to proactively defend against unsafe queries and adversarial attacks. Applying the guard model over five rounds of filtering and regeneration significantly enhances the precursor model's overall safety by an average of 40.9%. $\mathbf{(III)}$ Based on dual preference, we initiate the first exploration of multi-modal safety alignment within a constrained optimization. Experimental results demonstrate that Safe RLHF effectively improves both model helpfulness and safety. Specifically, Safe RLHF-V enhances model safety by 34.2% and helpfulness by 34.3%.

摘要: 多模式大型语言模型（MLLM）对于构建通用人工智能助手至关重要;然而，它们带来了越来越大的安全风险。我们如何确保MLLM的安全一致以防止不良行为？进一步说，探索如何微调MLLM以在满足安全限制的同时保留功能至关重要。从根本上讲，这个挑战可以被描述为一个最小-最大优化问题。然而，现有的数据集尚未将单一偏好信号分解为明确的安全约束，从而阻碍了这方面的系统性研究。此外，这些约束是否可以有效地纳入多模式模型的优化过程仍然是一个悬而未决的问题。在这项工作中，我们首次探索Safe RLHF-V --第一个多模式安全对齐框架。该框架包括：$\mathBF{（I）}$ BeaverTails-V，第一个开源数据集，具有帮助性和安全性的双重偏好注释，并辅之以多级别安全标签（轻微、中度、严重）; $\mathBF{（II）}$ Beaver-Guard-V，一个多级别护栏系统，用于主动防御不安全的查询和对抗性攻击。经过五轮过滤和再生应用防护模型，前体模型的整体安全性平均显着提高了40.9%。$\mathBF{（III）}$基于双重偏好，我们在约束优化中启动了多模式安全对齐的首次探索。实验结果表明，Safe RL HF有效提高了模型的帮助性和安全性。具体而言，Safe RLHF-V将模型的安全性提高了34.2%，帮助性提高了34.3%。



## **33. Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability**

意外失调：微调语言模型会引发意外漏洞 cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16789v1) [paper-pdf](http://arxiv.org/pdf/2505.16789v1)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models gain popularity, their vulnerability to adversarial attacks remains a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Misalignment, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity within our experimental datasets. We then evaluate the adversarial performance of these fine-tuned models and assess how dataset factors correlate with attack success rates. Lastly, we explore potential causal links, offering new insights into adversarial defense strategies and highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_misalignment.

摘要: 随着大型语言模型越来越受欢迎，它们对对抗攻击的脆弱性仍然是一个主要问题。虽然通常使用特定领域数据集的微调模型来提高模型性能，但它可能会在基础模型中引入漏洞。在这项工作中，我们调查了意外失准，即微调数据特征引起的意外漏洞。我们首先确定潜在的相关因素，如语言特征，语义相似性和毒性在我们的实验数据集。然后，我们评估这些微调模型的对抗性能，并评估数据集因素与攻击成功率的相关性。最后，我们探索了潜在的因果关系，为对抗性防御策略提供了新的见解，并强调了数据集设计在保持模型对齐方面的关键作用。我们的代码可在https://github.com/psyonp/accidental_misalignment上获取。



## **34. Experimental robustness benchmark of quantum neural network on a superconducting quantum processor**

超量子处理器上量子神经网络实验鲁棒性基准 quant-ph

There are 8 pages with 5 figures in the main text and 15 pages with  14 figures in the supplementary information

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16714v1) [paper-pdf](http://arxiv.org/pdf/2505.16714v1)

**Authors**: Hai-Feng Zhang, Zhao-Yun Chen, Peng Wang, Liang-Liang Guo, Tian-Le Wang, Xiao-Yan Yang, Ren-Ze Zhao, Ze-An Zhao, Sheng Zhang, Lei Du, Hao-Ran Tao, Zhi-Long Jia, Wei-Cheng Kong, Huan-Yu Liu, Athanasios V. Vasilakos, Yang Yang, Yu-Chun Wu, Ji Guan, Peng Duan, Guo-Ping Guo

**Abstract**: Quantum machine learning (QML) models, like their classical counterparts, are vulnerable to adversarial attacks, hindering their secure deployment. Here, we report the first systematic experimental robustness benchmark for 20-qubit quantum neural network (QNN) classifiers executed on a superconducting processor. Our benchmarking framework features an efficient adversarial attack algorithm designed for QNNs, enabling quantitative characterization of adversarial robustness and robustness bounds. From our analysis, we verify that adversarial training reduces sensitivity to targeted perturbations by regularizing input gradients, significantly enhancing QNN's robustness. Additionally, our analysis reveals that QNNs exhibit superior adversarial robustness compared to classical neural networks, an advantage attributed to inherent quantum noise. Furthermore, the empirical upper bound extracted from our attack experiments shows a minimal deviation ($3 \times 10^{-3}$) from the theoretical lower bound, providing strong experimental confirmation of the attack's effectiveness and the tightness of fidelity-based robustness bounds. This work establishes a critical experimental framework for assessing and improving quantum adversarial robustness, paving the way for secure and reliable QML applications.

摘要: 量子机器学习（QML）模型与经典模型一样，容易受到对抗攻击，从而阻碍其安全部署。在这里，我们报告了在高温处理器上执行的20量子位量子神经网络（QNN）分类器的第一个系统实验鲁棒性基准。我们的基准测试框架具有为QNN设计的高效对抗攻击算法，能够量化描述对抗稳健性和稳健性界限。根据我们的分析，我们验证了对抗训练通过规范化输入梯度来降低对目标扰动的敏感性，从而显着增强了QNN的鲁棒性。此外，我们的分析表明，与经典神经网络相比，QNN表现出更好的对抗鲁棒性，这一优势归因于固有的量子噪音。此外，从我们的攻击实验中提取的经验上限显示出与理论下限的最小偏差（$3 \x 10 '' s），为攻击的有效性和基于一致性的鲁棒性界限的严格性提供了强有力的实验证实。这项工作建立了一个关键的实验框架，用于评估和改进量子对抗鲁棒性，为安全可靠的QML应用铺平了道路。



## **35. BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization**

BadVLA：通过解耦优化实现对视觉-语言-动作模型的后门攻击 cs.CR

19 pages, 12 figures, 6 tables

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16640v1) [paper-pdf](http://arxiv.org/pdf/2505.16640v1)

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Hechang Wang, Pan Zhou, Lichao Sun

**Abstract**: Vision-Language-Action (VLA) models have advanced robotic control by enabling end-to-end decision-making directly from multimodal inputs. However, their tightly coupled architectures expose novel security vulnerabilities. Unlike traditional adversarial perturbations, backdoor attacks represent a stealthier, persistent, and practically significant threat-particularly under the emerging Training-as-a-Service paradigm-but remain largely unexplored in the context of VLA models. To address this gap, we propose BadVLA, a backdoor attack method based on Objective-Decoupled Optimization, which for the first time exposes the backdoor vulnerabilities of VLA models. Specifically, it consists of a two-stage process: (1) explicit feature-space separation to isolate trigger representations from benign inputs, and (2) conditional control deviations that activate only in the presence of the trigger, while preserving clean-task performance. Empirical results on multiple VLA benchmarks demonstrate that BadVLA consistently achieves near-100% attack success rates with minimal impact on clean task accuracy. Further analyses confirm its robustness against common input perturbations, task transfers, and model fine-tuning, underscoring critical security vulnerabilities in current VLA deployments. Our work offers the first systematic investigation of backdoor vulnerabilities in VLA models, highlighting an urgent need for secure and trustworthy embodied model design practices. We have released the project page at https://badvla-project.github.io/.

摘要: 视觉-语言-动作（VLA）模型通过直接从多模式输入进行端到端决策，实现了先进的机器人控制。然而，它们的紧密耦合架构暴露了新型安全漏洞。与传统的对抗性扰动不同，后门攻击代表了一种更隐蔽、持久且实际上重大的威胁--特别是在新兴的“服务培训”范式下--但在VLA模型的背景下，它在很大程度上尚未被探索。为了弥补这一差距，我们提出了BadVLA，这是一种基于Inbox-Decoupled优化的后门攻击方法，首次暴露了VLA模型的后门漏洞。具体来说，它由两阶段过程组成：（1）显式特征空间分离，以将触发器表示与良性输入隔离，以及（2）仅在触发器存在时激活的条件控制偏差，同时保持干净任务性能。多个VLA基准的经验结果表明，BadVLA始终实现接近100%的攻击成功率，对干净任务准确性的影响最小。进一步的分析证实了它对常见输入扰动、任务传输和模型微调的稳健性，凸显了当前VLA部署中的关键安全漏洞。我们的工作首次对VLA模型中的后门漏洞进行了系统性调查，凸显了对安全且值得信赖的具体模型设计实践的迫切需求。我们已在https://badvla-project.github.io/上发布了项目页面。



## **36. On the Lack of Robustness of Binary Function Similarity Systems**

论二元函数相似系统的鲁棒性缺乏 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.04163v2) [paper-pdf](http://arxiv.org/pdf/2412.04163v2)

**Authors**: Gianluca Capozzi, Tong Tang, Jie Wan, Ziqi Yang, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Lorenzo Cavallaro, Leonardo Querzoni

**Abstract**: Binary function similarity, which often relies on learning-based algorithms to identify what functions in a pool are most similar to a given query function, is a sought-after topic in different communities, including machine learning, software engineering, and security. Its importance stems from the impact it has in facilitating several crucial tasks, from reverse engineering and malware analysis to automated vulnerability detection. Whereas recent work cast light around performance on this long-studied problem, the research landscape remains largely lackluster in understanding the resiliency of the state-of-the-art machine learning models against adversarial attacks. As security requires to reason about adversaries, in this work we assess the robustness of such models through a simple yet effective black-box greedy attack, which modifies the topology and the content of the control flow of the attacked functions. We demonstrate that this attack is successful in compromising all the models, achieving average attack success rates of 57.06% and 95.81% depending on the problem settings (targeted and untargeted attacks). Our findings are insightful: top performance on clean data does not necessarily relate to top robustness properties, which explicitly highlights performance-robustness trade-offs one should consider when deploying such models, calling for further research.

摘要: 二进制函数相似性通常依赖于基于学习的算法来识别池中哪些函数与给定查询函数最相似，是不同社区（包括机器学习、软件工程和安全）中备受追捧的话题。它的重要性源于它对促进从反向工程和恶意软件分析到自动漏洞检测等几项关键任务的影响。尽管最近的工作揭示了这个长期研究问题的性能，但研究领域在了解最先进的机器学习模型对抗对抗攻击的弹性方面仍然缺乏活力。由于安全需要对对手进行推理，在这项工作中，我们通过简单而有效的黑匣子贪婪攻击来评估此类模型的稳健性，该攻击修改了被攻击功能的布局和控制流的内容。我们证明，这种攻击成功地破坏了所有模型，根据问题设置（有针对性和无针对性的攻击），平均攻击成功率分别为57.06%和95.81%。我们的发现很有洞察力：干净数据上的顶级性能不一定与顶级稳健性属性相关，这明确强调了在部署此类模型时应该考虑的性能稳健性权衡，需要进一步研究。



## **37. Implicit Jailbreak Attacks via Cross-Modal Information Concealment on Vision-Language Models**

通过视觉语言模型的跨模式信息隐藏进行隐性越狱攻击 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16446v1) [paper-pdf](http://arxiv.org/pdf/2505.16446v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities. However, the expanded input space introduces new attack surfaces. Previous jailbreak attacks often inject malicious instructions from text into less aligned modalities, such as vision. As MLLMs increasingly incorporate cross-modal consistency and alignment mechanisms, such explicit attacks become easier to detect and block. In this work, we propose a novel implicit jailbreak framework termed IJA that stealthily embeds malicious instructions into images via least significant bit steganography and couples them with seemingly benign, image-related textual prompts. To further enhance attack effectiveness across diverse MLLMs, we incorporate adversarial suffixes generated by a surrogate model and introduce a template optimization module that iteratively refines both the prompt and embedding based on model feedback. On commercial models like GPT-4o and Gemini-1.5 Pro, our method achieves attack success rates of over 90% using an average of only 3 queries.

摘要: 多模式大型语言模型（MLLM）实现强大的跨模式推理能力。然而，扩展的输入空间引入了新的攻击面。之前的越狱攻击经常将文本中的恶意指令注入到不一致的模式中，例如视觉。随着MLLM越来越多地结合跨模式一致性和对齐机制，此类显式攻击变得更容易检测和阻止。在这项工作中，我们提出了一种名为IJA的新型隐式越狱框架，该框架通过最低有效位隐写术将恶意指令秘密地嵌入到图像中，并将其与看似良性的图像相关文本提示相结合。为了进一步增强不同MLLM之间的攻击有效性，我们结合了代理模型生成的对抗性后缀，并引入了模板优化模块，该模块根据模型反馈迭代地细化提示和嵌入。在GPT-4 o和Gemini-1.5 Pro等商业型号上，我们的方法平均只需3个查询即可实现超过90%的攻击成功率。



## **38. AdvReal: Adversarial Patch Generation Framework with Application to Adversarial Safety Evaluation of Object Detection Systems**

AdvReal：对抗性补丁生成框架，应用于对象检测系统的对抗性安全评估 cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16402v1) [paper-pdf](http://arxiv.org/pdf/2505.16402v1)

**Authors**: Yuanhao Huang, Yilong Ren, Jinlei Wang, Lujia Huo, Xuesong Bai, Jinchuan Zhang, Haiyan Yu

**Abstract**: Autonomous vehicles are typical complex intelligent systems with artificial intelligence at their core. However, perception methods based on deep learning are extremely vulnerable to adversarial samples, resulting in safety accidents. How to generate effective adversarial examples in the physical world and evaluate object detection systems is a huge challenge. In this study, we propose a unified joint adversarial training framework for both 2D and 3D samples to address the challenges of intra-class diversity and environmental variations in real-world scenarios. Building upon this framework, we introduce an adversarial sample reality enhancement approach that incorporates non-rigid surface modeling and a realistic 3D matching mechanism. We compare with 5 advanced adversarial patches and evaluate their attack performance on 8 object detecotrs, including single-stage, two-stage, and transformer-based models. Extensive experiment results in digital and physical environments demonstrate that the adversarial textures generated by our method can effectively mislead the target detection model. Moreover, proposed method demonstrates excellent robustness and transferability under multi-angle attacks, varying lighting conditions, and different distance in the physical world. The demo video and code can be obtained at https://github.com/Huangyh98/AdvReal.git.

摘要: 自动驾驶汽车是典型的复杂智能系统，以人工智能为核心。然而，基于深度学习的感知方法极易受到对抗样本的影响，从而导致安全事故。如何在物理世界中生成有效的对抗示例并评估对象检测系统是一个巨大的挑战。在这项研究中，我们提出了一个针对2D和3D样本的统一联合对抗训练框架，以应对现实世界场景中班级内多样性和环境变化的挑战。在此框架的基础上，我们引入了一种对抗性样本现实增强方法，该方法结合了非刚性表面建模和真实的3D匹配机制。我们与5个高级对抗补丁进行了比较，并评估了它们对8个对象Detecotr的攻击性能，包括单阶段、两阶段和基于变换器的模型。数字和物理环境中的大量实验结果表明，我们的方法生成的对抗纹理可以有效地误导目标检测模型。此外，所提出的方法在多角度攻击、变化的光照条件和物理世界中不同距离下表现出出色的鲁棒性和可移植性。演示视频和代码可在https://github.com/Huangyh98/AdvReal.git上获取。



## **39. MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming**

MTSA：通过多轮红队进行LLM的多圈安全对准 cs.CR

19 pages,6 figures,ACL2025

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17147v1) [paper-pdf](http://arxiv.org/pdf/2505.17147v1)

**Authors**: Weiyang Guo, Jing Li, Wenya Wang, YU LI, Daojing He, Jun Yu, Min Zhang

**Abstract**: The proliferation of jailbreak attacks against large language models (LLMs) highlights the need for robust security measures. However, in multi-round dialogues, malicious intentions may be hidden in interactions, leading LLMs to be more prone to produce harmful responses. In this paper, we propose the \textbf{M}ulti-\textbf{T}urn \textbf{S}afety \textbf{A}lignment (\ourapproach) framework, to address the challenge of securing LLMs in multi-round interactions. It consists of two stages: In the thought-guided attack learning stage, the red-team model learns about thought-guided multi-round jailbreak attacks to generate adversarial prompts. In the adversarial iterative optimization stage, the red-team model and the target model continuously improve their respective capabilities in interaction. Furthermore, we introduce a multi-turn reinforcement learning algorithm based on future rewards to enhance the robustness of safety alignment. Experimental results show that the red-team model exhibits state-of-the-art attack capabilities, while the target model significantly improves its performance on safety benchmarks.

摘要: 针对大型语言模型（LLM）的越狱攻击的激增凸显了对强有力安全措施的必要性。然而，在多轮对话中，恶意意图可能隐藏在互动中，导致LLM更容易产生有害响应。在本文中，我们提出了\textBF{M}ulti-\textBF{T}urn \textBF{S} ajax\textBF{A} lignation（\ourapproach）框架，以解决在多轮交互中保护LLM的挑战。它由两个阶段组成：在思想引导的攻击学习阶段，红队模型学习思想引导的多轮越狱攻击以生成对抗提示。在对抗迭代优化阶段，红队模型和目标模型不断提高各自的交互能力。此外，我们引入了基于未来回报的多轮强化学习算法，以增强安全对齐的鲁棒性。实验结果表明，红队模型展现出最先进的攻击能力，而目标模型在安全基准方面的性能显着提高。



## **40. Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems**

针对基于R1的检索增强生成系统的思想链中毒攻击 cs.IR

7 pages,3 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16367v1) [paper-pdf](http://arxiv.org/pdf/2505.16367v1)

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan

**Abstract**: Retrieval-augmented generation (RAG) systems can effectively mitigate the hallucination problem of large language models (LLMs),but they also possess inherent vulnerabilities. Identifying these weaknesses before the large-scale real-world deployment of RAG systems is of great importance, as it lays the foundation for building more secure and robust RAG systems in the future. Existing adversarial attack methods typically exploit knowledge base poisoning to probe the vulnerabilities of RAG systems, which can effectively deceive standard RAG models. However, with the rapid advancement of deep reasoning capabilities in modern LLMs, previous approaches that merely inject incorrect knowledge are inadequate when attacking RAG systems equipped with deep reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this paper extracts reasoning process templates from R1-based RAG systems, uses these templates to wrap erroneous knowledge into adversarial documents, and injects them into the knowledge base to attack RAG systems. The key idea of our approach is that adversarial documents, by simulating the chain-of-thought patterns aligned with the model's training signals, may be misinterpreted by the model as authentic historical reasoning processes, thus increasing their likelihood of being referenced. Experiments conducted on the MS MARCO passage ranking dataset demonstrate the effectiveness of our proposed method.

摘要: 检索增强生成（RAG）系统可以有效地缓解大型语言模型（LLM）的幻觉问题，但它们也具有固有的漏洞。在RAG系统大规模现实部署之前识别这些弱点非常重要，因为它为未来构建更安全、更强大的RAG系统奠定了基础。现有的对抗攻击方法通常利用知识库中毒来探测RAG系统的漏洞，这可以有效地欺骗标准RAG模型。然而，随着现代LLM深度推理能力的迅速进步，以前仅仅注入错误知识的方法在攻击配备深度推理能力的RAG系统时是不够的。受LLM深度思维能力的启发，本文从基于R1的RAG系统中提取推理过程模板，使用这些模板将错误知识包装到对抗文档中，并将其注入知识库中以攻击RAG系统。我们方法的关键思想是，通过模拟与模型训练信号一致的思维链模式，对抗性文档可能会被模型误解为真实的历史推理过程，从而增加它们被引用的可能性。在MS MARCO通过排名数据集上进行的实验证明了我们提出的方法的有效性。



## **41. SuperPure: Efficient Purification of Localized and Distributed Adversarial Patches via Super-Resolution GAN Models**

SuperPure：通过超分辨率GAN模型有效纯化局部和分布式对抗补丁 cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16318v1) [paper-pdf](http://arxiv.org/pdf/2505.16318v1)

**Authors**: Hossein Khalili, Seongbin Park, Venkat Bollapragada, Nader Sehatbakhsh

**Abstract**: As vision-based machine learning models are increasingly integrated into autonomous and cyber-physical systems, concerns about (physical) adversarial patch attacks are growing. While state-of-the-art defenses can achieve certified robustness with minimal impact on utility against highly-concentrated localized patch attacks, they fall short in two important areas: (i) State-of-the-art methods are vulnerable to low-noise distributed patches where perturbations are subtly dispersed to evade detection or masking, as shown recently by the DorPatch attack; (ii) Achieving high robustness with state-of-the-art methods is extremely time and resource-consuming, rendering them impractical for latency-sensitive applications in many cyber-physical systems.   To address both robustness and latency issues, this paper proposes a new defense strategy for adversarial patch attacks called SuperPure. The key novelty is developing a pixel-wise masking scheme that is robust against both distributed and localized patches. The masking involves leveraging a GAN-based super-resolution scheme to gradually purify the image from adversarial patches. Our extensive evaluations using ImageNet and two standard classifiers, ResNet and EfficientNet, show that SuperPure advances the state-of-the-art in three major directions: (i) it improves the robustness against conventional localized patches by more than 20%, on average, while also improving top-1 clean accuracy by almost 10%; (ii) It achieves 58% robustness against distributed patch attacks (as opposed to 0% in state-of-the-art method, PatchCleanser); (iii) It decreases the defense end-to-end latency by over 98% compared to PatchCleanser. Our further analysis shows that SuperPure is robust against white-box attacks and different patch sizes. Our code is open-source.

摘要: 随着基于视觉的机器学习模型越来越多地集成到自主和网络物理系统中，人们对（物理）对抗性补丁攻击的担忧也越来越多。虽然最先进的防御方法可以在对高度集中的局部补丁攻击的效用影响最小的情况下实现经认证的鲁棒性，但它们在两个重要领域存在不足：（i）最先进的方法容易受到低噪声分布式补丁的攻击，其中扰动被巧妙地分散以逃避检测或掩蔽，如最近的DorPatch攻击所示;（ii）用最先进的方法实现高鲁棒性非常耗时和耗费资源，使得它们对于许多网络物理系统中的延迟敏感应用程序来说不切实际。   为了解决鲁棒性和延迟问题，本文提出了一种新的对抗补丁攻击的防御策略，称为SuperPure。关键的新颖性是开发一个像素级掩蔽方案，该方案对分布式和局部补丁都具有鲁棒性。掩蔽涉及利用基于GAN的超分辨率方案来逐渐从对抗补丁中净化图像。我们使用ImageNet和两个标准分类器ResNet和EfficientNet进行的广泛评估表明，SuperPure在三个主要方向上推进了最先进的技术：（i）平均而言，它将传统局部补丁的鲁棒性提高了20%以上，同时还将top-1的清洁准确性提高了近10%;（ii）它对分布式补丁攻击的鲁棒性达到了58%（而最先进的方法PatchCleanser中的鲁棒性为0%）;（iii）与PatchCleanser相比，它将防御端到端延迟降低了98%以上。我们的进一步分析表明，SuperPure对于白盒攻击和不同补丁大小具有强大的鲁棒性。我们的代码是开源的。



## **42. Accelerating Targeted Hard-Label Adversarial Attacks in Low-Query Black-Box Settings**

加速低查询黑匣子设置中的有针对性的硬标签对抗攻击 cs.CV

This paper contains 11 pages, 7 figures and 3 tables. For associated  supplementary code, see https://github.com/mdppml/TEA

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16313v1) [paper-pdf](http://arxiv.org/pdf/2505.16313v1)

**Authors**: Arjhun Swaminathan, Mete Akgün

**Abstract**: Deep neural networks for image classification remain vulnerable to adversarial examples -- small, imperceptible perturbations that induce misclassifications. In black-box settings, where only the final prediction is accessible, crafting targeted attacks that aim to misclassify into a specific target class is particularly challenging due to narrow decision regions. Current state-of-the-art methods often exploit the geometric properties of the decision boundary separating a source image and a target image rather than incorporating information from the images themselves. In contrast, we propose Targeted Edge-informed Attack (TEA), a novel attack that utilizes edge information from the target image to carefully perturb it, thereby producing an adversarial image that is closer to the source image while still achieving the desired target classification. Our approach consistently outperforms current state-of-the-art methods across different models in low query settings (nearly 70\% fewer queries are used), a scenario especially relevant in real-world applications with limited queries and black-box access. Furthermore, by efficiently generating a suitable adversarial example, TEA provides an improved target initialization for established geometry-based attacks.

摘要: 用于图像分类的深度神经网络仍然容易受到对抗性示例的影响--这些小而难以察觉的扰动会导致错误分类。在黑匣子环境中，只有最终预测才能访问，由于决策区域狭窄，精心设计旨在错误分类到特定目标类别的有针对性的攻击尤其具有挑战性。当前最先进的方法通常利用分离源图像和目标图像的决策边界的几何属性，而不是合并来自图像本身的信息。相比之下，我们提出了目标边缘信息攻击（TEA），这是一种新型攻击，利用目标图像的边缘信息仔细扰动它，从而产生更接近源图像的对抗图像，同时仍然实现所需的目标分类。在低查询设置（使用的查询减少了近70%）下，我们的方法在不同模型上始终优于当前最先进的方法，这种情况在查询和黑匣子访问有限的现实世界应用程序中尤其相关。此外，通过有效生成合适的对抗示例，TEA为已建立的基于几何的攻击提供了改进的目标初始化。



## **43. Timestamp Manipulation: Timestamp-based Nakamoto-style Blockchains are Vulnerable**

时间戳操纵：基于时间戳的中本风格区块链很脆弱 cs.CR

26 pages, 6 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.05328v3) [paper-pdf](http://arxiv.org/pdf/2505.05328v3)

**Authors**: Junjie Hu, Na Ruan, Sisi Duan

**Abstract**: Nakamoto consensus are the most widely adopted decentralized consensus mechanism in cryptocurrency systems. Since it was proposed in 2008, many studies have focused on analyzing its security. Most of them focus on maximizing the profit of the adversary. Examples include the selfish mining attack [FC '14] and the recent riskless uncle maker (RUM) attack [CCS '23]. In this work, we introduce the Staircase-Unrestricted Uncle Maker (SUUM), the first block withholding attack targeting the timestamp-based Nakamoto-style blockchain. Through block withholding, timestamp manipulation, and difficulty risk control, SUUM adversaries are capable of launching persistent attacks with zero cost and minimal difficulty risk characteristics, indefinitely exploiting rewards from honest participants. This creates a self-reinforcing cycle that threatens the security of blockchains. We conduct a comprehensive and systematic evaluation of SUUM, including the attack conditions, its impact on blockchains, and the difficulty risks. Finally, we further discuss four feasible mitigation measures against SUUM.

摘要: Nakamoto共识是加密货币系统中最广泛采用的去中心化共识机制。自2008年提出以来，许多研究都集中在分析其安全性上。他们中的大多数都专注于使对手的利润最大化。例子包括自私的采矿攻击[FC ' 14]和最近的无风险叔叔制造商（RUM）攻击[CS ' 23]。在这项工作中，我们介绍了Staircase-Unrestricted Maker（SUUM），这是第一个针对基于时间戳的Nakamoto风格区块链的区块扣留攻击。通过区块扣留、时间戳操纵和难度风险控制，SUUM对手能够以零成本和最小难度风险特征发起持续攻击，无限期地利用诚实参与者的回报。这造成了一个自我强化的循环，威胁区块链的安全。我们对SUUM进行全面、系统的评估，包括攻击条件、对区块链的影响以及难度风险。最后，我们进一步讨论了四个可行的缓解措施，对SUUM。



## **44. Decentralized Nonconvex Robust Optimization over Unsafe Multiagent Systems: System Modeling, Utility, Resilience, and Privacy Analysis**

不安全多智能体系统上的分散非凸鲁棒优化：系统建模、效用、弹性和隐私分析 math.OC

15 pages, 15 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2409.18632v7) [paper-pdf](http://arxiv.org/pdf/2409.18632v7)

**Authors**: Jinhui Hu, Guo Chen, Huaqing Li, Huqiang Cheng, Xiaoyu Guo, Tingwen Huang

**Abstract**: Privacy leakage and Byzantine failures are two adverse factors to the intelligent decision-making process of multi-agent systems (MASs). Considering the presence of these two issues, this paper targets the resolution of a class of nonconvex optimization problems under the Polyak-{\L}ojasiewicz (P-{\L}) condition. To address this problem, we first identify and construct the adversary system model. To enhance the robustness of stochastic gradient descent methods, we mask the local gradients with Gaussian noises and adopt a resilient aggregation method self-centered clipping (SCC) to design a differentially private (DP) decentralized Byzantine-resilient algorithm, namely DP-SCC-PL, which simultaneously achieves differential privacy and Byzantine resilience. The convergence analysis of DP-SCC-PL is challenging since the convergence error can be contributed jointly by privacy-preserving and Byzantine-resilient mechanisms, as well as the nonconvex relaxation, which is addressed via seeking the contraction relationships among the disagreement measure of reliable agents before and after aggregation, together with the optimal gap. Theoretical results reveal that DP-SCC-PL achieves consensus among all reliable agents and sublinear (inexact) convergence with well-designed step-sizes. It has also been proved that if there are no privacy issues and Byzantine agents, then the asymptotic exact convergence can be recovered. Numerical experiments verify the utility, resilience, and differential privacy of DP-SCC-PL by tackling a nonconvex optimization problem satisfying the P-{\L} condition under various Byzantine attacks.

摘要: 隐私泄露和拜占庭式故障是多代理系统（MAS）智能决策过程的两个不利因素。考虑到这两个问题的存在，本文针对Polyak-{\L}ojasiewicz（P-{\L}）条件下的一类非凸优化问题进行了求解。为了解决这个问题，我们首先识别并构建对手系统模型。为了增强随机梯度下降方法的鲁棒性，我们用高斯噪音掩盖局部梯度，并采用弹性聚合方法自中心剪裁（SCC）设计了一种差异私密（DP）去中心化拜占庭弹性算法，即DP-SCC-PL，同时实现了差异隐私和拜占庭弹性。DP-SCC-PL的收敛分析具有挑战性，因为收敛误差可以由隐私保护和拜占庭弹性机制以及非凸松弛共同造成，非凸松弛是通过寻求聚集前后可靠代理人的分歧测量之间的收缩关系来解决的，以及最佳差距。理论结果表明，DP-SCC-PL在所有可靠代理之间实现了共识，并通过精心设计的步进大小实现了亚线性（不精确）收敛。事实还证明，如果不存在隐私问题和拜占庭代理，则可以恢复渐进精确收敛。数值实验通过解决各种拜占庭攻击下满足P-{\L}条件的非凸优化问题来验证DP-SCC-PL的实用性、弹性和差异隐私。



## **45. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

PoisonArena：揭露检索增强一代中的竞争中毒攻击 cs.IR

29 pages

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.12574v3) [paper-pdf](http://arxiv.org/pdf/2505.12574v3)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.

摘要: 检索增强生成（RAG）系统，广泛用于改善大型语言模型（LLM）的事实基础，越来越容易受到中毒攻击，其中对手将操纵的内容注入检索器的语料库。虽然以前的研究主要集中在单个攻击者的设置，但现实世界的场景往往涉及多个相互竞争的攻击者，这些攻击者的目标相互冲突。在这项工作中，我们介绍PoisonArena，第一个基准系统地研究和评估竞争中毒攻击在RAG。我们形式化的多攻击者威胁模型，攻击者争夺控制答案相同的查询使用互斥的错误信息。PoisonArena利用Bradley-Terry模型来量化每种方法在此类对抗环境中的竞争有效性。通过对Natural Questions和MS MARCO数据集的广泛实验，我们证明了许多孤立成功的攻击策略在竞争压力下失败。我们的研究结果强调了攻击成功率（SVR）和F1评分等传统评估指标的局限性，并强调了竞争性评估来评估现实世界攻击稳健性的必要性。PoisonArena提供了一个标准化的框架，可以在更现实的多对手条件下基准和开发未来的攻击和防御策略。项目页面：https://github.com/yxf203/PoisonArena。



## **46. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

ErasableMass：针对黑匣子人脸识别模型的稳健且可擦除的隐私保护方案 cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.17038v4) [paper-pdf](http://arxiv.org/pdf/2412.17038v4)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Yueyun Shang, Zhihong Tian

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.

摘要: 虽然人脸识别模型在人脸验证和识别方面带来了显着的便利，但它们也对公众构成了重大的隐私风险。现有的人脸隐私保护方案通常采用对抗性样本来破坏FR模型的人脸验证。然而，这些计划往往遭受弱的可转移性对黑箱FR模型和永久损坏的可识别信息，不能满足授权操作，如取证和认证的要求。为了解决这些限制，我们提出了ErasableMask，一个强大的和可擦除的隐私保护计划，对黑盒FR模型。具体来说，通过重新思考代理FR模型之间的内在关系，ErasableMass引入了一种新型的元辅助攻击，该攻击通过在稳定和平衡的优化策略中学习更多通用特征来提高黑匣子的可移植性。它还提供了一种扰动误差机制，该机制支持受保护面部中的语义扰动误差，而不会降低图像质量。为了进一步提高性能，ErasableMass采用课程学习策略来减轻对抗攻击和扰动误差之间的优化冲突。对CelebA-HQ和FFHQ数据集的广泛实验表明，ErasableMass在可移植性方面实现了最先进的性能，在商用FR系统中平均置信度超过72%。此外，ErasableMass还表现出出色的扰动出错性能，出错成功率超过90%。



## **47. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

PandaGuard：针对越狱攻击的LLM安全性系统评估 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.13862v2) [paper-pdf](http://arxiv.org/pdf/2505.13862v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.

摘要: 大型语言模型（LLM）已经取得了卓越的能力，但仍然容易受到被称为越狱的对抗性提示的影响，这可能会绕过安全对齐并引发有害的输出。尽管LLM安全研究的努力越来越多，但现有的评估往往是分散的，集中在孤立的攻击或防御技术上，缺乏系统的，可重复的分析。在这项工作中，我们引入了PandaGuard，一个统一的模块化框架，将LLM越狱安全建模为一个由攻击者，防御者和法官组成的多代理系统。我们的框架实现了19种攻击方法和12种防御机制，以及多种判断策略，所有这些都在一个灵活的插件架构中，支持多种LLM接口，多种交互模式和配置驱动的实验，从而增强了可重复性和实际部署。基于这个框架，我们开发了PandaBench，这是一个全面的基准，可评估49个LLM和各种判断方法之间的相互作用，需要超过30亿个代币来执行。我们的广泛评估揭示了对模型漏洞、国防成本-性能权衡和判断一致性的关键见解。我们发现，没有一种防御在所有维度上都是最佳的，而且判断分歧会在安全评估中引入非平凡的方差。我们发布代码、配置和评估结果，以支持LLM安全性方面的透明和可重复研究。



## **48. SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning**

SafeKey：放大啊哈时刻洞察以实现安全推理 cs.AI

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16186v1) [paper-pdf](http://arxiv.org/pdf/2505.16186v1)

**Authors**: Kaiwen Zhou, Xuandong Zhao, Gaowen Liu, Jayanth Srinivasa, Aosong Feng, Dawn Song, Xin Eric Wang

**Abstract**: Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations.

摘要: 大型推理模型（LRM）引入了新一代的显式推理范式，在回答之前，导致显着改善复杂的任务。然而，它们对有害查询和对抗性攻击构成了巨大的安全风险。虽然最近LRM的主流安全措施，监督微调（SFT），提高了安全性能，我们发现，SFT对齐的模型很难推广到看不见的越狱提示。在对LRM一代进行彻底调查后，我们发现了一个可以激活安全推理并导致安全响应的安全啊哈时刻。这个啊哈时刻通常出现在“关键时刻”中，它遵循模型的查询理解过程，并且可以指示模型是否将安全地继续进行。基于这些见解，我们提出了SafeKey，其中包括两个补充目标，以更好地激活关键句中的安全啊哈时刻：（1）双路径安全头，以增强关键句之前模型内部表示中的安全信号，（2）查询面具建模目标，以提高模型对其查询理解的关注度，这具有重要的安全提示。跨多个安全基准的实验表明，我们的方法显着提高了对各种越狱攻击和分发外有害提示的安全概括性，将平均危害率降低9.6%，同时保持一般能力。我们的分析揭示了SafeKey如何通过重塑内部注意力和提高隐藏表示的质量来增强安全性。



## **49. TRAIL: Transferable Robust Adversarial Images via Latent diffusion**

TRAIL：通过潜在扩散的可转移鲁棒对抗图像 cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16166v1) [paper-pdf](http://arxiv.org/pdf/2505.16166v1)

**Authors**: Yuhao Xue, Zhifei Zhang, Xinyang Jiang, Yifei Shen, Junyao Gao, Wentao Gu, Jiale Zhao, Miaojing Shi, Cairong Zhao

**Abstract**: Adversarial attacks exploiting unrestricted natural perturbations present severe security risks to deep learning systems, yet their transferability across models remains limited due to distribution mismatches between generated adversarial features and real-world data. While recent works utilize pre-trained diffusion models as adversarial priors, they still encounter challenges due to the distribution shift between the distribution of ideal adversarial samples and the natural image distribution learned by the diffusion model. To address the challenge, we propose Transferable Robust Adversarial Images via Latent Diffusion (TRAIL), a test-time adaptation framework that enables the model to generate images from a distribution of images with adversarial features and closely resembles the target images. To mitigate the distribution shift, during attacks, TRAIL updates the diffusion U-Net's weights by combining adversarial objectives (to mislead victim models) and perceptual constraints (to preserve image realism). The adapted model then generates adversarial samples through iterative noise injection and denoising guided by these objectives. Experiments demonstrate that TRAIL significantly outperforms state-of-the-art methods in cross-model attack transferability, validating that distribution-aligned adversarial feature synthesis is critical for practical black-box attacks.

摘要: 利用不受限制的自然扰动的对抗攻击给深度学习系统带来了严重的安全风险，但由于生成的对抗特征与现实世界数据之间的分布不匹配，它们在模型之间的可移植性仍然有限。虽然最近的作品利用预先训练的扩散模型作为对抗性先验，但由于理想对抗性样本的分布和扩散模型学习的自然图像分布之间的分布变化，它们仍然面临挑战。为了应对这一挑战，我们提出了通过潜在扩散传输的鲁棒对抗图像（TRAIL），这是一种测试时自适应框架，使模型能够从具有对抗特征且与目标图像非常相似的图像分布中生成图像。为了减轻分布偏移，在攻击过程中，TRAIL通过结合对抗目标（误导受害者模型）和感知约束（保持图像真实性）来更新扩散U-Net的权重。然后，适应模型通过迭代噪声注入和这些目标指导的去噪生成对抗样本。实验表明，TRAIL在跨模型攻击可转移性方面明显优于最先进的方法，验证了分布对齐的对抗性特征合成对于实际黑盒攻击至关重要。



## **50. Robustifying Vision-Language Models via Dynamic Token Reweighting**

通过动态令牌重新加权来增强视觉语言模型 cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17132v1) [paper-pdf](http://arxiv.org/pdf/2505.17132v1)

**Authors**: Tanqiu Jiang, Jiacheng Liang, Rongyi Zhu, Jiawei Zhou, Fenglong Ma, Ting Wang

**Abstract**: Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. The code for replicating DTR is available: https://anonymous.4open.science/r/DTR-2755 (warning: this paper contains potentially harmful content generated by VLMs.)

摘要: 大型视觉语言模型（VLM）极易受到越狱攻击，这些攻击利用视觉与文本交互来绕过安全护栏。在本文中，我们提出了DTR，这是一种新型的推理时防御，通过优化模型的key-Value（KV）缓存来减轻多模式越狱攻击。我们不是依赖精心策划的安全特定数据或昂贵的图像到文本转换，而是引入了视觉模式引发的安全相关分布转变的新公式。该公式使DTR能够动态调整视觉令牌权重，最大限度地减少对抗视觉输入的影响，同时保留模型的一般能力和推理效率。对各种VLM和攻击基准的广泛评估表明，\sys在攻击稳健性和良性任务性能方面都优于现有防御，标志着在多模式基础模型中首次成功应用KV缓存优化来增强安全性。复制DTR的代码可获取：https://anonymous.4open.science/r/DTR-2755（警告：本文包含VLM生成的潜在有害内容。）



