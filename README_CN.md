# Latest Adversarial Attack Papers
**update at 2025-12-30 12:28:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing**

对基于LLM的学术评论的多语言隐藏提示注入攻击 cs.CL

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23684v1) [paper-pdf](https://arxiv.org/pdf/2512.23684v1)

**Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai. -Doss

**Abstract**: Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.

摘要: 大型语言模型（LLM）越来越多地被考虑用于高影响力的工作流程，包括学术同行评审。然而，LLM很容易受到文档级隐藏提示注入攻击。在这项工作中，我们构建了一个由ICML接受的大约500篇真实学术论文组成的数据集，并评估在这些文档中嵌入隐藏的对抗提示的效果。每份论文都注入了四种不同语言的语义等效指令，并使用LLM进行审查。我们发现，及时注射会导致英语、日语和中文注射的审查分数和接受/拒绝决定发生重大变化，而阿拉伯语注射几乎没有影响。这些结果凸显了基于LLM的审查系统对文档级提示注入的敏感性，并揭示了不同语言之间脆弱性的显着差异。



## **2. RobustMask: Certified Robustness against Adversarial Neural Ranking Attack via Randomized Masking**

RobustMass：通过随机掩蔽来对抗性神经排名攻击的鲁棒性 cs.CR

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23307v1) [paper-pdf](https://arxiv.org/pdf/2512.23307v1)

**Authors**: Jiawei Liu, Zhuo Chen, Rui Zhu, Miaokun Chen, Yuyang Gong, Wei Lu, Xiaofeng Wang

**Abstract**: Neural ranking models have achieved remarkable progress and are now widely deployed in real-world applications such as Retrieval-Augmented Generation (RAG). However, like other neural architectures, they remain vulnerable to adversarial manipulations: subtle character-, word-, or phrase-level perturbations can poison retrieval results and artificially promote targeted candidates, undermining the integrity of search engines and downstream systems. Existing defenses either rely on heuristics with poor generalization or on certified methods that assume overly strong adversarial knowledge, limiting their practical use. To address these challenges, we propose RobustMask, a novel defense that combines the context-prediction capability of pretrained language models with a randomized masking-based smoothing mechanism. Our approach strengthens neural ranking models against adversarial perturbations at the character, word, and phrase levels. Leveraging both the pairwise comparison ability of ranking models and probabilistic statistical analysis, we provide a theoretical proof of RobustMask's certified top-K robustness. Extensive experiments further demonstrate that RobustMask successfully certifies over 20% of candidate documents within the top-10 ranking positions against adversarial perturbations affecting up to 30% of their content. These results highlight the effectiveness of RobustMask in enhancing the adversarial robustness of neural ranking models, marking a significant step toward providing stronger security guarantees for real-world retrieval systems.

摘要: 神经排名模型已经取得了显着的进展，现已广泛部署在现实世界的应用中，例如检索增强生成（RAG）。然而，与其他神经架构一样，它们仍然容易受到对抗性操纵：微妙的字符、单词或短语级扰动可能会毒害检索结果并人为地促进目标候选，从而破坏搜索引擎和下游系统的完整性。现有的防御要么依赖于概括性较差的启发式方法，要么依赖于假设过于强大的对抗性知识的认证方法，从而限制了其实际使用。为了应对这些挑战，我们提出了RobustMasking，这是一种新型防御，将预训练语言模型的上下文预测能力与基于随机掩蔽的平滑机制相结合。我们的方法增强了神经排名模型，以对抗字符、单词和短语级别的对抗性扰动。利用排名模型的成对比较能力和概率统计分析，我们提供了RobustMass认证的Top K稳健性的理论证明。广泛的实验进一步表明，RobustMass成功认证了前10名排名中超过20%的候选文档，免受影响多达30%内容的对抗性干扰。这些结果凸显了RobustMass在增强神经排名模型对抗鲁棒性方面的有效性，标志着朝着为现实世界的检索系统提供更强的安全保障迈出了重要一步。



## **3. It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents**

这是一个陷阱！任务重定向代理Web代理说服基准 cs.HC

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23128v1) [paper-pdf](https://arxiv.org/pdf/2512.23128v1)

**Authors**: Karolina Korgul, Yushi Yang, Arkadiusz Drohomirecki, Piotr Błaszczyk, Will Howard, Lukas Aichberger, Chris Russell, Philip H. S. Torr, Adam Mahdi, Adel Bibi

**Abstract**: Web-based agents powered by large language models are increasingly used for tasks such as email management or professional networking. Their reliance on dynamic web content, however, makes them vulnerable to prompt injection attacks: adversarial instructions hidden in interface elements that persuade the agent to divert from its original task. We introduce the Task-Redirecting Agent Persuasion Benchmark (TRAP), an evaluation for studying how persuasion techniques misguide autonomous web agents on realistic tasks. Across six frontier models, agents are susceptible to prompt injection in 25\% of tasks on average (13\% for GPT-5 to 43\% for DeepSeek-R1), with small interface or contextual changes often doubling success rates and revealing systemic, psychologically driven vulnerabilities in web-based agents. We also provide a modular social-engineering injection framework with controlled experiments on high-fidelity website clones, allowing for further benchmark expansion.

摘要: 由大型语言模型支持的基于Web的代理越来越多地用于电子邮件管理或专业网络等任务。然而，它们对动态网络内容的依赖使它们容易受到提示注入攻击：隐藏在界面元素中的对抗指令，说服代理从其原始任务转移。我们介绍了任务重定向代理说服基准（TRAP），这是一项评估，旨在研究说服技术如何在现实任务中误导自主网络代理。在六个前沿模型中，代理人平均容易在25%的任务中立即注入（GPT-5为13%，DeepSeek-R1为43%），微小的界面或上下文变化通常会使成功率翻倍，并揭示了基于网络的代理中系统性、心理驱动的漏洞。我们还提供了一个模块化的社会工程注入框架，在高保真网站克隆上进行受控实验，允许进一步的基准扩展。



## **4. DECEPTICON: How Dark Patterns Manipulate Web Agents**

DECEPTICON：暗模式如何操纵Web代理 cs.CR

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22894v1) [paper-pdf](https://arxiv.org/pdf/2512.22894v1)

**Authors**: Phil Cuvin, Hao Zhu, Diyi Yang

**Abstract**: Deceptive UI designs, widely instantiated across the web and commonly known as dark patterns, manipulate users into performing actions misaligned with their goals. In this paper, we show that dark patterns are highly effective in steering agent trajectories, posing a significant risk to agent robustness. To quantify this risk, we introduce DECEPTICON, an environment for testing individual dark patterns in isolation. DECEPTICON includes 700 web navigation tasks with dark patterns -- 600 generated tasks and 100 real-world tasks, designed to measure instruction-following success and dark pattern effectiveness. Across state-of-the-art agents, we find dark patterns successfully steer agent trajectories towards malicious outcomes in over 70% of tested generated and real-world tasks -- compared to a human average of 31%. Moreover, we find that dark pattern effectiveness correlates positively with model size and test-time reasoning, making larger, more capable models more susceptible. Leading countermeasures against adversarial attacks, including in-context prompting and guardrail models, fail to consistently reduce the success rate of dark pattern interventions. Our findings reveal dark patterns as a latent and unmitigated risk to web agents, highlighting the urgent need for robust defenses against manipulative designs.

摘要: 欺骗性的UI设计在网络上广泛实例化，通常称为黑暗模式，操纵用户执行与其目标不一致的操作。在本文中，我们表明暗模式在引导代理轨迹方面非常有效，对代理稳健性构成了重大风险。为了量化这种风险，我们引入了DECPTICON，这是一种用于隔离测试单个暗图案的环境。DECPTICON包括700个具有黑暗模式的网络导航任务--600个生成任务和100个现实世界任务，旨在衡量描述跟踪成功和黑暗模式有效性。在最先进的代理中，我们发现在超过70%的测试生成和现实世界任务中，黑暗模式成功地将代理轨迹引导到恶意结果，而人类的平均水平为31%。此外，我们发现暗模式有效性与模型大小和测试时推理正相关，使更大、更强大的模型更容易受到影响。针对对抗性攻击的主要对策，包括背景提示和护栏模型，未能持续降低暗模式干预的成功率。我们的研究结果揭示了黑暗模式对网络代理来说是一种潜在且不可减轻的风险，凸显了对操纵性设计的强有力防御的迫切需要。



## **5. Adaptive Trust Consensus for Blockchain IoT: Comparing RL, DRL, and MARL Against Naive, Collusive, Adaptive, Byzantine, and Sleeper Attacks**

区块链物联网的自适应信任共识：比较RL、DRL和MARL与天真、共谋、自适应、拜占庭和休眠攻击 cs.CR

34 pages, 19 figures, 10 tables. Code available at https://github.com/soham-padia/blockchain-iot-trust

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22860v1) [paper-pdf](https://arxiv.org/pdf/2512.22860v1)

**Authors**: Soham Padia, Dhananjay Vaidya, Ramchandra Mangrulkar

**Abstract**: Securing blockchain-enabled IoT networks against sophisticated adversarial attacks remains a critical challenge. This paper presents a trust-based delegated consensus framework integrating Fully Homomorphic Encryption (FHE) with Attribute-Based Access Control (ABAC) for privacy-preserving policy evaluation, combined with learning-based defense mechanisms. We systematically compare three reinforcement learning approaches -- tabular Q-learning (RL), Deep RL with Dueling Double DQN (DRL), and Multi-Agent RL (MARL) -- against five distinct attack families: Naive Malicious Attack (NMA), Collusive Rumor Attack (CRA), Adaptive Adversarial Attack (AAA), Byzantine Fault Injection (BFI), and Time-Delayed Poisoning (TDP). Experimental results on a 16-node simulated IoT network reveal significant performance variations: MARL achieves superior detection under collusive attacks (F1=0.85 vs. DRL's 0.68 and RL's 0.50), while DRL and MARL both attain perfect detection (F1=1.00) against adaptive attacks where RL fails (F1=0.50). All agents successfully defend against Byzantine attacks (F1=1.00). Most critically, the Time-Delayed Poisoning attack proves catastrophic for all agents, with F1 scores dropping to 0.11-0.16 after sleeper activation, demonstrating the severe threat posed by trust-building adversaries. Our findings indicate that coordinated multi-agent learning provides measurable advantages for defending against sophisticated trust manipulation attacks in blockchain IoT environments.

摘要: 保护支持区块链的物联网网络免受复杂的对抗性攻击仍然是一个关键挑战。提出了一种基于信任的委托共识框架，该框架将全同态加密（FHE）和基于属性的访问控制（ABAC）相结合，结合基于学习的防御机制，用于隐私保护策略评估。我们系统地比较了三种强化学习方法-表格Q学习（RL），Deep RL with Dueling Double DQN（DRL）和Multi-Agent RL（MARL）-针对五种不同的攻击家族：天真恶意攻击（NMA），共谋谣言攻击（CRA），自适应对抗攻击（AAA），拜占庭故障注入（BFI）和延时中毒（TDP）。在16节点模拟物联网网络上的实验结果显示了显著的性能差异：MARL在共谋攻击下实现了卓越的检测（F1=0.85 vs. DRL的0.68和RL的0.50），而DRL和MARL在RL失败（F1=0.50）的自适应攻击下都实现了完美的检测（F1 =1.00）。所有特工都成功抵御了拜占庭攻击（F1=1.00）。最重要的是，事实证明，延时中毒攻击对所有特工来说都是灾难性的，休眠激活后F1评分下降至0.11-0.16，这表明建立信任的对手构成了严重威胁。我们的研究结果表明，协调的多代理学习为防御区块链物联网环境中复杂的信任操纵攻击提供了可衡量的优势。



## **6. Reach-Avoid Differential game with Reachability Analysis for UAVs: A decomposition approach**

具有无人机可达性分析的可达-避免差异博弈：分解方法 eess.SY

Paper version accepted to the Journal of Guidance, Control, and Dynamics (JGCD)

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22793v1) [paper-pdf](https://arxiv.org/pdf/2512.22793v1)

**Authors**: Minh Bui, Simon Monckton, Mo Chen

**Abstract**: Reach-avoid (RA) games have significant applications in security and defense, particularly for unmanned aerial vehicles (UAVs). These problems are inherently challenging due to the need to consider obstacles, consider the adversarial nature of opponents, ensure optimality, and account for nonlinear dynamics. Hamilton-Jacobi (HJ) reachability analysis has emerged as a powerful tool for tackling these challenges; however, while it has been applied to games involving two spatial dimensions, directly extending this approach to three spatial dimensions is impossible due to high dimensionality. On the other hand, alternative approaches for solving RA games lack the generality to consider games with three spatial dimensions involving agents with non-trivial system dynamics. In this work, we propose a novel framework for dimensionality reduction by decomposing the problem into a horizontal RA sub-game and a vertical RA sub-game. We then solve each sub-game using HJ reachability analysis and consider second-order dynamics that account for the defender's acceleration. To reconstruct the solution to the original RA game from the sub-games, we introduce a HJ-based tracking control algorithm in each sub-game that not only guarantees capture of the attacker but also tracking of the attacker thereafter. We prove the conditions under which the capture guarantees are maintained. The effectiveness of our approach is demonstrated via numerical simulations, showing that the decomposition maintains optimality and guarantees in the original problem. Our methods are also validated in a Gazebo physics simulator, achieving successful capture of quadrotors in three spatial dimensions space for the first time to the best of our knowledge.

摘要: 避免触及（RA）游戏在安全和国防方面有着重要的应用，特别是对于无人机（UFO）。这些问题本质上具有挑战性，因为需要考虑障碍、考虑对手的对抗性、确保最优性并考虑非线性动态。汉密尔顿-雅各比（TJ）可达性分析已成为应对这些挑战的强大工具;然而，虽然它已应用于涉及两个空间维度的游戏，但由于维度较高，将这种方法直接扩展到三个空间维度是不可能的。另一方面，解决RA游戏的替代方法缺乏考虑涉及具有非平凡系统动态的主体的三个空间维度游戏的通用性。在这项工作中，我们提出了一个新颖的降维框架，通过将问题分解为水平RA子博弈和垂直RA子博弈。然后我们使用HJ可达性分析来解决每个子博弈，并考虑考虑防守者加速度的二阶动力学。为了从子游戏中重建原始RA游戏的解决方案，我们在每个子游戏中引入了一种基于TJ的跟踪控制算法，不仅保证捕获攻击者，而且还保证随后跟踪攻击者。我们证明维持捕获保证的条件。通过数值模拟证明了我们方法的有效性，表明分解保持了最优性并保证了原始问题。我们的方法还在Gazebo物理模拟器中得到了验证，据我们所知，首次实现了在三维空间中成功捕获四螺旋桨。



## **7. Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks**

对尖峰神经网络的对抗鲁棒性进行可靠评估 cs.LG

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.22522v1) [paper-pdf](https://arxiv.org/pdf/2512.22522v1)

**Authors**: Jihang Wang, Dongcheng Zhao, Ruolin Chen, Qian Zhang, Yi Zeng

**Abstract**: Spiking Neural Networks (SNNs) utilize spike-based activations to mimic the brain's energy-efficient information processing. However, the binary and discontinuous nature of spike activations causes vanishing gradients, making adversarial robustness evaluation via gradient descent unreliable. While improved surrogate gradient methods have been proposed, their effectiveness under strong adversarial attacks remains unclear. We propose a more reliable framework for evaluating SNN adversarial robustness. We theoretically analyze the degree of gradient vanishing in surrogate gradients and introduce the Adaptive Sharpness Surrogate Gradient (ASSG), which adaptively evolves the shape of the surrogate function according to the input distribution during attack iterations, thereby enhancing gradient accuracy while mitigating gradient vanishing. In addition, we design an adversarial attack with adaptive step size under the $L_\infty$ constraint-Stable Adaptive Projected Gradient Descent (SA-PGD), achieving faster and more stable convergence under imprecise gradients. Extensive experiments show that our approach substantially increases attack success rates across diverse adversarial training schemes, SNN architectures and neuron models, providing a more generalized and reliable evaluation of SNN adversarial robustness. The experimental results further reveal that the robustness of current SNNs has been significantly overestimated and highlighting the need for more dependable adversarial training methods.

摘要: 尖峰神经网络（SNN）利用基于尖峰的激活来模拟大脑的节能信息处理。然而，尖峰激活的二元和不连续性质会导致梯度消失，从而使得通过梯度下降进行的对抗鲁棒性评估不可靠。虽然已经提出了改进的替代梯度方法，但它们在强对抗攻击下的有效性仍不清楚。我们提出了一个更可靠的框架来评估SNN对抗稳健性。我们从理论上分析了代理梯度中梯度消失的程度，并引入了自适应Shareptium Surrogate Gradient（ASSG），它根据攻击迭代期间的输入分布自适应地进化代理函数的形状，从而在减轻梯度消失的同时提高了梯度准确性。此外，我们在$L_\infty$ constraint-Stable Adaptive Projected Gradient Down（SA-PVD）下设计了一种具有自适应步进大小的对抗攻击，在不精确的梯度下实现更快、更稳定的收敛。大量实验表明，我们的方法大大提高了各种对抗训练方案、SNN架构和神经元模型的攻击成功率，为SNN对抗鲁棒性提供了更普遍和可靠的评估。实验结果进一步表明，当前SNN的稳健性被显着高估，并凸显了对更可靠的对抗训练方法的需求。



## **8. NOWA: Null-space Optical Watermark for Invisible Capture Fingerprinting and Tamper Localization**

NOWA：用于隐形捕获指纹识别和篡改定位的零空间光学水印 cs.CR

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.22501v1) [paper-pdf](https://arxiv.org/pdf/2512.22501v1)

**Authors**: Edwin Vargas

**Abstract**: Ensuring the authenticity and ownership of digital images is increasingly challenging as modern editing tools enable highly realistic forgeries. Existing image protection systems mainly rely on digital watermarking, which is susceptible to sophisticated digital attacks. To address this limitation, we propose a hybrid optical-digital framework that incorporates physical authentication cues during image formation and preserves them through a learned reconstruction process. At the optical level, a phase mask in the camera aperture produces a Null-space Optical Watermark (NOWA) that lies in the Null Space of the imaging operator and therefore remains invisible in the captured image. Then, a Null-Space Network (NSN) performs measurement-consistent reconstruction that delivers high-quality protected images while preserving the NOWA signature. The proposed design enables tamper localization by projecting the image onto the camera's null space and detecting pixel-level inconsistencies. Our design preserves perceptual quality, resists common degradations such as compression, and establishes a structural security asymmetry: without access to the optical or NSN parameters, adversaries cannot forge the NOWA signature. Experiments with simulations and a prototype camera demonstrate competitive performance in terms of image quality preservation, and tamper localization accuracy compared to state-of-the-art digital watermarking and learning-based authentication methods.

摘要: 确保数字图像的真实性和所有权越来越具有挑战性，因为现代编辑工具可以实现高度逼真的图像。现有的图像保护系统主要依赖于数字水印技术，但数字水印技术容易受到复杂的数字攻击。为了解决这一限制，我们提出了一个混合的光学数字框架，在图像形成过程中结合了物理认证线索，并通过学习重建过程保留它们。在光学层面上，相机孔径中的相位掩模产生位于成像算子的光空间中的零空间光学水印（NOWA），因此在捕获的图像中保持不可见。然后，空空间网络（NSN）执行测量一致的重建，提供高质量的受保护图像，同时保留NOWA签名。提出的设计通过将图像投影到相机的零空间并检测像素级不一致来实现篡改定位。我们的设计保留了感知质量，抵抗压缩等常见降级，并建立了结构性安全不对称：如果不访问光学或NSN参数，对手就无法伪造NOWA签名。模拟和原型相机的实验表明，与最先进的数字水印和基于学习的认证方法相比，在图像质量保存和篡改定位准确性方面具有竞争力。



## **9. PHANTOM: Physics-Aware Adversarial Attacks against Federated Learning-Coordinated EV Charging Management System**

PHANTOM：针对联邦学习协调电动汽车充电管理系统的物理感知对抗攻击 cs.ET

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.22381v1) [paper-pdf](https://arxiv.org/pdf/2512.22381v1)

**Authors**: Mohammad Zakaria Haider, Amit Kumar Podder, Prabin Mali, Aranya Chakrabortty, Sumit Paudyal, Mohammad Ashiqur Rahman

**Abstract**: The rapid deployment of electric vehicle charging stations (EVCS) within distribution networks necessitates intelligent and adaptive control to maintain the grid's resilience and reliability. In this work, we propose PHANTOM, a physics-aware adversarial network that is trained and optimized through a multi-agent reinforcement learning model. PHANTOM integrates a physics-informed neural network (PINN) enabled by federated learning (FL) that functions as a digital twin of EVCS-integrated systems, ensuring physically consistent modeling of operational dynamics and constraints. Building on this digital twin, we construct a multi-agent RL environment that utilizes deep Q-networks (DQN) and soft actor-critic (SAC) methods to derive adversarial false data injection (FDI) strategies capable of bypassing conventional detection mechanisms. To examine the broader grid-level consequences, a transmission and distribution (T and D) dual simulation platform is developed, allowing us to capture cascading interactions between EVCS disturbances at the distribution level and the operations of the bulk transmission system. Results demonstrate how learned attack policies disrupt load balancing and induce voltage instabilities that propagate across T and D boundaries. These findings highlight the critical need for physics-aware cybersecurity to ensure the resilience of large-scale vehicle-grid integration.

摘要: 配电网内电动汽车充电站（EVCS）的快速部署需要智能和自适应控制，以维持电网的弹性和可靠性。在这项工作中，我们提出了PHANTOM，这是一个物理感知的对抗网络，通过多智能体强化学习模型进行训练和优化。PHANTOM集成了一个由联邦学习（FL）支持的物理信息神经网络（PINN），该网络充当ECVS集成系统的数字孪生体，确保操作动态和约束的物理一致建模。在这个数字双胞胎的基础上，我们构建了一个多智能体RL环境，该环境利用深度Q网络（DQN）和软行动者评论家（SAC）方法来推导能够绕过传统检测机制的对抗性虚假数据注入（Direct）策略。为了检查更广泛的电网级后果，开发了输电和配电（T和D）双重模拟平台，使我们能够捕捉配电级EVCS干扰与批量输电系统运行之间的级联相互作用。结果展示了学习到的攻击策略如何破坏负载平衡并引发跨越T和D边界传播的电压不稳定性。这些发现凸显了对物理感知网络安全的迫切需要，以确保大规模车网集成的弹性。



## **10. Scaling Adversarial Training via Data Selection**

通过数据选择扩展对抗训练 cs.LG

6 pages. Conference workshop paper

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.22069v1) [paper-pdf](https://arxiv.org/pdf/2512.22069v1)

**Authors**: Youran Ye, Dejin Wang, Ajinkya Bhandare

**Abstract**: Projected Gradient Descent (PGD) is a strong and widely used first-order adversarial attack, yet its computational cost scales poorly, as all training samples undergo identical iterative inner-loop optimization despite contributing unequally to robustness. Motivated by this inefficiency, we propose \emph{Selective Adversarial Training}, which perturbs only a subset of critical samples in each minibatch. Specifically, we introduce two principled selection criteria: (1) margin-based sampling, which prioritizes samples near the decision boundary, and (2) gradient-matching sampling, which selects samples whose gradients align with the dominant batch optimization direction. Adversarial examples are generated only for the selected subset, while the remaining samples are trained cleanly using a mixed objective. Experiments on MNIST and CIFAR-10 show that the proposed methods achieve robustness comparable to, or even exceeding, full PGD adversarial training, while reducing adversarial computation by up to $50\%$, demonstrating that informed sample selection is sufficient for scalable adversarial robustness.

摘要: 投影梯度下降（PGD）是一种强大且广泛使用的一阶对抗攻击，但其计算成本很低，因为所有训练样本都经过相同的迭代内环优化，尽管对鲁棒性的贡献不一样。出于这种低效率的动机，我们提出了\n {选择性对抗训练}，它只扰动每个小批量中的关键样本的子集。具体来说，我们引入了两个原则性的选择标准：（1）基于边缘的采样，它优先考虑决策边界附近的样本，以及（2）梯度匹配采样，它选择梯度与主要批次优化方向一致的样本。对抗性示例仅针对所选子集生成，而其余样本则使用混合目标进行干净训练。MNIST和CIFAR-10上的实验表明，所提出的方法实现了与完整PVD对抗训练相当甚至超过的鲁棒性，同时将对抗计算减少了高达50%$，这表明明智的样本选择足以实现可扩展的对抗鲁棒性。



## **11. Few Tokens Matter: Entropy Guided Attacks on Vision-Language Models**

很少有令牌重要：对视觉语言模型的熵引导攻击 cs.CV

19 Pages,11 figures,8 tables

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.21815v1) [paper-pdf](https://arxiv.org/pdf/2512.21815v1)

**Authors**: Mengqi He, Xinyu Tian, Xin Shen, Jinhong Ni, Shu Zou, Zhaoyuan Yang, Jing Zhang

**Abstract**: Vision-language models (VLMs) achieve remarkable performance but remain vulnerable to adversarial attacks. Entropy, a measure of model uncertainty, is strongly correlated with the reliability of VLM. Prior entropy-based attacks maximize uncertainty at all decoding steps, implicitly assuming that every token contributes equally to generation instability. We show instead that a small fraction (about 20%) of high-entropy tokens, i.e., critical decision points in autoregressive generation, disproportionately governs output trajectories. By concentrating adversarial perturbations on these positions, we achieve semantic degradation comparable to global methods while using substantially smaller budgets. More importantly, across multiple representative VLMs, such selective attacks convert 35-49% of benign outputs into harmful ones, exposing a more critical safety risk. Remarkably, these vulnerable high-entropy forks recur across architecturally diverse VLMs, enabling feasible transferability (17-26% harmful rates on unseen targets). Motivated by these findings, we propose Entropy-bank Guided Adversarial attacks (EGA), which achieves competitive attack success rates (93-95%) alongside high harmful conversion, thereby revealing new weaknesses in current VLM safety mechanisms.

摘要: 视觉语言模型（VLM）取得了出色的性能，但仍然容易受到对抗攻击。模型不确定性的衡量指标--与VLM的可靠性密切相关。先前的基于信息量的攻击最大化了所有解码步骤的不确定性，隐含地假设每个令牌对生成不稳定性的贡献相同。相反，我们表明一小部分（约20%）高熵代币，即自回归生成中的关键决策点不成比例地控制着产出轨迹。通过将对抗性扰动集中在这些位置上，我们实现了与全球方法相当的语义降级，同时使用更少的预算。更重要的是，在多个代表性的VLM中，此类选择性攻击将35-49%的良性输出转化为有害输出，暴露了更严重的安全风险。值得注意的是，这些脆弱的高熵分叉会在架构多样的VLM中重复出现，从而实现了可行的可转移性（对不可见目标的有害率为17-26%）。受这些发现的启发，我们提出了熵库引导的对抗攻击（EGA），它实现了竞争性攻击成功率（93-95%）以及高有害转换，从而揭示了当前VLM安全机制的新弱点。



## **12. LLM-Driven Feature-Level Adversarial Attacks on Android Malware Detectors**

LLM驱动的Android恶意软件检测器的冲突级对抗攻击 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21404v1) [paper-pdf](https://arxiv.org/pdf/2512.21404v1)

**Authors**: Tianwei Lan, Farid Naït-Abdesselam

**Abstract**: The rapid growth in both the scale and complexity of Android malware has driven the widespread adoption of machine learning (ML) techniques for scalable and accurate malware detection. Despite their effectiveness, these models remain vulnerable to adversarial attacks that introduce carefully crafted feature-level perturbations to evade detection while preserving malicious functionality. In this paper, we present LAMLAD, a novel adversarial attack framework that exploits the generative and reasoning capabilities of large language models (LLMs) to bypass ML-based Android malware classifiers. LAMLAD employs a dual-agent architecture composed of an LLM manipulator, which generates realistic and functionality-preserving feature perturbations, and an LLM analyzer, which guides the perturbation process toward successful evasion. To improve efficiency and contextual awareness, LAMLAD integrates retrieval-augmented generation (RAG) into the LLM pipeline. Focusing on Drebin-style feature representations, LAMLAD enables stealthy and high-confidence attacks against widely deployed Android malware detection systems. We evaluate LAMLAD against three representative ML-based Android malware detectors and compare its performance with two state-of-the-art adversarial attack methods. Experimental results demonstrate that LAMLAD achieves an attack success rate (ASR) of up to 97%, requiring on average only three attempts per adversarial sample, highlighting its effectiveness, efficiency, and adaptability in practical adversarial settings. Furthermore, we propose an adversarial training-based defense strategy that reduces the ASR by more than 30% on average, significantly enhancing model robustness against LAMLAD-style attacks.

摘要: Android恶意软件规模和复杂性的快速增长推动了机器学习（ML）技术的广泛采用，以进行可扩展和准确的恶意软件检测。尽管它们有效，但这些模型仍然容易受到对抗攻击，这些攻击引入精心设计的功能级扰动，以逃避检测，同时保留恶意功能。在本文中，我们介绍了LAMRAD，这是一种新型的对抗性攻击框架，它利用大型语言模型（LLM）的生成和推理能力来绕过基于ML的Android恶意软件分类器。LAMLAT采用双代理架构，由LLM操纵器和LLM分析器组成，LLM操纵器生成真实且功能保留的特征扰动，LLM分析器引导扰动过程成功规避。为了提高效率和上下文感知，LAMRAD将检索增强生成（RAG）集成到LLM管道中。LAMRAD专注于Drebin风格的特征表示，能够针对广泛部署的Android恶意软件检测系统进行隐蔽且高可信度的攻击。我们针对三种代表性的基于ML的Android恶意软件检测器评估LAMRAD，并将其性能与两种最先进的对抗攻击方法进行比较。实验结果表明，LAMRAD的攻击成功率（ASB）高达97%，每个对抗样本平均只需尝试三次，凸显了其在实际对抗环境中的有效性、效率和适应性。此外，我们提出了一种基于对抗训练的防御策略，平均将ASR降低了30%以上，显著增强了模型对LAMLAD攻击的鲁棒性。



## **13. CoTDeceptor:Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents**

CoTDeceptor：针对CoT增强LLM代码代理的对抗代码混淆 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21250v1) [paper-pdf](https://arxiv.org/pdf/2512.21250v1)

**Authors**: Haoyang Li, Mingjin Li, Jinxin Zuo, Siqi Li, Xiao Li, Hao Wu, Yueming Lu, Xiaochuan He

**Abstract**: LLM-based code agents(e.g., ChatGPT Codex) are increasingly deployed as detector for code review and security auditing tasks. Although CoT-enhanced LLM vulnerability detectors are believed to provide improved robustness against obfuscated malicious code, we find that their reasoning chains and semantic abstraction processes exhibit exploitable systematic weaknesses.This allows attackers to covertly embed malicious logic, bypass code review, and propagate backdoored components throughout real-world software supply chains.To investigate this issue, we present CoTDeceptor, the first adversarial code obfuscation framework targeting CoT-enhanced LLM detectors. CoTDeceptor autonomously constructs evolving, hard-to-reverse multi-stage obfuscation strategy chains that effectively disrupt CoT-driven detection logic.We obtained malicious code provided by security enterprise, experimental results demonstrate that CoTDeceptor achieves stable and transferable evasion performance against state-of-the-art LLMs and vulnerability detection agents. CoTDeceptor bypasses 14 out of 15 vulnerability categories, compared to only 2 bypassed by prior methods. Our findings highlight potential risks in real-world software supply chains and underscore the need for more robust and interpretable LLM-powered security analysis systems.

摘要: 基于LLM的代码代理（例如，ChatGPT Codex）越来越多地被部署为代码审查和安全审计任务的检测器。尽管CoT增强型LLM漏洞检测器被认为可以针对混淆的恶意代码提供更好的鲁棒性，但我们发现它们的推理链和语义抽象过程表现出可利用的系统弱点。这使得攻击者能够秘密嵌入恶意逻辑、绕过代码审查并在整个现实世界的软件供应链中传播后门组件。为了研究这个问题，我们提出了CoTDeceptor，第一个针对CoT增强型LLM检测器的对抗代码混淆框架。CoTDeceptor自主构建不断发展的、难以逆转的多阶段混淆策略链，有效扰乱CoT驱动的检测逻辑。我们获得了安全企业提供的恶意代码，实验结果表明CoTDeceptor针对最先进的LLM和漏洞检测代理实现了稳定且可转移的规避性能。CoTDeceptor绕过了15个漏洞类别中的14个，而之前的方法只绕过了2个。我们的研究结果强调了现实世界软件供应链中的潜在风险，并强调了对更强大和可解释的LLM支持的安全分析系统的需求。



## **14. Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks**

提高搜索高效硬标签攻击的射线搜索优化的收敛率 cs.LG

Published at AAAI 2026 (Oral). This version corresponds to the conference proceedings; v2 will include the appendix

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21241v1) [paper-pdf](https://arxiv.org/pdf/2512.21241v1)

**Authors**: Xinjie Xu, Shuyu Cheng, Dongwei Xu, Qi Xuan, Chen Ma

**Abstract**: In hard-label black-box adversarial attacks, where only the top-1 predicted label is accessible, the prohibitive query complexity poses a major obstacle to practical deployment. In this paper, we focus on optimizing a representative class of attacks that search for the optimal ray direction yielding the minimum $\ell_2$-norm perturbation required to move a benign image into the adversarial region. Inspired by Nesterov's Accelerated Gradient (NAG), we propose a momentum-based algorithm, ARS-OPT, which proactively estimates the gradient with respect to a future ray direction inferred from accumulated momentum. We provide a theoretical analysis of its convergence behavior, showing that ARS-OPT enables more accurate directional updates and achieves faster, more stable optimization. To further accelerate convergence, we incorporate surrogate-model priors into ARS-OPT's gradient estimation, resulting in PARS-OPT with enhanced performance. The superiority of our approach is supported by theoretical guarantees under standard assumptions. Extensive experiments on ImageNet and CIFAR-10 demonstrate that our method surpasses 13 state-of-the-art approaches in query efficiency.

摘要: 在硬标签黑匣子对抗攻击中，只能访问前1名的预测标签，令人望而却步的查询复杂性对实际部署构成了主要障碍。在本文中，我们重点优化一类代表性攻击，这些攻击搜索最佳射线方向，产生将良性图像移动到对抗区域所需的最小$\ell_2 $-norm扰动。受Nesterov加速梯度（NAG）的启发，我们提出了一种基于动量的算法ARS-OPT，该算法主动估计相对于从累积动量推断的未来射线方向的梯度。我们对其收敛行为进行了理论分析，表明ARS-OPT能够实现更准确的方向更新，并实现更快、更稳定的优化。为了进一步加速收敛，我们将代理模型先验纳入ARS-OPT的梯度估计中，从而产生性能增强的PARS-OPT。我们方法的优越性得到了标准假设下的理论保证的支持。ImageNet和CIFAR-10上的大量实验表明，我们的方法在查询效率方面超过了13种最先进的方法。



## **15. Time-Bucketed Balance Records: Bounded-Storage Ephemeral Tokens for Resource-Constrained Systems**

分时段平衡记录：资源受限系统的有界存储短暂令牌 cs.DS

14 pages, 1 figure, 1 Algorithm, 3 Theorems

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20962v1) [paper-pdf](https://arxiv.org/pdf/2512.20962v1)

**Authors**: Shaun Scovil, Bhargav Chickmagalur Nanjundappa

**Abstract**: Fungible tokens with time-to-live (TTL) semantics require tracking individual expiration times for each deposited unit. A naive implementation creates a new balance record per deposit, leading to unbounded storage growth and vulnerability to denial-of-service attacks. We present time-bucketed balance records, a data structure that bounds storage to O(k) records per account while guaranteeing that tokens never expire before their configured TTL. Our approach discretizes time into k buckets, coalescing deposits within the same bucket to limit unique expiration timestamps. We prove three key properties: (1) storage is bounded by k+1 records regardless of deposit frequency, (2) actual expiration time is always at least the configured TTL, and (3) adversaries cannot increase a victim's operation cost beyond O(k)[amortized] worst case. We provide a reference implementation in Solidity with measured gas costs demonstrating practical efficiency.

摘要: 具有生存时间（TLR）语义的可替代代币需要跟踪每个存入单位的单独到期时间。天真的实施会为每次存款创建新的余额记录，从而导致存储无限增长并容易受到拒绝服务攻击。我们提供分时段的余额记录，这是一种数据结构，将每个帐户的存储限制为O（k）个记录，同时保证令牌不会在其配置的TLR之前到期。我们的方法将时间离散化到k个桶中，将存款合并在同一桶中以限制唯一的到期时间戳。我们证明了三个关键属性：（1）无论存款频率如何，存储都以k+1条记录为界限，（2）实际到期时间始终至少为配置的TLR，以及（3）对手不能将受害者的操作成本增加到O（k）以上[摊销]最坏情况。我们在Solidity中提供了一个参考实施，其中测量的天然气成本证明了实际效率。



## **16. The Imitation Game: Using Large Language Models as Chatbots to Combat Chat-Based Cybercrimes**

模仿游戏：使用大型语言模型作为聊天机器人来打击基于聊天的网络犯罪 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21371v1) [paper-pdf](https://arxiv.org/pdf/2512.21371v1)

**Authors**: Yifan Yao, Baojuan Wang, Jinhao Duan, Kaidi Xu, ChuanKai Guo, Zhibo Eric Sun, Yue Zhang

**Abstract**: Chat-based cybercrime has emerged as a pervasive threat, with attackers leveraging real-time messaging platforms to conduct scams that rely on trust-building, deception, and psychological manipulation. Traditional defense mechanisms, which operate on static rules or shallow content filters, struggle to identify these conversational threats, especially when attackers use multimedia obfuscation and context-aware dialogue.   In this work, we ask a provocative question inspired by the classic Imitation Game: Can machines convincingly pose as human victims to turn deception against cybercriminals? We present LURE (LLM-based User Response Engagement), the first system to deploy Large Language Models (LLMs) as active agents, not as passive classifiers, embedded within adversarial chat environments.   LURE combines automated discovery, adversarial interaction, and OCR-based analysis of image-embedded payment data. Applied to the setting of illicit video chat scams on Telegram, our system engaged 53 actors across 98 groups. In over 56 percent of interactions, the LLM maintained multi-round conversations without being noticed as a bot, effectively "winning" the imitation game. Our findings reveal key behavioral patterns in scam operations, such as payment flows, upselling strategies, and platform migration tactics.

摘要: 基于聊天的网络犯罪已成为一种普遍存在的威胁，攻击者利用实时消息平台来实施依赖于信任建立、欺骗和心理操纵的诈骗。传统防御机制基于静态规则或浅层内容过滤器，难以识别这些对话威胁，尤其是当攻击者使用多媒体混淆和上下文感知对话时。   在这部作品中，我们提出了一个受经典模仿游戏启发的挑衅性问题：机器能否令人信服地冒充人类受害者，利用欺骗手段对付网络犯罪分子？我们介绍了LURE（基于LLM的用户响应参与），这是第一个将大型语言模型（LLM）部署为嵌入在对抗性聊天环境中的主动代理而不是被动分类器的系统。   LURE结合了自动发现、对抗交互和基于OCR的图像嵌入式支付数据分析。应用于Telegram上的非法视频聊天诈骗设置，我们的系统涉及98个群组的53名参与者。在超过56%的互动中，LLM保持多轮对话，而不会被机器人注意到，有效地“赢得”了模仿游戏。我们的调查结果揭示了诈骗操作中的关键行为模式，例如支付流、向上销售策略和平台迁移策略。



## **17. Robustness Certificates for Neural Networks against Adversarial Attacks**

神经网络抗对抗性攻击的鲁棒性证明 cs.LG

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20865v1) [paper-pdf](https://arxiv.org/pdf/2512.20865v1)

**Authors**: Sara Taheri, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Majid Zamani

**Abstract**: The increasing use of machine learning in safety-critical domains amplifies the risk of adversarial threats, especially data poisoning attacks that corrupt training data to degrade performance or induce unsafe behavior. Most existing defenses lack formal guarantees or rely on restrictive assumptions about the model class, attack type, extent of poisoning, or point-wise certification, limiting their practical reliability. This paper introduces a principled formal robustness certification framework that models gradient-based training as a discrete-time dynamical system (dt-DS) and formulates poisoning robustness as a formal safety verification problem. By adapting the concept of barrier certificates (BCs) from control theory, we introduce sufficient conditions to certify a robust radius ensuring that the terminal model remains safe under worst-case ${\ell}_p$-norm based poisoning. To make this practical, we parameterize BCs as neural networks trained on finite sets of poisoned trajectories. We further derive probably approximately correct (PAC) bounds by solving a scenario convex program (SCP), which yields a confidence lower bound on the certified robustness radius generalizing beyond the training set. Importantly, our framework also extends to certification against test-time attacks, making it the first unified framework to provide formal guarantees in both training and test-time attack settings. Experiments on MNIST, SVHN, and CIFAR-10 show that our approach certifies non-trivial perturbation budgets while being model-agnostic and requiring no prior knowledge of the attack or contamination level.

摘要: 机器学习在安全关键领域的使用越来越多，放大了对抗威胁的风险，特别是破坏训练数据以降低性能或引发不安全行为的数据中毒攻击。大多数现有的防御缺乏正式保证或依赖于有关模型类别、攻击类型、中毒程度或逐点认证的限制性假设，从而限制了其实际可靠性。本文介绍了一个有原则的正式鲁棒性认证框架，该框架将基于梯度的训练建模为离散时间动态系统（dt-DS），并将中毒鲁棒性制定为正式安全验证问题。通过改编来自控制理论的屏障证书（BC）概念，我们引入了充分条件来证明稳健半径，以确保终端模型在最坏情况下${\ell}_p$-norm基于中毒的情况下保持安全。为了实现这一点，我们将BC参数化为在有限组中毒轨迹上训练的神经网络。我们进一步通过求解场景凸规划（SCP）来推导出可能大致正确（PAC）界限，这会产生扩展到训练集之外的认证稳健性半径的置信下限。重要的是，我们的框架还扩展到针对测试时攻击的认证，使其成为第一个在训练和测试时攻击环境中提供正式保证的统一框架。MNIST、SVHN和CIFAR-10上的实验表明，我们的方法可以证明非平凡的扰动预算，同时是模型不可知的，并且不需要攻击或污染水平的先验知识。



## **18. Defending against adversarial attacks using mixture of experts**

使用混合专家抵御对抗攻击 cs.LG

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20821v1) [paper-pdf](https://arxiv.org/pdf/2512.20821v1)

**Authors**: Mohammad Meymani, Roozbeh Razavi-Far

**Abstract**: Machine learning is a powerful tool enabling full automation of a huge number of tasks without explicit programming. Despite recent progress of machine learning in different domains, these models have shown vulnerabilities when they are exposed to adversarial threats. Adversarial threats aim to hinder the machine learning models from satisfying their objectives. They can create adversarial perturbations, which are imperceptible to humans' eyes but have the ability to cause misclassification during inference. Moreover, they can poison the training data to harm the model's performance or they can query the model to steal its sensitive information. In this paper, we propose a defense system, which devises an adversarial training module within mixture-of-experts architecture to enhance its robustness against adversarial threats. In our proposed defense system, we use nine pre-trained experts with ResNet-18 as their backbone. During end-to-end training, the parameters of expert models and gating mechanism are jointly updated allowing further optimization of the experts. Our proposed defense system outperforms state-of-the-art defense systems and plain classifiers, which use a more complex architecture than our model's backbone.

摘要: 机器学习是一种强大的工具，无需显式编程即可实现大量任务的完全自动化。尽管机器学习最近在不同领域取得了进展，但这些模型在面临对抗威胁时仍表现出脆弱性。对抗性威胁旨在阻碍机器学习模型实现其目标。它们可以产生对抗性扰动，人类肉眼无法察觉，但有能力在推理过程中导致错误分类。此外，他们可以毒害训练数据以损害模型的性能，或者他们可以查询模型以窃取其敏感信息。在本文中，我们提出了一种防御系统，该系统在混合专家架构中设计了一个对抗训练模块，以增强其对对抗威胁的鲁棒性。在我们提出的防御系统中，我们使用九名经过预先培训的专家，以ResNet-18为骨干。在端到端训练过程中，专家模型和门控机制的参数联合更新，从而进一步优化专家。我们提出的防御系统优于最先进的防御系统和普通分类器，后者使用比我们模型的主干更复杂的架构。



## **19. Safety Alignment of LMs via Non-cooperative Games**

通过非合作博弈实现LM的安全调整 cs.AI

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20806v1) [paper-pdf](https://arxiv.org/pdf/2512.20806v1)

**Authors**: Anselm Paulus, Ilia Kulikov, Brandon Amos, Rémi Munos, Ivan Evtimov, Kamalika Chaudhuri, Arman Zharmagambetov

**Abstract**: Ensuring the safety of language models (LMs) while maintaining their usefulness remains a critical challenge in AI alignment. Current approaches rely on sequential adversarial training: generating adversarial prompts and fine-tuning LMs to defend against them. We introduce a different paradigm: framing safety alignment as a non-zero-sum game between an Attacker LM and a Defender LM trained jointly via online reinforcement learning. Each LM continuously adapts to the other's evolving strategies, driving iterative improvement. Our method uses a preference-based reward signal derived from pairwise comparisons instead of point-wise scores, providing more robust supervision and potentially reducing reward hacking. Our RL recipe, AdvGame, shifts the Pareto frontier of safety and utility, yielding a Defender LM that is simultaneously more helpful and more resilient to adversarial attacks. In addition, the resulting Attacker LM converges into a strong, general-purpose red-teaming agent that can be directly deployed to probe arbitrary target models.

摘要: 确保语言模型（LM）的安全性同时保持其有用性仍然是人工智能协调的一个关键挑战。当前的方法依赖于顺序对抗训练：生成对抗提示并微调LM以抵御它们。我们引入了一种不同的范式：将安全对齐框架为攻击者LM和防御者LM之间通过在线强化学习联合训练的非零和游戏。每个LM都不断适应对方不断发展的策略，推动迭代改进。我们的方法使用基于偏好的奖励信号，而不是逐点比较，提供更强大的监督，并可能减少奖励黑客。我们的RL配方AdvGame改变了安全性和实用性的帕累托边界，产生了一个防御者LM，同时对对抗性攻击更有帮助，更有弹性。此外，由此产生的攻击者LM收敛到一个强大的，通用的红队代理，可以直接部署到探测任意目标模型。



## **20. Failure Analysis of Safety Controllers in Autonomous Vehicles Under Object-Based LiDAR Attacks**

基于对象的LiDART攻击下自动驾驶汽车安全控制器的故障分析 cs.SE

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.22244v1) [paper-pdf](https://arxiv.org/pdf/2512.22244v1)

**Authors**: Daniyal Ganiuly, Nurzhau Bolatbek, Assel Smaiyl

**Abstract**: Autonomous vehicles rely on LiDAR based perception to support safety critical control functions such as adaptive cruise control and automatic emergency braking. While previous research has shown that LiDAR perception can be manipulated through object based spoofing and injection attacks, the impact of such attacks on vehicle safety controllers is still not well understood. This paper presents a systematic failure analysis of longitudinal safety controllers under object based LiDAR attacks in highway driving scenarios. The study focuses on realistic cut in and car following situations in which adversarial objects introduce persistent perception errors without directly modifying vehicle control software. A high fidelity simulation framework integrating LiDAR perception, object tracking, and closed loop vehicle control is used to evaluate how false and displaced object detections propagate through the perception planning and control pipeline. The results demonstrate that even short duration LiDAR induced object hallucinations can trigger unsafe braking, delayed responses to real hazards, and unstable control behavior. In cut in scenarios, a clear increase in unsafe deceleration events and time to collision violations is observed when compared to benign conditions, despite identical controller parameters. The analysis further shows that controller failures are more strongly influenced by the temporal consistency of spoofed objects than by spatial inaccuracies alone. These findings reveal a critical gap between perception robustness and control level safety guarantees in autonomous driving systems. By explicitly characterizing safety controller failure modes under adversarial perception, this work provides practical insights for the design of attack aware safety mechanisms and more resilient control strategies for LiDAR dependent autonomous vehicles.

摘要: 自动驾驶汽车依靠基于LiDART的感知来支持安全关键控制功能，例如自适应巡航控制和自动紧急制动。虽然之前的研究表明，LiDART感知可以通过基于对象的欺骗和注入攻击来操纵，但此类攻击对车辆安全控制器的影响仍然没有得到很好的理解。本文对高速公路驾驶场景中纵向安全控制器在基于对象的LiDART攻击下的系统性故障进行了分析。该研究的重点是现实的切入和汽车跟随情况，其中对抗对象在不直接修改车辆控制软件的情况下会引入持续的感知错误。集成LiDART感知、对象跟踪和闭环车辆控制的高保真度模拟框架用于评估错误和位移对象检测如何通过感知规划和控制管道传播。结果表明，即使是短时间的LiDART诱导的物体幻觉也会引发不安全的制动、对真实危险的延迟反应以及不稳定的控制行为。在切入场景中，尽管控制器参数相同，但与良性条件相比，观察到不安全减速事件和碰撞违规时间明显增加。分析进一步表明，控制器故障受到欺骗对象的时间一致性的影响比仅受空间不准确性的影响更大。这些发现揭示了自动驾驶系统中感知鲁棒性和控制级别安全保证之间的关键差距。通过在对抗性感知下明确描述安全控制器故障模式，这项工作为设计基于LiDART的自动驾驶车辆的攻击感知安全机制和更具弹性的控制策略提供了实用见解。



## **21. Satellite Cybersecurity Across Orbital Altitudes: Analyzing Ground-Based Threats to LEO, MEO, and GEO**

跨轨道高度的卫星网络安全：分析对LEO、MEO和GEO的地面威胁 cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.21367v1) [paper-pdf](https://arxiv.org/pdf/2512.21367v1)

**Authors**: Mark Ballard, Guanqun Song, Ting Zhu

**Abstract**: The rapid proliferation of satellite constellations, particularly in Low Earth Orbit (LEO), has fundamentally altered the global space infrastructure, shifting the risk landscape from purely kinetic collisions to complex cyber-physical threats. While traditional safety frameworks focus on debris mitigation, ground-based adversaries increasingly exploit radio-frequency links, supply chain vulnerabilities, and software update pathways to degrade space assets. This paper presents a comparative analysis of satellite cybersecurity across LEO, Medium Earth Orbit (MEO), and Geostationary Earth Orbit (GEO) regimes. By synthesizing data from 60 publicly documented security incidents with key vulnerability proxies--including Telemetry, Tracking, and Command (TT&C) anomalies, encryption weaknesses, and environmental stressors--we characterize how orbital altitude dictates attack feasibility and impact. Our evaluation reveals distinct threat profiles: GEO systems are predominantly targeted via high-frequency uplink exposure, whereas LEO constellations face unique risks stemming from limited power budgets, hardware constraints, and susceptibility to thermal and radiation-induced faults. We further bridge the gap between security and sustainability, arguing that unmitigated cyber vulnerabilities accelerate hardware obsolescence and debris accumulation, undermining efforts toward carbon-neutral space operations. The results demonstrate that weak encryption and command path irregularities are the most consistent predictors of adversarial success across all orbits.

摘要: 卫星星座的迅速扩散，特别是在低地轨道（LEO）中，从根本上改变了全球太空基础设施，将风险格局从纯粹的动能碰撞转变为复杂的网络物理威胁。虽然传统的安全框架侧重于碎片减缓，但地面对手越来越多地利用射频链路、供应链漏洞和软件更新途径来降低太空资产的质量。本文对低地球轨道、中地球轨道（MEO）和地球同步地球轨道（GEO）制度下的卫星网络安全进行了比较分析。通过将60起公开记录的安全事件的数据与关键漏洞代理（包括远程通信、跟踪和命令（TT & C）异常、加密弱点和环境压力源）综合起来，我们描述了轨道高度如何决定攻击的可行性和影响。我们的评估揭示了不同的威胁概况：GEO系统主要通过高频上行链路暴露来攻击，而LEO星座则面临着来自有限的电力预算、硬件限制以及对热和辐射引发故障的敏感性的独特风险。我们进一步弥合了安全性和可持续性之间的差距，认为毫无缓解的网络漏洞加速了硬件报废和碎片积累，破坏了碳中和太空行动的努力。结果表明，弱加密和命令路径不规则是所有轨道上对抗成功的最一致预测因素。



## **22. Real-World Adversarial Attacks on RF-Based Drone Detectors**

现实世界对基于射频的无人机探测器的对抗攻击 cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20712v1) [paper-pdf](https://arxiv.org/pdf/2512.20712v1)

**Authors**: Omer Gazit, Yael Itzhakev, Yuval Elovici, Asaf Shabtai

**Abstract**: Radio frequency (RF) based systems are increasingly used to detect drones by analyzing their RF signal patterns, converting them into spectrogram images which are processed by object detection models. Existing RF attacks against image based models alter digital features, making over-the-air (OTA) implementation difficult due to the challenge of converting digital perturbations to transmittable waveforms that may introduce synchronization errors and interference, and encounter hardware limitations. We present the first physical attack on RF image based drone detectors, optimizing class-specific universal complex baseband (I/Q) perturbation waveforms that are transmitted alongside legitimate communications. We evaluated the attack using RF recordings and OTA experiments with four types of drones. Our results show that modest, structured I/Q perturbations are compatible with standard RF chains and reliably reduce target drone detection while preserving detection of legitimate drones.

摘要: 基于射频（RF）的系统越来越多地用于通过分析无人机的RF信号模式来检测无人机，将其转换为由对象检测模型处理的谱图图像。现有的针对基于图像的模型的RF攻击会改变数字特征，使空中（OTA）实施变得困难，因为将数字扰动转换为可传输的波型具有挑战性，这可能会引入同步错误和干扰，并遇到硬件限制。我们首次对基于RF图像的无人机检测器进行物理攻击，优化与合法通信一起传输的特定类别通用复基带（I/Q）扰动波。我们使用四种无人机的射频记录和OTA实验评估了这次攻击。我们的结果表明，适度的结构化I/Q扰动与标准RF链兼容，并可靠地减少目标无人机检测，同时保留对合法无人机的检测。



## **23. Evasion-Resilient Detection of DNS-over-HTTPS Data Exfiltration: A Practical Evaluation and Toolkit**

DNS over-HTTPS数据泄露的规避检测：实用评估和工具包 cs.CR

61 pages Advisor : Dr Darren Hurley-Smith

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20423v1) [paper-pdf](https://arxiv.org/pdf/2512.20423v1)

**Authors**: Adam Elaoumari

**Abstract**: The purpose of this project is to assess how well defenders can detect DNS-over-HTTPS (DoH) file exfiltration, and which evasion strategies can be used by attackers. While providing a reproducible toolkit to generate, intercept and analyze DoH exfiltration, and comparing Machine Learning vs threshold-based detection under adversarial scenarios. The originality of this project is the introduction of an end-to-end, containerized pipeline that generates configurable file exfiltration over DoH using several parameters (e.g., chunking, encoding, padding, resolver rotation). It allows for file reconstruction at the resolver side, while extracting flow-level features using a fork of DoHLyzer. The pipeline contains a prediction side, which allows the training of machine learning models based on public labelled datasets and then evaluates them side-by-side with threshold-based detection methods against malicious and evasive DNS-Over-HTTPS traffic. We train Random Forest, Gradient Boosting and Logistic Regression classifiers on a public DoH dataset and benchmark them against evasive DoH exfiltration scenarios. The toolkit orchestrates traffic generation, file capture, feature extraction, model training and analysis. The toolkit is then encapsulated into several Docker containers for easy setup and full reproducibility regardless of the platform it is run on. Future research regarding this project is directed at validating the results on mixed enterprise traffic, extending the protocol coverage to HTTP/3/QUIC request, adding a benign traffic generation, and working on real-time traffic evaluation. A key objective is to quantify when stealth constraints make DoH exfiltration uneconomical and unworthy for the attacker.

摘要: 该项目的目的是评估防御者如何检测DNS over HTTPS（DoH）文件泄露，以及攻击者可以使用哪些规避策略。同时提供一个可复制的工具包来生成，拦截和分析DoH渗出，并在对抗场景下比较机器学习与基于阈值的检测。该项目的独创性在于引入了一个端到端的容器化管道，该管道使用多个参数（例如，分块、编码、填充、解析器旋转）。它允许在解析器端进行文件重建，同时使用DoHLyzer的分叉提取流级特征。该管道包含一个预测端，它允许基于公共标签数据集训练机器学习模型，然后使用基于阈值的检测方法针对恶意和规避性DNS-Over-HTTPS流量并行评估它们。我们在公共DoH数据集上训练随机森林、梯度增强和逻辑回归分类器，并针对规避DoH外流场景对它们进行基准测试。该工具包协调流量生成、文件捕获、特征提取、模型训练和分析。然后，该工具包被封装到多个Docker容器中，无论其运行在什么平台上，都可以轻松设置和完全可重复性。有关该项目的未来研究旨在验证混合企业流量的结果，将协议覆盖范围扩展到HTTP/3/QUIC请求，添加良性流量生成，并进行实时流量评估。一个关键目标是量化何时隐形限制使DoH撤离对攻击者来说不经济且不值得。



## **24. Contingency Model-based Control (CMC) for Communicationless Cooperative Collision Avoidance in Robot Swarms**

机器人群中无通信协作避碰的基于应急模型的控制（MC） math.OC

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.20391v2) [paper-pdf](https://arxiv.org/pdf/2512.20391v2)

**Authors**: Georg Schildbach

**Abstract**: Cooperative collision avoidance between robots in swarm operations remains an open challenge. Assuming a decentralized architecture, each robot is responsible for making its own control decisions, including motion planning. To this end, most existing approaches mostly rely some form of (wireless) communication between the agents of the swarm. In reality, however, communication is brittle. It may be affected by latency, further delays and packet losses, transmission faults, and is subject to adversarial attacks, such as jamming or spoofing. This paper proposes Contingency Model-based Control (CMC) as a communicationless alternative. It follows the implicit cooperation paradigm, under which the design of the robots is based on consensual (offline) rules, similar to traffic rules. They include the definition of a contingency trajectory for each robot, and a method for construction of mutual collision avoidance constraints. The setup is shown to guarantee the recursive feasibility and collision avoidance between all swarm members in closed-loop operation. Moreover, CMC naturally satisfies the Plug \& Play paradigm, i.e., for new robots entering the swarm. Two numerical examples demonstrate that the collision avoidance guarantee is intact and that the robot swarm operates smoothly under the CMC regime.

摘要: 群操作中机器人之间的协作避免碰撞仍然是一个悬而未决的挑战。假设采用分散式架构，每个机器人负责做出自己的控制决策，包括运动规划。为此，大多数现有的方法大多依赖于群体代理之间某种形式的（无线）通信。然而，事实上，沟通是脆弱的。它可能会受到延迟、进一步延迟和数据包丢失、传输故障的影响，并容易受到对抗攻击，例如干扰或欺骗。本文提出了基于应急模型的控制（SMC）作为一种无通信替代方案。它遵循隐性合作范式，在该范式下，机器人的设计基于共识（离线）规则，类似于交通规则。它们包括定义每个机器人的应急轨迹，以及构建相互碰撞避免约束的方法。该设置是为了保证递归的可行性和所有群体成员之间的碰撞避免在闭环操作。此外，CMC自然满足即插即用范例，即，新的机器人进入蜂群。两个数值例子表明，避免碰撞的保证是完整的，机器人群体下CMC政权顺利运作。



## **25. Optimistic TEE-Rollups: A Hybrid Architecture for Scalable and Verifiable Generative AI Inference on Blockchain**

乐观的TEE-Rollops：区块链上可扩展和可验证的生成式人工智能推理的混合架构 cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20176v1) [paper-pdf](https://arxiv.org/pdf/2512.20176v1)

**Authors**: Aaron Chan, Alex Ding, Frank Chen, Alan Wu, Bruce Zhang, Arther Tian

**Abstract**: The rapid integration of Large Language Models (LLMs) into decentralized physical infrastructure networks (DePIN) is currently bottlenecked by the Verifiability Trilemma, which posits that a decentralized inference system cannot simultaneously achieve high computational integrity, low latency, and low cost. Existing cryptographic solutions, such as Zero-Knowledge Machine Learning (ZKML), suffer from superlinear proving overheads (O(k NlogN)) that render them infeasible for billionparameter models. Conversely, optimistic approaches (opML) impose prohibitive dispute windows, preventing real-time interactivity, while recent "Proof of Quality" (PoQ) paradigms sacrifice cryptographic integrity for subjective semantic evaluation, leaving networks vulnerable to model downgrade attacks and reward hacking. In this paper, we introduce Optimistic TEE-Rollups (OTR), a hybrid verification protocol that harmonizes these constraints. OTR leverages NVIDIA H100 Confidential Computing Trusted Execution Environments (TEEs) to provide sub-second Provisional Finality, underpinned by an optimistic fraud-proof mechanism and stochastic Zero-Knowledge spot-checks to mitigate hardware side-channel risks. We formally define Proof of Efficient Attribution (PoEA), a consensus mechanism that cryptographically binds execution traces to hardware attestations, thereby guaranteeing model authenticity. Extensive simulations demonstrate that OTR achieves 99% of the throughput of centralized baselines with a marginal cost overhead of $0.07 per query, maintaining Byzantine fault tolerance against rational adversaries even in the presence of transient hardware vulnerabilities.

摘要: 大型语言模型（LLM）快速集成到去中心化物理基础设施网络（DePin）中目前受到可验证性三困境的限制，该困境认为去中心化推理系统无法同时实现高计算完整性、低延迟和低成本。现有的加密解决方案，例如零知识机器学习（ZKML），存在超线性证明费用（O（k NlogN））的问题，这使得它们对于十亿个参数模型来说不可行。相反，乐观方法（opML）施加了禁止性的争议窗口，阻止了实时交互，而最近的“质量证明”（PoQ）范式则牺牲了加密完整性来进行主观语义评估，使网络容易受到模型降级攻击和奖励黑客攻击。在本文中，我们介绍了乐观TEE-Rollup（OTR），这是一种协调这些约束的混合验证协议。OTR利用NVIDIA H100机密计算可信执行环境（TEE）提供亚秒级临时最终结果，并以乐观的防欺诈机制和随机零知识抽查为基础，以减轻硬件侧通道风险。我们正式定义了有效归因证明（PoEA），这是一种共识机制，通过加密方式将执行跟踪与硬件证明绑定，从而保证模型的真实性。广泛的模拟表明，OTR实现了集中式基线99%的吞吐量，每次查询的边际成本费用为0.07美元，即使存在暂时性硬件漏洞，也能对理性对手保持拜占庭式的耐药性。



## **26. Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography**

Odysseus：通过双重隐写术破解商业多模式法学硕士集成系统 cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20168v1) [paper-pdf](https://arxiv.org/pdf/2512.20168v1)

**Authors**: Songze Li, Jiameng Cheng, Yiming Li, Xiaojun Jia, Dacheng Tao

**Abstract**: By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems.

摘要: 通过将语言理解与图像等感知模式集成起来，多模式大型语言模型（MLLM）构成了现代人工智能系统的重要基础，特别是在开放和交互环境中运行的智能代理。然而，它们的可访问性不断增加也增加了滥用风险，例如产生有害或不安全内容。为了减轻这些风险，通常应用对齐技术来将模型行为与人类价值观对齐。尽管做出了这些努力，最近的研究表明，越狱攻击可能会绕过对齐并引发不安全的输出。目前，大多数现有的越狱方法都是针对开源模型量身定制的，并且对于商业MLLM集成系统（通常使用额外的过滤器）的有效性有限。这些过滤器可以检测和防止恶意输入和输出内容，从而显着减少越狱威胁。在本文中，我们揭示了这些安全过滤器的成功在很大程度上依赖于一个关键假设，即恶意内容必须在输入或输出中显式可见。这种假设虽然通常适用于传统的LLM集成系统，但在MLLM集成系统中却出现了问题，攻击者可以利用多种模式来隐藏对抗意图，从而导致现有的MLLM集成系统中出现错误的安全感。为了挑战这一假设，我们提出了Odysseus，一种新的越狱范例，它引入了双重隐写术来秘密地将恶意查询和响应嵌入到看起来很好的图像中。在基准数据集上进行的大量实验表明，我们的Odysseus成功地越狱了几个开创性和现实的MLLM集成系统，攻击成功率高达99%。它暴露了现有防御中的一个基本盲点，并呼吁重新思考MLLM集成系统中的跨模式安全性。



## **27. AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications**

超越核心领域的人工智能安全：简历筛选作为专业LLM应用中对抗漏洞的案例研究 cs.CL

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20164v1) [paper-pdf](https://arxiv.org/pdf/2512.20164v1)

**Authors**: Honglin Mu, Jinghao Liu, Kaiyang Wan, Rui Xing, Xiuying Chen, Timothy Baldwin, Wanxiang Che

**Abstract**: Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.

摘要: 大型语言模型（LLM）擅长文本理解和生成，非常适合代码审查和内容审核等自动化任务。然而，我们的研究发现了一个漏洞：LLM可能会被隐藏在输入数据（例如简历或代码）中的“对抗指令”操纵，导致它们偏离预期任务。值得注意的是，虽然代码审查等成熟领域可能存在防御措施，但在简历筛选和同行审查等其他常见应用中通常不存在防御措施。本文引入了一个基准来评估简历筛选中的此漏洞，揭示了某些攻击类型的攻击成功率超过80%。我们评估了两种防御机制：基于预算的防御可以减少10.1%的攻击，错误拒绝增加12.5%，而我们提出的使用LoRA适应的FIDS（通过分离的外部指令检测）可以减少15.4%的攻击，错误拒绝增加10.4%。组合方法可减少26.3%的攻击，证明训练时防御在安全性和实用程序保存方面优于推理时缓解措施。



## **28. IoT-based Android Malware Detection Using Graph Neural Network With Adversarial Defense**

使用具有对抗性防御的图神经网络进行基于物联网的Android恶意软件检测 cs.CR

13 pages

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20004v1) [paper-pdf](https://arxiv.org/pdf/2512.20004v1)

**Authors**: Rahul Yumlembam, Biju Issac, Seibu Mary Jacob, Longzhi Yang

**Abstract**: Since the Internet of Things (IoT) is widely adopted using Android applications, detecting malicious Android apps is essential. In recent years, Android graph-based deep learning research has proposed many approaches to extract relationships from applications as graphs to generate graph embeddings. First, we demonstrate the effectiveness of graph-based classification using a Graph Neural Network (GNN)-based classifier to generate API graph embeddings. The graph embeddings are combined with Permission and Intent features to train multiple machine learning and deep learning models for Android malware detection. The proposed classification approach achieves an accuracy of 98.33 percent on the CICMaldroid dataset and 98.68 percent on the Drebin dataset. However, graph-based deep learning models are vulnerable, as attackers can add fake relationships to evade detection by the classifier. Second, we propose a Generative Adversarial Network (GAN)-based attack algorithm named VGAE-MalGAN targeting graph-based GNN Android malware classifiers. The VGAE-MalGAN generator produces adversarial malware API graphs, while the VGAE-MalGAN substitute detector attempts to mimic the target detector. Experimental results show that VGAE-MalGAN can significantly reduce the detection rate of GNN-based malware classifiers. Although the model initially fails to detect adversarial malware, retraining with generated adversarial samples improves robustness and helps mitigate adversarial attacks.

摘要: 由于物联网（IoT）广泛采用Android应用程序，因此检测恶意Android应用程序至关重要。近年来，基于Android图形的深度学习研究提出了许多方法，从应用程序中提取关系作为图形来生成图形嵌入。首先，我们使用基于图神经网络（GNN）的分类器来证明基于图的分类生成API图嵌入的有效性。图嵌入与权限和意图功能相结合，可以训练多个机器学习和深度学习模型来进行Android恶意软件检测。所提出的分类方法在CICMaldroid数据集上实现了98.33%的准确率，在Drebin数据集上实现了98.68%的准确率。然而，基于图的深度学习模型是脆弱的，因为攻击者可以添加虚假的关系来逃避分类器的检测。其次，我们提出了一种基于生成对抗网络（GAN）的攻击算法VGAE-MalGAN，目标是基于图的GNN Android恶意软件分类器。VGAE-MalGAN生成器生成对抗性恶意软件API图，而VGAE-MalGAN替代检测器尝试模仿目标检测器。实验结果表明，VGAE-MalGAN可以显着降低基于GNN的恶意软件分类器的检测率。尽管该模型最初未能检测到对抗性恶意软件，但使用生成的对抗性样本进行重新训练可以提高鲁棒性并有助于减轻对抗性攻击。



## **29. Conditional Adversarial Fragility in Financial Machine Learning under Macroeconomic Stress**

宏观经济压力下金融机器学习中的条件对抗脆弱性 cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19935v1) [paper-pdf](https://arxiv.org/pdf/2512.19935v1)

**Authors**: Samruddhi Baviskar

**Abstract**: Machine learning models used in financial decision systems operate in nonstationary economic environments, yet adversarial robustness is typically evaluated under static assumptions. This work introduces Conditional Adversarial Fragility, a regime dependent phenomenon in which adversarial vulnerability is systematically amplified during periods of macroeconomic stress. We propose a regime aware evaluation framework for time indexed tabular financial classification tasks that conditions robustness assessment on external indicators of economic stress. Using volatility based regime segmentation as a proxy for macroeconomic conditions, we evaluate model behavior across calm and stress periods while holding model architecture, attack methodology, and evaluation protocols constant. Baseline predictive performance remains comparable across regimes, indicating that economic stress alone does not induce inherent performance degradation. Under adversarial perturbations, however, models operating during stress regimes exhibit substantially greater degradation across predictive accuracy, operational decision thresholds, and risk sensitive outcomes. We further demonstrate that this amplification propagates to increased false negative rates, elevating the risk of missed high risk cases during adverse conditions. To complement numerical robustness metrics, we introduce an interpretive governance layer based on semantic auditing of model explanations using large language models. Together, these results demonstrate that adversarial robustness in financial machine learning is a regime dependent property and motivate stress aware approaches to model risk assessment in high stakes financial deployments.

摘要: 金融决策系统中使用的机器学习模型在非平稳经济环境中运行，但对抗稳健性通常是在静态假设下评估的。这项工作引入了条件对抗脆弱性，这是一种依赖政权的现象，其中对抗脆弱性在宏观经济压力时期被系统性放大。我们提出了一个用于时间索引表格财务分类任务的制度意识评估框架，该框架以经济压力的外部指标为条件进行稳健性评估。使用基于波动性的制度分割作为宏观经济状况的代理，我们评估平静和压力时期的模型行为，同时保持模型架构、攻击方法和评估协议不变。不同制度之间的基线预测性能保持可比性，这表明经济压力本身不会导致固有的性能下降。然而，在对抗性扰动下，在压力制度下运行的模型在预测准确性、操作决策阈值和风险敏感结果方面表现出明显更大的退化。我们进一步证明，这种放大会传播到假阴性率增加，从而增加了在不利条件下错过高风险病例的风险。为了补充数字稳健性指标，我们引入了一个基于使用大型语言模型对模型解释进行语义审计的解释治理层。总而言之，这些结果表明，金融机器学习中的对抗稳健性是一种依赖于制度的属性，并激励压力感知方法对高风险金融部署中的风险评估进行建模。



## **30. Multi-Layer Confidence Scoring for Detection of Out-of-Distribution Samples, Adversarial Attacks, and In-Distribution Misclassifications**

用于检测分布外样本、对抗性攻击和分布内错误分类的多层置信度评分 cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19472v1) [paper-pdf](https://arxiv.org/pdf/2512.19472v1)

**Authors**: Lorenzo Capelli, Leandro de Souza Rosa, Gianluca Setti, Mauro Mangia, Riccardo Rovatti

**Abstract**: The recent explosive growth in Deep Neural Networks applications raises concerns about the black-box usage of such models, with limited trasparency and trustworthiness in high-stakes domains, which have been crystallized as regulatory requirements such as the European Union Artificial Intelligence Act. While models with embedded confidence metrics have been proposed, such approaches cannot be applied to already existing models without retraining, limiting their broad application. On the other hand, post-hoc methods, which evaluate pre-trained models, focus on solving problems related to improving the confidence in the model's predictions, and detecting Out-Of-Distribution or Adversarial Attacks samples as independent applications. To tackle the limited applicability of already existing methods, we introduce Multi-Layer Analysis for Confidence Scoring (MACS), a unified post-hoc framework that analyzes intermediate activations to produce classification-maps. From the classification-maps, we derive a score applicable for confidence estimation, detecting distributional shifts and adversarial attacks, unifying the three problems in a common framework, and achieving performances that surpass the state-of-the-art approaches in our experiments with the VGG16 and ViTb16 models with a fraction of their computational overhead.

摘要: 深度神经网络应用程序最近的爆炸式增长引发了人们对此类模型黑匣子使用的担忧，因为在高风险领域的传输性和可信度有限，而这些已被具体化为欧盟人工智能法案等监管要求。虽然已经提出了具有嵌入置信指标的模型，但如果不进行重新培训，此类方法就无法应用于现有的模型，从而限制了其广泛应用。另一方面，评估预训练模型的事后方法专注于解决与提高模型预测的置信度相关的问题，并将分布外或对抗性攻击样本作为独立应用程序检测。为了解决现有方法的有限适用性问题，我们引入了置信度评分多层分析（MACS），这是一个统一的事后框架，可以分析中间激活以生成分类图。从分类图中，我们推导出适用于置信度估计、检测分布变化和对抗性攻击、将这三个问题统一在一个通用框架中，并在我们的实验中实现超越最先进方法的性能VGG 16和ViTb 16模型，其计算费用很小。



## **31. GShield: Mitigating Poisoning Attacks in Federated Learning**

GShield：减轻联邦学习中的中毒攻击 cs.CR

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2512.19286v2) [paper-pdf](https://arxiv.org/pdf/2512.19286v2)

**Authors**: Sameera K. M., Serena Nicolazzo, Antonino Nocera, Vinod P., Rafidha Rehiman K. A

**Abstract**: Federated Learning (FL) has recently emerged as a revolutionary approach to collaborative training Machine Learning models. In particular, it enables decentralized model training while preserving data privacy, but its distributed nature makes it highly vulnerable to a severe attack known as Data Poisoning. In such scenarios, malicious clients inject manipulated data into the training process, thereby degrading global model performance or causing targeted misclassification. In this paper, we present a novel defense mechanism called GShield, designed to detect and mitigate malicious and low-quality updates, especially under non-independent and identically distributed (non-IID) data scenarios. GShield operates by learning the distribution of benign gradients through clustering and Gaussian modeling during an initial round, enabling it to establish a reliable baseline of trusted client behavior. With this benign profile, GShield selectively aggregates only those updates that align with the expected gradient patterns, effectively isolating adversarial clients and preserving the integrity of the global model. An extensive experimental campaign demonstrates that our proposed defense significantly improves model robustness compared to the state-of-the-art methods while maintaining a high accuracy of performance across both tabular and image datasets. Furthermore, GShield improves the accuracy of the targeted class by 43\% to 65\% after detecting malicious and low-quality clients.

摘要: 联合学习（FL）最近成为协作训练机器学习模型的革命性方法。特别是，它能够实现去中心化模型训练，同时保护数据隐私，但其分布式性质使其极易受到称为数据中毒的严重攻击。在此类情况下，恶意客户端将操纵数据注入到训练过程中，从而降低全局模型性能或导致有针对性的错误分类。在本文中，我们提出了一种名为GShield的新型防御机制，旨在检测和减轻恶意和低质量更新，特别是在非独立和同分布（非IID）数据场景下。GShield通过在初始一轮期间通过集群和高斯建模学习良性梯度的分布来运作，使其能够建立可信客户行为的可靠基线。通过这种良性配置文件，GShield选择性地仅聚合那些与预期梯度模式一致的更新，从而有效地隔离敌对客户并保持全球模型的完整性。一项广泛的实验活动表明，与最先进的方法相比，我们提出的防御显着提高了模型的稳健性，同时在表格和图像数据集中保持了高准确性的性能。此外，GShield在检测到恶意和低质量客户端后将目标类的准确性提高了43%至65%。



## **32. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

有害在线内容的对抗稳健检测：计算设计科学方法 cs.LG

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.17367v3) [paper-pdf](https://arxiv.org/pdf/2512.17367v3)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.

摘要: 社交媒体平台受到仇恨言论、错误信息和极端主义言论等有害内容的困扰。机器学习（ML）模型被广泛采用来检测此类内容;然而，它们仍然极易受到对抗攻击，其中恶意用户会巧妙地修改文本以逃避检测。因此，增强对抗鲁棒性至关重要，需要检测器能够抵御各种攻击（可概括性），同时保持高的总体准确性。然而，同时实现最佳概括性和准确性是一项挑战。遵循计算设计科学范式，本研究采用顺序方法，首先提出了一种新颖的框架（基于大语言模型的样本生成和聚合，LLM-LGA），通过识别文本对抗攻击的关键不变性并利用它们来确保框架内实例化的检测器具有很强的概括性。其次，我们实例化我们的检测器（对抗鲁棒有害在线内容检测器，ARHOCD）具有三个新颖的设计组件来提高检测准确性：（1）利用其互补优势的多个基本检测器的集成;（2）一种新颖的权重分配方法，其基于每个样本的可预测性和每个碱基检测器的能力动态调整权重，权重使用领域知识初始化并通过Bayesian推理更新;以及（3）一种新颖的对抗训练策略，迭代优化基本检测器和权重分配器。我们解决了现有对抗鲁棒性增强研究的几个局限性，并在跨越仇恨言论、谣言和极端主义内容的三个数据集中对ARHOCD进行了实证评估。结果表明，ARHOCD具有很强的概括性，并提高了对抗条件下的检测准确性。



## **33. Machine Unlearning in Speech Emotion Recognition via Forget Set Alone**

通过Forget Set Alone实现语音情感识别的机器去学习 cs.SD

Submitted to ICASSP 2026

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2510.04251v2) [paper-pdf](https://arxiv.org/pdf/2510.04251v2)

**Authors**: Zhao Ren, Rathi Adarshi Rammohan, Kevin Scheck, Tanja Schultz

**Abstract**: Speech emotion recognition aims to identify emotional states from speech signals and has been widely applied in human-computer interaction, education, healthcare, and many other fields. However, since speech data contain rich sensitive information, partial data can be required to be deleted by speakers due to privacy concerns. Current machine unlearning approaches largely depend on data beyond the samples to be forgotten. However, this reliance poses challenges when data redistribution is restricted and demands substantial computational resources in the context of big data. We propose a novel adversarial-attack-based approach that fine-tunes a pre-trained speech emotion recognition model using only the data to be forgotten. The experimental results demonstrate that the proposed approach can effectively remove the knowledge of the data to be forgotten from the model, while preserving high model performance on the test set for emotion recognition.

摘要: 语音情感识别是从语音信号中识别情感状态的一种方法，在人机交互、教育、医疗等领域有着广泛的应用。然而，由于语音数据包含丰富的敏感信息，部分数据可能会被要求删除的发言者出于隐私问题。目前的机器学习方法在很大程度上依赖于被遗忘的样本之外的数据。然而，当数据再分配受到限制并且在大数据的背景下需要大量计算资源时，这种依赖带来了挑战。我们提出了一种新的基于对抗性攻击的方法，该方法只使用要忘记的数据来微调预训练的语音情感识别模型。实验结果表明，该方法能够有效地去除模型中待遗忘数据的知识，同时保持模型在情感识别测试集上的高性能.



## **34. Seeing Isn't Believing: Context-Aware Adversarial Patch Synthesis via Conditional GAN**

亲眼目睹并不可信：通过条件GAN的上下文感知对抗补丁合成 cs.CV

**SubmitDate**: 2025-12-27    [abs](http://arxiv.org/abs/2509.22836v2) [paper-pdf](https://arxiv.org/pdf/2509.22836v2)

**Authors**: Roie Kazoom, Alon Goldberg, Hodaya Cohen, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a severe threat to deep neural networks, yet most existing approaches rely on unrealistic white-box assumptions, untargeted objectives, or produce visually conspicuous patches that limit real-world applicability. In this work, we introduce a novel framework for fully controllable adversarial patch generation, where the attacker can freely choose both the input image x and the target class y target, thereby dictating the exact misclassification outcome. Our method combines a generative U-Net design with Grad-CAM-guided patch placement, enabling semantic-aware localization that maximizes attack effectiveness while preserving visual realism. Extensive experiments across convolutional networks (DenseNet-121, ResNet-50) and vision transformers (ViT-B/16, Swin-B/16, among others) demonstrate that our approach achieves state-of-the-art performance across all settings, with attack success rates (ASR) and target-class success (TCS) consistently exceeding 99%.   Importantly, we show that our method not only outperforms prior white-box attacks and untargeted baselines, but also surpasses existing non-realistic approaches that produce detectable artifacts. By simultaneously ensuring realism, targeted control, and black-box applicability-the three most challenging dimensions of patch-based attacks-our framework establishes a new benchmark for adversarial robustness research, bridging the gap between theoretical attack strength and practical stealthiness.

摘要: 对抗性补丁攻击对深度神经网络构成严重威胁，但大多数现有方法依赖于不切实际的白盒假设、无针对性的目标，或产生视觉上明显的补丁，从而限制了现实世界的适用性。在这项工作中，我们引入了一种新颖的框架，用于完全可控的对抗补丁生成，其中攻击者可以自由选择输入图像x和目标类别y目标，从而决定确切的误分类结果。我们的方法将生成式U-Net设计与Grad-CAM引导的补丁放置相结合，实现语义感知的本地化，从而最大限度地提高攻击效果，同时保持视觉真实感。跨卷积网络（DenseNet-121、ResNet-50）和视觉转换器（ViT-B/16、Swin-B/16等）的广泛实验表明，我们的方法在所有设置中都实现了最先进的性能，攻击成功率（ASB）和目标级成功率（TCS）始终超过99%。   重要的是，我们表明我们的方法不仅优于之前的白盒攻击和无针对性基线，而且还优于现有的产生可检测伪影的非现实方法。通过同时确保真实性、有针对性的控制和黑匣子适用性（基于补丁的攻击中最具挑战性的三个方面），我们的框架为对抗稳健性研究建立了新的基准，弥合了理论攻击强度和实际隐蔽性之间的差距。



## **35. Membership Inference Attack with Partial Features**

具有部分特征的成员推断攻击 cs.LG

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2508.06244v2) [paper-pdf](https://arxiv.org/pdf/2508.06244v2)

**Authors**: Xurun Wang, Guangrui Liu, Xinjie Li, Haoyu He, Lin Yao, Zhongyun Hua, Weizhe Zhang

**Abstract**: Machine learning models are vulnerable to membership inference attack, which can be used to determine whether a given sample appears in the training data. Most existing methods assume the attacker has full access to the features of the target sample. This assumption, however, does not hold in many real-world scenarios where only partial features are available, thereby limiting the applicability of these methods. In this work, we introduce Partial Feature Membership Inference (PFMI), a scenario where the adversary observes only partial features of each sample and aims to infer whether this observed subset was present in the training set. To address this problem, we propose MRAD (Memory-guided Reconstruction and Anomaly Detection), a two-stage attack framework that works in both white-box and black-box settings. In the first stage, MRAD leverages the latent memory of the target model to reconstruct the unknown features of the sample. We observe that when the known features are absent from the training set, the reconstructed sample deviates significantly from the true data distribution. Consequently, in the second stage, we use anomaly detection algorithms to measure the deviation between the reconstructed sample and the training data distribution, thereby determining whether the known features belong to a member of the training set. Empirical results demonstrate that MRAD is effective across various datasets, and maintains compatibility with off-the-shelf anomaly detection techniques. For example, on STL-10, our attack exceeds an AUC of around 0.75 even with 60% of the missing features.

摘要: 机器学习模型容易受到隶属度推理攻击，该攻击可用于确定给定样本是否出现在训练数据中。大多数现有方法都假设攻击者可以完全访问目标样本的特征。然而，这一假设在许多只有部分特征可用的现实世界场景中并不成立，从而限制了这些方法的适用性。在这项工作中，我们引入了部分特征隶属推理（PFMI），在这种情况下，对手仅观察每个样本的部分特征，并旨在推断该观察到的子集是否存在于训练集中。为了解决这个问题，我们提出了MRAD（内存引导重建和异常检测），这是一种两阶段攻击框架，适用于白盒和黑盒设置。在第一阶段，MRAD利用目标模型的潜在记忆来重建样本的未知特征。我们观察到，当训练集中缺乏已知特征时，重建的样本会显着偏离真实数据分布。因此，在第二阶段，我们使用异常检测算法来测量重建样本和训练数据分布之间的偏差，从而确定已知特征是否属于训练集的成员。经验结果表明，MRAD在各种数据集中都有效，并保持与现成的异常检测技术的兼容性。例如，在SPL-10上，即使有60%的特征缺失，我们的攻击也超过了约0.75的AUR。



## **36. Simulated Ensemble Attack: Transferring Jailbreaks Across Fine-tuned Vision-Language Models**

模拟集群攻击：通过微调的视觉语言模型转移越狱 cs.CV

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2508.01741v2) [paper-pdf](https://arxiv.org/pdf/2508.01741v2)

**Authors**: Ruofan Wang, Xin Wang, Yang Yao, Xuan Tong, Xingjun Ma

**Abstract**: Fine-tuning open-source Vision-Language Models (VLMs) creates a critical yet underexplored attack surface: vulnerabilities in the base VLM could be retained in fine-tuned variants, rendering them susceptible to transferable jailbreak attacks. To demonstrate this risk, we introduce the Simulated Ensemble Attack (SEA), a novel grey-box jailbreak method in which the adversary has full access to the base VLM but no knowledge of the fine-tuned target's weights or training configuration. To improve jailbreak transferability across fine-tuned VLMs, SEA combines two key techniques: Fine-tuning Trajectory Simulation (FTS) and Targeted Prompt Guidance (TPG). FTS generates transferable adversarial images by simulating the vision encoder's parameter shifts, while TPG is a textual strategy that steers the language decoder toward adversarially optimized outputs. Experiments on the Qwen2-VL family (2B and 7B) demonstrate that SEA achieves high transfer attack success rates exceeding 86.5% and toxicity rates near 49.5% across diverse fine-tuned variants, even those specifically fine-tuned to improve safety behaviors. Notably, while direct PGD-based image jailbreaks rarely transfer across fine-tuned VLMs, SEA reliably exploits inherited vulnerabilities from the base model, significantly enhancing transferability. These findings highlight an urgent need to safeguard fine-tuned proprietary VLMs against transferable vulnerabilities inherited from open-source foundations, motivating the development of holistic defenses across the entire model lifecycle.

摘要: 微调开源视觉语言模型（VLM）创建了一个关键但未充分探索的攻击表面：基础VLM中的漏洞可能会保留在微调的变体中，使其容易受到可转移的越狱攻击。为了证明这种风险，我们引入了模拟集群攻击（SEA），这是一种新型的灰箱越狱方法，其中对手可以完全访问基本VLM，但不知道微调目标的权重或训练配置。为了提高微调VLM之间的越狱转移性，SEA结合了两项关键技术：微调弹道模拟（FTS）和定向即时引导（TPG）。FTS通过模拟视觉编码器的参数变化来生成可转移的对抗图像，而TPG是一种文本策略，可以引导语言解码器朝着对抗优化的输出方向发展。Qwen 2-BL家族（2B和7 B）的实验表明，SEA在各种微调变体中实现了超过86.5%的高转移攻击成功率和接近49.5%的毒性率，即使是那些专门微调以改善安全行为的变体。值得注意的是，虽然直接基于PGD的图像越狱很少通过微调的VLM传输，但SEA可靠地利用了从基本模型继承的漏洞，显着增强了可传输性。这些发现凸显了迫切需要保护微调的专有VLM免受从开源基金会继承的可转移漏洞的影响，从而激励整个模型生命周期中的整体防御开发。



## **37. BeDKD: Backdoor Defense based on Dynamic Knowledge Distillation and Directional Mapping Modulator**

BeDKD：基于动态知识蒸馏和方向映射调制器的后门防御 cs.CR

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2508.01595v2) [paper-pdf](https://arxiv.org/pdf/2508.01595v2)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Yinghan Zhou, Changtong dou, Yiming Xue

**Abstract**: Although existing backdoor defenses have gained success in mitigating backdoor attacks, they still face substantial challenges. In particular, most of them rely on large amounts of clean data to weaken the backdoor mapping but generally struggle with residual trigger effects, resulting in persistently high attack success rates (ASR). Therefore, in this paper, we propose a novel Backdoor defense method based on Directional mapping module and adversarial Knowledge Distillation (BeDKD), which balances the trade-off between defense effectiveness and model performance using a small amount of clean and poisoned data. We first introduce a directional mapping module to identify poisoned data, which destroys clean mapping while keeping backdoor mapping on a small set of flipped clean data. Then, the adversarial knowledge distillation is designed to reinforce clean mapping and suppress backdoor mapping through a cycle iteration mechanism between trust and punish distillations using clean and identified poisoned data. We conduct experiments to mitigate mainstream attacks on three datasets, and experimental results demonstrate that BeDKD surpasses the state-of-the-art defenses and reduces the ASR by 98% without significantly reducing the CACC. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD.

摘要: 尽管现有的后门防御措施在缓解后门攻击方面取得了成功，但它们仍然面临着巨大的挑战。特别是，它们中的大多数依赖大量干净的数据来削弱后门映射，但通常会与残余触发效应作斗争，从而导致攻击成功率（ASB）持续很高。因此，本文提出了一种基于方向映射模块和对抗性知识蒸馏（BeDKD）的新型后门防御方法，该方法使用少量干净和有毒数据来平衡防御有效性和模型性能之间的权衡。我们首先引入一个方向性映射模块来识别有毒数据，这会破坏干净映射，同时在一小群翻转干净数据上保留后门映射。然后，对抗性知识蒸馏旨在通过信任和惩罚蒸馏之间的循环迭代机制来加强干净映射并抑制后门映射，使用干净和已识别的有毒数据。我们进行了实验来缓解对三个数据集的主流攻击，实验结果表明，BeDKD超越了最先进的防御能力，并在不显着降低CACC的情况下将ASB降低了98%。我们的代码可在https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD上获取。



## **38. Improving Large Language Model Safety with Contrastive Representation Learning**

通过对比表示学习提高大型语言模型安全性 cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2506.11938v2) [paper-pdf](https://arxiv.org/pdf/2506.11938v2)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense

摘要: 大型语言模型（LLM）是具有深远社会影响的强大工具，但它们对各种不受控制的输入产生响应的能力使它们容易受到对抗性攻击。虽然现有的防御通常很难概括不同的攻击类型，但表示工程的最新进展提供了有希望的替代方案。在这项工作中，我们提出了一个防御框架，制定模型防御作为一个对比表示学习（CRL）的问题。我们的方法使用基于三元组的损失结合对抗性硬负面挖掘来微调模型，以鼓励良性和有害表示之间的分离。我们跨多个模型的实验结果表明，我们的方法优于基于先验表示工程的防御，在不损害标准性能的情况下提高了针对输入级和嵌入空间攻击的鲁棒性。我们的代码可在https://github.com/samuelsimko/crl-llm-defense上获取



## **39. SoK: Are Watermarks in LLMs Ready for Deployment?**

SoK：LLM中的水印准备好部署了吗？ cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2506.05594v3) [paper-pdf](https://arxiv.org/pdf/2506.05594v3)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah, My T. Thai

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs.   To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.

摘要: 大型语言模型（LLM）改变了自然语言处理，在不同任务中展示了令人印象深刻的能力。然而，部署这些模型会带来与知识产权侵犯和潜在滥用相关的关键风险，特别是因为对手可以模仿这些模型来窃取服务或产生误导性输出。我们特别关注模型窃取攻击，因为它们与专有LLM高度相关，并对其安全、收入和道德部署构成严重威胁。虽然已经出现了各种水印技术来减轻这些风险，但目前尚不清楚社区和行业在LLM中开发和部署水印方面取得了多大进展。   为了弥合这一差距，我们的目标是通过1）提供LLM中水印的详细分类，2）提出一种新型知识产权分类器来探索水印在攻击和无攻击环境下的有效性和影响，3）分析LLM中现有水印的局限性，4）讨论LLM中水印的实际挑战和潜在的未来方向。通过广泛的实验，我们表明，尽管研究成果令人鼓舞，领先公司和社区也对部署水印给予了极大的关注，但由于这些技术对LLM和下游任务的模型效用产生不利影响，这些技术尚未在现实世界应用中充分发挥潜力。我们的研究结果提供了对LLM中的水印的深刻理解，强调了针对LLM部署量身定制的实用水印解决方案的必要性。



## **40. Towards Dataset Copyright Evasion Attack against Personalized Text-to-Image Diffusion Models**

针对个性化文本到图像扩散模型的数据集版权规避攻击 cs.CV

Accepted by IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2505.02824v2) [paper-pdf](https://arxiv.org/pdf/2505.02824v2)

**Authors**: Kuofeng Gao, Yufei Zhu, Yiming Li, Jiawang Bai, Yong Yang, Zhifeng Li, Shu-Tao Xia

**Abstract**: Text-to-image (T2I) diffusion models enable high-quality image generation conditioned on textual prompts. However, fine-tuning these pre-trained models for personalization raises concerns about unauthorized dataset usage. To address this issue, dataset ownership verification (DOV) has recently been proposed, which embeds watermarks into fine-tuning datasets via backdoor techniques. These watermarks remain dormant on benign samples but produce owner-specified outputs when triggered. Despite its promise, the robustness of DOV against copyright evasion attacks (CEA) remains unexplored. In this paper, we investigate how adversaries can circumvent these mechanisms, enabling models trained on watermarked datasets to bypass ownership verification. We begin by analyzing the limitations of potential attacks achieved by backdoor removal, including TPD and T2IShield. In practice, TPD suffers from inconsistent effectiveness due to randomness, while T2IShield fails when watermarks are embedded as local image patches. To this end, we introduce CEAT2I, the first CEA specifically targeting DOV in T2I diffusion models. CEAT2I consists of three stages: (1) motivated by the observation that T2I models converge faster on watermarked samples with respect to intermediate features rather than training loss, we reliably detect watermarked samples; (2) we iteratively ablate tokens from the prompts of detected samples and monitor feature shifts to identify trigger tokens; and (3) we apply a closed-form concept erasure method to remove the injected watermarks. Extensive experiments demonstrate that CEAT2I effectively evades state-of-the-art DOV mechanisms while preserving model performance. The code is available at https://github.com/csyufei/CEAT2I.

摘要: 文本到图像（T2 I）扩散模型能够根据文本提示生成高质量图像。然而，微调这些预先训练的模型以实现个性化会引发人们对未经授权的数据集使用的担忧。为了解决这个问题，最近提出了数据集所有权验证（DOV），通过后门技术将水印嵌入到微调数据集中。这些水印在良性样本上保持休眠状态，但在触发时会产生所有者指定的输出。尽管DOV有着承诺，但其针对版权规避攻击（CAE）的稳健性仍然有待探索。在本文中，我们研究对手如何绕过这些机制，使在带水印数据集上训练的模型能够绕过所有权验证。我们首先分析后门删除所实现的潜在攻击的局限性，包括DPD和T2 IShield。在实践中，DPD由于随机性而导致有效性不一致，而T2 IShield则在水印作为本地图像补丁嵌入时失败。为此，我们引入了CEAT 2 I，这是T2 I扩散模型中第一个专门针对DOV的CAE。CEAT 2 I由三个阶段组成：（1）由于观察到T2 I模型在中间特征方面更快地收敛在带水印样本上，而不是训练损失，因此我们可靠地检测带水印样本;（2）我们从检测到的样本的提示中迭代地消融令牌并监控特征变化以识别触发令牌;以及（3）我们应用封闭形式的概念擦除方法来去除注入的水印。大量实验表明，CEAT 2 I有效地规避了最先进的DOV机制，同时保留了模型性能。该代码可在https://github.com/csyufei/CEAT2I上获取。



## **41. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

AutoAdv：大型语言模型多回合越狱的自动对抗预算 cs.CR

We encountered issues with the paper being hosted under my personal account, so we republished it under a different account associated with a university email, which makes updates and management easier. As a result, this version is a duplicate of arXiv:2511.02376

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2507.01020v2) [paper-pdf](https://arxiv.org/pdf/2507.01020v2)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities to jailbreaking attacks: carefully crafted malicious inputs intended to circumvent safety guardrails and elicit harmful responses. As such, we present AutoAdv, a novel framework that automates adversarial prompt generation to systematically evaluate and expose vulnerabilities in LLM safety mechanisms. Our approach leverages a parametric attacker LLM to produce semantically disguised malicious prompts through strategic rewriting techniques, specialized system prompts, and optimized hyperparameter configurations. The primary contribution of our work is a dynamic, multi-turn attack methodology that analyzes failed jailbreak attempts and iteratively generates refined follow-up prompts, leveraging techniques such as roleplaying, misdirection, and contextual manipulation. We quantitatively evaluate attack success rate (ASR) using the StrongREJECT (arXiv:2402.10260 [cs.CL]) framework across sequential interaction turns. Through extensive empirical evaluation of state-of-the-art models--including ChatGPT, Llama, and DeepSeek--we reveal significant vulnerabilities, with our automated attacks achieving jailbreak success rates of up to 86% for harmful content generation. Our findings reveal that current safety mechanisms remain susceptible to sophisticated multi-turn attacks, emphasizing the urgent need for more robust defense strategies.

摘要: 大型语言模型（LLM）继续表现出越狱攻击的漏洞：精心设计的恶意输入，旨在绕过安全护栏并引发有害响应。因此，我们提出了AutoAdv，这是一个新颖的框架，可以自动生成对抗提示，以系统地评估和暴露LLM安全机制中的漏洞。我们的方法利用参数攻击者LLM通过战略重写技术、专门的系统提示和优化的超参数配置来产生语义伪装的恶意提示。我们工作的主要贡献是一种动态、多回合攻击方法，该方法分析失败的越狱尝试，并利用角色扮演、误导和上下文操纵等技术迭代生成细化的后续提示。我们使用StrongRESYS（arXiv：2402.10260 [cs.CL]）框架在连续交互回合中量化评估攻击成功率（ASB）。通过对最先进模型（包括ChatGPT、Llama和DeepSeek）进行广泛的实证评估，我们揭示了重大漏洞，我们的自动攻击在有害内容生成方面实现了高达86%的越狱成功率。我们的研究结果表明，当前的安全机制仍然容易受到复杂的多回合攻击，这凸显了对更强大的防御策略的迫切需要。



## **42. CAE-Net: Generalized Deepfake Image Detection using Convolution and Attention Mechanisms with Spatial and Frequency Domain Features**

CAE-Net：使用具有空间和频域特征的卷积和注意力机制的广义Deepfake图像检测 cs.CV

Published in Journal of Visual Communication and Image Representation

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2502.10682v3) [paper-pdf](https://arxiv.org/pdf/2502.10682v3)

**Authors**: Anindya Bhattacharjee, Kaidul Islam, Kafi Anan, Ashir Intesher, Abrar Assaeem Fuad, Utsab Saha, Hafiz Imtiaz

**Abstract**: The spread of deepfakes poses significant security concerns, demanding reliable detection methods. However, diverse generation techniques and class imbalance in datasets create challenges. We propose CAE-Net, a Convolution- and Attention-based weighted Ensemble network combining spatial and frequency-domain features for effective deepfake detection. The architecture integrates EfficientNet, Data-Efficient Image Transformer (DeiT), and ConvNeXt with wavelet features to learn complementary representations. We evaluated CAE-Net on the diverse IEEE Signal Processing Cup 2025 (DF-Wild Cup) dataset, which has a 5:1 fake-to-real class imbalance. To address this, we introduce a multistage disjoint-subset training strategy, sequentially training the model on non-overlapping subsets of the fake class while retaining knowledge across stages. Our approach achieved $94.46\%$ accuracy and a $97.60\%$ AUC, outperforming conventional class-balancing methods. Visualizations confirm the network focuses on meaningful facial regions, and our ensemble design demonstrates robustness against adversarial attacks, positioning CAE-Net as a dependable and generalized deepfake detection framework.

摘要: Deepfakes的传播带来了重大的安全问题，需要可靠的检测方法。然而，多样化的生成技术和数据集的类别不平衡带来了挑战。我们提出CAE-Net，这是一种基于卷积和注意力的加权Ensemble网络，结合了空间和频域特征，用于有效的深度伪造检测。该架构将EfficientNet、数据高效图像Transformer（DeiT）和ConvNeXt与子波功能集成，以学习补充表示。我们在不同的IEEE Signal Process Cup 2025（DF-Wild Cup）数据集上评估了CAE-Net，该数据集的真实与真实类别失衡为5：1。为了解决这个问题，我们引入了一种多阶段不相交子集训练策略，在假类的非重叠子集上顺序训练模型，同时保留跨阶段的知识。我们的方法实现了94.46美元的准确性和97.60美元的UC，优于传统的类别平衡方法。可视化证实该网络专注于有意义的面部区域，我们的整体设计展示了针对对抗攻击的鲁棒性，将CAE-Net定位为可靠且通用的Deepfake检测框架。



## **43. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

Accepted by USENIX Security 2025

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2502.01386v3) [paper-pdf](https://arxiv.org/pdf/2502.01386v3)

**Authors**: Yuyang Gong, Zhuo Chen, Jiawei Liu, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型（LLM）的检索增强生成（RAG）系统对于问答和内容生成等任务来说已变得至关重要。然而，由于其固有的漏洞，它们对公众舆论和信息传播的影响越来越大，使它们成为安全研究的关键焦点。之前的研究主要针对针对事实或单一查询操纵的攻击。在本文中，我们讨论了一个更实际的场景：对RAG模型的面向主题的对抗性意见操纵攻击，其中LLM需要推理和综合多个观点，使其特别容易受到系统性知识中毒的影响。具体来说，我们提出了Topic-FlipRAG，这是一种两阶段操纵攻击管道，可以战略性地制造对抗性扰动，以影响相关查询的意见。该方法结合了传统的对抗性排名攻击技术，并利用LLM的广泛内部相关知识和推理能力来执行语义级别的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显着影响用户信息感知。当前的缓解方法无法有效防御此类攻击，这凸显了加强RAG系统保护措施的必要性，并为LLM安全研究提供了重要见解。



## **44. When Should Selfish Miners Double-Spend?**

什么时候自私的矿工应该加倍花钱？ cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2501.03227v4) [paper-pdf](https://arxiv.org/pdf/2501.03227v4)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Conventional double-spending attack models ignore the revenue losses stemming from the orphan blocks. On the other hand, selfish mining literature usually ignores the chance of the attacker to double-spend at no-cost in each attack cycle. In this paper, we give a rigorous stochastic analysis of an attack where the goal of the adversary is to double-spend while mining selfishly. To do so, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, the adversary gets a free shot at double-spending. At each cycle, for a given stubbornness level, we rigorously formulate how great the probability of double-spending is. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability.

摘要: 传统的双重支出攻击模型忽略了孤儿区块带来的收入损失。另一方面，自私的采矿文献通常忽视攻击者在每个攻击周期中免费重复支出的机会。在本文中，我们给出了一个严格的随机分析的攻击对手的目标是双花，而自私地挖掘。为此，我们首先结合顽固和自私的采矿攻击，即构建一个策略，让攻击者表现得顽固，直到其私人分支达到一定长度，然后转向自私。我们为每个参数制度提供最佳的确定性。接下来，我们提供了仍然比诚实采矿更有利可图的最大顽固度，并论证了顽固度水平与$k$-确认规则之间的联系。我们表明，在每个攻击周期中，如果顽固程度高于$k$，对手就可以免费获得双重消费的机会。在每个周期中，对于给定的顽固度水平，我们严格制定双重消费的可能性有多大。我们进一步修改顽固政权中的攻击，以隐藏攻击并增加双重消费的概率。



## **45. FlippedRAG: Black-Box Opinion Manipulation Adversarial Attacks to Retrieval-Augmented Generation Models**

FlippedRAG：黑盒意见操纵对抗性攻击检索增强生成模型 cs.IR

Accepted by 32nd ACM Conference on Computer and Communications Security (ACM CCS 2025)

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2501.02968v6) [paper-pdf](https://arxiv.org/pdf/2501.02968v6)

**Authors**: Zhuo Chen, Yuyang Gong, Jiawei Liu, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) enriches LLMs by dynamically retrieving external knowledge, reducing hallucinations and satisfying real-time information needs. While existing research mainly targets RAG's performance and efficiency, emerging studies highlight critical security concerns. Yet, current adversarial approaches remain limited, mostly addressing white-box scenarios or heuristic black-box attacks without fully investigating vulnerabilities in the retrieval phase. Additionally, prior works mainly focus on factoid Q&A tasks, their attacks lack complexity and can be easily corrected by advanced LLMs. In this paper, we investigate a more realistic and critical threat scenario: adversarial attacks intended for opinion manipulation against black-box RAG models, particularly on controversial topics. Specifically, we propose FlippedRAG, a transfer-based adversarial attack against black-box RAG systems. We first demonstrate that the underlying retriever of a black-box RAG system can be reverse-engineered, enabling us to train a surrogate retriever. Leveraging the surrogate retriever, we further craft target poisoning triggers, altering vary few documents to effectively manipulate both retrieval and subsequent generation. Extensive empirical results show that FlippedRAG substantially outperforms baseline methods, improving the average attack success rate by 16.7%. FlippedRAG achieves on average a 50% directional shift in the opinion polarity of RAG-generated responses, ultimately causing a notable 20% shift in user cognition. Furthermore, we evaluate the performance of several potential defensive measures, concluding that existing mitigation strategies remain insufficient against such sophisticated manipulation attacks. These results highlight an urgent need for developing innovative defensive solutions to ensure the security and trustworthiness of RAG systems.

摘要: 检索增强生成（RAG）通过动态检索外部知识、减少幻觉和满足实时信息需求来丰富LLM。虽然现有研究主要针对RAG的性能和效率，但新出现的研究强调了关键的安全问题。然而，当前的对抗方法仍然有限，主要解决白盒场景或启发式黑匣子攻击，而没有在检索阶段充分调查漏洞。此外，之前的作品主要集中在事实问答任务上，其攻击缺乏复杂性，并且可以通过高级LLM轻松纠正。在本文中，我们研究了一个更现实和关键的威胁场景：针对黑箱RAG模型的意见操纵的对抗性攻击，特别是在有争议的话题上。具体来说，我们提出了FlippedRAG，这是一种针对黑盒RAG系统的基于传输的对抗性攻击。我们首先证明了一个黑盒RAG系统的底层检索器可以进行逆向工程，使我们能够训练一个代理检索器。利用代理检索器，我们进一步工艺目标中毒触发器，改变不同的几个文件，以有效地操纵检索和后续生成。广泛的实证结果表明，FlippedRAG的性能大大优于基线方法，将平均攻击成功率提高了16.7%。FlippedRAG平均实现了RAG生成的响应的意见两极50%的方向性转变，最终导致用户认知发生了20%的显着转变。此外，我们评估了几种潜在防御措施的性能，得出的结论是，现有的缓解策略仍然不足以应对此类复杂的操纵攻击。这些结果凸显了开发创新防御解决方案的迫切需要，以确保RAG系统的安全性和可信性。



## **46. Quantifying True Robustness: Synonymity-Weighted Similarity for Trustworthy XAI Evaluation**

量化真正稳健性：可信赖XAI评估的同义加权相似性 cs.LG

10 pages, 2 figures, 6 tables. Changes to title, abstract and minor edits to the content as a result of acceptance to the 59th Hawaii International Conference on System Sciences

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2501.01516v2) [paper-pdf](https://arxiv.org/pdf/2501.01516v2)

**Authors**: Christopher Burger

**Abstract**: Adversarial attacks challenge the reliability of Explainable AI (XAI) by altering explanations while the model's output remains unchanged. The success of these attacks on text-based XAI is often judged using standard information retrieval metrics. We argue these measures are poorly suited in the evaluation of trustworthiness, as they treat all word perturbations equally while ignoring synonymity, which can misrepresent an attack's true impact. To address this, we apply synonymity weighting, a method that amends these measures by incorporating the semantic similarity of perturbed words. This produces more accurate vulnerability assessments and provides an important tool for assessing the robustness of AI systems. Our approach prevents the overestimation of attack success, leading to a more faithful understanding of an XAI system's true resilience against adversarial manipulation.

摘要: 对抗性攻击通过改变解释而模型的输出保持不变，挑战了可解释人工智能（XAI）的可靠性。通常使用标准信息检索指标来判断这些对基于文本的XAI的攻击是否成功。我们认为，这些措施不适合评估可信度，因为它们平等地对待所有单词扰动，而忽略了同义性，这可能会歪曲攻击的真实影响。为了解决这个问题，我们应用了同义加权，这是一种通过合并受干扰单词的语义相似性来修改这些测量的方法。这可以产生更准确的漏洞评估，并为评估人工智能系统的稳健性提供了重要工具。我们的方法可以防止高估攻击成功，从而更忠实地了解XAI系统对抗对抗操纵的真正弹性。



## **47. Improving Graph Neural Network Training, Defense and Hypergraph Partitioning via Adversarial Robustness Evaluation**

通过对抗稳健性评估改进图神经网络训练、防御和超图划分 cs.LG

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2412.14738v10) [paper-pdf](https://arxiv.org/pdf/2412.14738v10)

**Authors**: Yongyu Wang

**Abstract**: Graph Neural Networks (GNNs) are a highly effective neural network architecture for processing graph-structured data. Unlike traditional neural networks that rely solely on the features of the data as input, GNNs leverage both the graph structure, which represents the relationships between data points, and the feature matrix of the data to optimize their feature representation. This unique capability enables GNNs to achieve superior performance across various tasks. However, it also makes GNNs more susceptible to noise and adversarial attacks from both the graph structure and data features, which can significantly increase the training difficulty and degrade their performance. Similarly, a hypergraph is a highly complex structure, and partitioning a hypergraph is a challenging task. This paper leverages spectral adversarial robustness evaluation to effectively address key challenges in complex-graph algorithms. By using spectral adversarial robustness evaluation to distinguish robust nodes from non-robust ones and treating them differently, we propose a training-set construction strategy that improves the training quality of GNNs. In addition, we develop algorithms to enhance both the adversarial robustness of GNNs and the performance of hypergraph partitioning. Experimental results show that this series of methods is highly effective.

摘要: 图神经网络（GNN）是一种用于处理图结构数据的高效神经网络架构。与仅依赖数据特征作为输入的传统神经网络不同，GNN利用表示数据点之间关系的图结构和数据的特征矩阵来优化其特征表示。这种独特的功能使GNN能够在各种任务中实现卓越的性能。然而，它也使GNN更容易受到来自图结构和数据特征的噪音和对抗攻击，这可能会显着增加训练难度并降低其性能。同样，超图是一种高度复杂的结构，划分超图是一项具有挑战性的任务。本文利用谱对抗鲁棒性评估来有效解决复杂图算法中的关键挑战。通过使用谱对抗鲁棒性评估来区分鲁棒节点和非鲁棒节点并区别对待它们，我们提出了一种提高GNN训练质量的训练集构建策略。此外，我们还开发了算法来增强GNN的对抗鲁棒性和超图分区的性能。实验结果表明，该系列方法非常有效。



## **48. Trust-free Personalized Decentralized Learning**

无需信任的个性化分散学习 cs.LG

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2410.11378v2) [paper-pdf](https://arxiv.org/pdf/2410.11378v2)

**Authors**: Yawen Li, Yan Li, Junping Du, Yingxia Shao, Meiyu Liang, Guanhua Ye

**Abstract**: Personalized collaborative learning in federated settings faces a critical trade-off between customization and participant trust. Existing approaches typically rely on centralized coordinators or trusted peer groups, limiting their applicability in open, trust-averse environments. While recent decentralized methods explore anonymous knowledge sharing, they often lack global scalability and robust mechanisms against malicious peers. To bridge this gap, we propose TPFed, a \textit{Trust-free Personalized Decentralized Federated Learning} framework. TPFed replaces central aggregators with a blockchain-based bulletin board, enabling participants to dynamically select global communication partners based on Locality-Sensitive Hashing (LSH) and peer ranking. Crucially, we introduce an ``all-in-one'' knowledge distillation protocol that simultaneously handles knowledge transfer, model quality evaluation, and similarity verification via a public reference dataset. This design ensures secure, globally personalized collaboration without exposing local models or data. Extensive experiments demonstrate that TPFed significantly outperforms traditional federated baselines in both learning accuracy and system robustness against adversarial attacks.

摘要: 联邦环境中的个性化协作学习面临着定制和参与者信任之间的关键权衡。现有的方法通常依赖于集中式协调员或受信任的同侪团体，限制了它们在开放、厌恶信任的环境中的适用性。虽然最近的去中心化方法探索了匿名知识共享，但它们通常缺乏全球可扩展性和针对恶意对等点的强大机制。为了弥合这一差距，我们提出了TPFed，这是一个\textit{Trust-free Personalized Decentralized Federated Learning}框架。TPFed用基于区块链的公告板取代了中央聚合器，使参与者能够根据本地敏感哈希（LSH）和对等排名动态选择全球通信合作伙伴。至关重要的是，我们引入了一个“一体化”的知识蒸馏协议，同时处理知识转移，模型质量评估，并通过公共参考数据集的相似性验证。这种设计确保了安全、全球个性化的协作，而不会暴露本地模型或数据。大量的实验表明，TPFed在学习准确性和系统对对抗性攻击的鲁棒性方面都显着优于传统的联邦基线。



## **49. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

一个扰动就足够了：关于针对视觉语言预训练模型生成普遍对抗性扰动 cs.CV

Accepted by ICCV-2025

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2406.05491v4) [paper-pdf](https://arxiv.org/pdf/2406.05491v4)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Hao Wu, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these methods are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also vulnerable to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC) to achieve the attack. In light that the pivotal multimodal alignment is achieved through the advanced contrastive learning technique, we devise to turn this powerful weapon against themselves, i.e., employ a malicious version of contrastive learning to train the C-PGC based on our carefully crafted positive and negative image-text pairs for essentially destroying the alignment relationship learned by VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus essentially enhancing attacks across various victim models and V+L tasks. The GitHub repository is available at https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.

摘要: 通过充分利用多模式对齐，视觉语言预训练（VLP）模型在许多应用中展现出前所未有的能力。然而，之前的研究表明，它们很容易受到恶意制作的对抗样本的影响。尽管最近取得了成功，但这些方法通常是特定于实例的，并且需要为每个输入样本生成扰动。在本文中，我们揭示了VLP模型也容易受到实例不可知的普遍对抗扰动（UAP）的影响。具体来说，我们设计了一种新颖的具有跨模式条件的对比训练扰动生成器（C-PGC）来实现攻击。鉴于关键的多模式对齐是通过先进的对比学习技术实现的，我们设计将这一强大的武器用来对抗自己，即使用恶意版本的对比学习来基于我们精心制作的正和负图像-文本对来训练C-PGC，以本质上破坏VLP模型学习的对齐关系。此外，C-PGC充分利用视觉与语言（V+L）场景的特点，整合单模式和跨模式信息作为有效指导。大量实验表明，C-PGC成功迫使对抗样本离开VLP模型特征空间中的原始区域，从而从本质上增强了对各种受害者模型和V+L任务的攻击。GitHub存储库可访问https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks。



## **50. Achieving Dalenius' Goal of Data Privacy with Practical Assumptions**

通过实际假设实现Dalenius的数据隐私目标 cs.CR

50 pages

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/1703.07474v6) [paper-pdf](https://arxiv.org/pdf/1703.07474v6)

**Authors**: Genqiang Wu, Xianyao Xia, Yeping He

**Abstract**: Current differential privacy frameworks face significant challenges: vulnerability to correlated data attacks and suboptimal utility-privacy tradeoffs. To address these limitations, we establish a novel information-theoretic foundation for Dalenius' privacy vision using Shannon's perfect secrecy framework. By leveraging the fundamental distinction between cryptographic systems (small secret keys) and privacy mechanisms (massive datasets), we replace differential privacy's restrictive independence assumption with practical partial knowledge constraints ($H(X) \geq b$).   We propose an information privacy framework achieving Dalenius security with quantifiable utility-privacy tradeoffs. Crucially, we prove that foundational mechanisms -- random response, exponential, and Gaussian channels -- satisfy Dalenius' requirements while preserving group privacy and composition properties. Our channel capacity analysis reduces infinite-dimensional evaluations to finite convex optimizations, enabling direct application of information-theoretic tools.   Empirical evaluation demonstrates that individual channel capacity (maximal information leakage of each individual) decreases with increasing entropy constraint $b$, and our framework achieves superior utility-privacy tradeoffs compared to classical differential privacy mechanisms under equivalent privacy guarantees. The framework is extended to computationally bounded adversaries via Yao's theory, unifying cryptographic and statistical privacy paradigms. Collectively, these contributions provide a theoretically grounded path toward practical, composable privacy -- subject to future resolution of the tradeoff characterization -- with enhanced resilience to correlation attacks.

摘要: 当前的差异隐私框架面临着重大挑战：容易受到相关数据攻击以及次优的公用事业与隐私权衡。为了解决这些局限性，我们使用香农的完美保密框架为Dalenius的隐私愿景建立了一个新颖的信息理论基础。通过利用加密系统（小秘密密钥）和隐私机制（大规模数据集）之间的根本区别，我们用实际的部分知识约束（$H（X）\geq b$）取代差异隐私的限制性独立性假设。   我们提出了一个信息隐私框架，通过可量化的公用事业与隐私权衡来实现Dalenius安全。至关重要的是，我们证明了基本机制--随机响应、指数和高斯通道--可以满足Dalenius的要求，同时保留组隐私和组合属性。我们的渠道容量分析将无限维评估简化为有限凸优化，从而能够直接应用信息论工具。   经验评估表明，个人通道容量（每个人的最大信息泄露）随着信息量约束$b$的增加而减少，并且与等效隐私保证下的经典差异隐私机制相比，我们的框架实现了更好的实用性-隐私权衡。该框架通过Yao的理论扩展到计算有界限的对手，统一了加密和统计隐私范式。总的来说，这些贡献为实现实用的、可组合的隐私提供了一条理论上的基础路径--取决于权衡特征的未来解决方案--并增强了对相关性攻击的弹性。



