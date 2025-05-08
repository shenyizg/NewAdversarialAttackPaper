# Latest Adversarial Attack Papers
**update at 2025-05-08 18:51:16**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization**

以毒攻毒：通过奖励中和防御恶意RL微调 cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04578v1) [paper-pdf](http://arxiv.org/pdf/2505.04578v1)

**Authors**: Wenjun Cao

**Abstract**: Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models.

摘要: 强化学习（RL）微调改变了大型语言模型，同时创建了我们实验验证的漏洞：我们的实验表明，恶意RL微调以显着的效率突破了安全护栏，只需要50个步骤和最少的对抗提示，有害的升级从0-2升级到7-9。这种攻击载体特别威胁具有参数级访问权限的开源模型。事实证明，针对监督式微调的现有防御措施对RL的动态反馈机制无效。我们引入了奖励中和，这是第一个专门针对RL微调攻击而设计的防御框架，建立了简洁的拒绝模式，使恶意奖励信号无效。我们的方法训练模型以产生攻击者无法利用的最小信息拒绝，系统性地抵消针对有害输出进行优化的尝试。实验验证了我们的方法在200次攻击步骤后保持较低的有害分数（不大于2），而标准模型迅速恶化。这项工作提供了第一个建设性的证据，证明可以实现针对日益容易获得的RL攻击的强大防御，解决了开权模型的关键安全差距。



## **2. Mitigating Many-Shot Jailbreaking**

减轻多枪越狱 cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2504.09604v2) [paper-pdf](http://arxiv.org/pdf/2504.09604v2)

**Authors**: Christopher M. Ackerman, Nina Panickssery

**Abstract**: Many-shot jailbreaking (MSJ) is an adversarial technique that exploits the long context windows of modern LLMs to circumvent model safety training by including in the prompt many examples of a "fake" assistant responding inappropriately before the final request. With enough examples, the model's in-context learning abilities override its safety training, and it responds as if it were the "fake" assistant. In this work, we probe the effectiveness of different fine-tuning and input sanitization approaches on mitigating MSJ attacks, alone and in combination. We find incremental mitigation effectiveness for each, and show that the combined techniques significantly reduce the effectiveness of MSJ attacks, while retaining model performance in benign in-context learning and conversational tasks. We suggest that our approach could meaningfully ameliorate this vulnerability if incorporated into model safety post-training.

摘要: 多镜头越狱（MSJ）是一种对抗性技术，它利用现代LLM的长上下文窗口来规避模型安全培训，方法是在提示中包含许多“假”助理在最终请求之前做出不当反应的示例。有了足够多的例子，该模型的上下文学习能力就会凌驾于其安全培训之上，并且它的反应就好像它是“假”助手一样。在这项工作中，我们探讨了不同的微调和输入清理方法单独和组合在减轻MSJ攻击方面的有效性。我们发现每种技术的增量缓解效果，并表明组合技术显着降低了MSJ攻击的有效性，同时保留了良性上下文学习和对话任务中的模型性能。我们认为，如果将我们的方法纳入模型安全培训后，可以有意义地改善这种脆弱性。



## **3. Machine Learning Cryptanalysis of a Quantum Random Number Generator**

量子随机数发生器的机器学习密码分析 cs.LG

Published article is at https://ieeexplore.ieee.org/document/8396276.  Related code is at  https://github.com/Nano-Neuro-Research-Lab/Machine-Learning-Cryptanalysis-of-a-Quantum-Random-Number-Generator

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/1905.02342v3) [paper-pdf](http://arxiv.org/pdf/1905.02342v3)

**Authors**: Nhan Duy Truong, Jing Yan Haw, Syed Muhamad Assad, Ping Koy Lam, Omid Kavehei

**Abstract**: Random number generators (RNGs) that are crucial for cryptographic applications have been the subject of adversarial attacks. These attacks exploit environmental information to predict generated random numbers that are supposed to be truly random and unpredictable. Though quantum random number generators (QRNGs) are based on the intrinsic indeterministic nature of quantum properties, the presence of classical noise in the measurement process compromises the integrity of a QRNG. In this paper, we develop a predictive machine learning (ML) analysis to investigate the impact of deterministic classical noise in different stages of an optical continuous variable QRNG. Our ML model successfully detects inherent correlations when the deterministic noise sources are prominent. After appropriate filtering and randomness extraction processes are introduced, our QRNG system, in turn, demonstrates its robustness against ML. We further demonstrate the robustness of our ML approach by applying it to uniformly distributed random numbers from the QRNG and a congruential RNG. Hence, our result shows that ML has potentials in benchmarking the quality of RNG devices.

摘要: 对加密应用至关重要的随机数生成器（RNG）一直是对抗攻击的对象。这些攻击利用环境信息来预测生成的随机数，这些随机数应该是真正随机且不可预测的。尽管量子随机数发生器（QRNG）基于量子性质的固有不确定性，但测量过程中经典噪音的存在会损害QRNG的完整性。本文中，我们开发了一种预测机器学习（ML）分析，以研究确定性经典噪音在光学连续变量QRNG不同阶段的影响。当确定性噪音源突出时，我们的ML模型成功检测到固有相关性。引入适当的过滤和随机性提取过程后，我们的QRNG系统反过来又展示了其对ML的鲁棒性。我们通过将ML方法应用于来自QRNG和全合RNG的均匀分布随机数，进一步证明了ML方法的鲁棒性。因此，我们的结果表明ML在对RNG设备的质量进行基准测试方面具有潜力。



## **4. Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks**

可靠的解纠缠多视图学习对抗视图对抗攻击 cs.LG

11 pages, 11 figures, accepted by International Joint Conference on  Artificial Intelligence (IJCAI 2025)

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04046v1) [paper-pdf](http://arxiv.org/pdf/2505.04046v1)

**Authors**: Xuyang Wang, Siyuan Duan, Qizhi Li, Guiduo Duan, Yuan Sun, Dezhong Peng

**Abstract**: Recently, trustworthy multi-view learning has attracted extensive attention because evidence learning can provide reliable uncertainty estimation to enhance the credibility of multi-view predictions. Existing trusted multi-view learning methods implicitly assume that multi-view data is secure. In practice, however, in safety-sensitive applications such as autonomous driving and security monitoring, multi-view data often faces threats from adversarial perturbations, thereby deceiving or disrupting multi-view learning models. This inevitably leads to the adversarial unreliability problem (AUP) in trusted multi-view learning. To overcome this tricky problem, we propose a novel multi-view learning framework, namely Reliable Disentanglement Multi-view Learning (RDML). Specifically, we first propose evidential disentanglement learning to decompose each view into clean and adversarial parts under the guidance of corresponding evidences, which is extracted by a pretrained evidence extractor. Then, we employ the feature recalibration module to mitigate the negative impact of adversarial perturbations and extract potential informative features from them. Finally, to further ignore the irreparable adversarial interferences, a view-level evidential attention mechanism is designed. Extensive experiments on multi-view classification tasks with adversarial attacks show that our RDML outperforms the state-of-the-art multi-view learning methods by a relatively large margin.

摘要: 最近，值得信赖的多视图学习引起了广泛关注，因为证据学习可以提供可靠的不确定性估计，以增强多视图预测的可信度。现有的可信多视图学习方法隐含地假设多视图数据是安全的。然而，在实践中，在自动驾驶和安全监控等安全敏感应用中，多视图数据经常面临来自对抗扰动的威胁，从而欺骗或扰乱多视图学习模型。这不可避免地会导致可信多视图学习中的对抗不可靠性问题（AUP）。为了克服这个棘手的问题，我们提出了一种新颖的多视图学习框架，即可靠解纠缠多视图学习（RDML）。具体来说，我们首先提出证据解纠缠学习，在相应证据的指导下将每个视图分解为干净且对抗的部分，这些证据由预先训练的证据提取器提取。然后，我们使用特征重新校准模块来减轻对抗性扰动的负面影响，并从中提取潜在的信息特征。最后，为了进一步忽略不可挽回的对抗干扰，设计了视角级证据关注机制。针对具有对抗性攻击的多视图分类任务的大量实验表明，我们的RDML以相对较大的优势优于最先进的多视图学习方法。



## **5. MergeGuard: Efficient Thwarting of Trojan Attacks in Machine Learning Models**

MergeGuard：有效阻止机器学习模型中的特洛伊木马攻击 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.04015v1) [paper-pdf](http://arxiv.org/pdf/2505.04015v1)

**Authors**: Soheil Zibakhsh Shabgahi, Yaman Jandali, Farinaz Koushanfar

**Abstract**: This paper proposes MergeGuard, a novel methodology for mitigation of AI Trojan attacks. Trojan attacks on AI models cause inputs embedded with triggers to be misclassified to an adversary's target class, posing a significant threat to model usability trained by an untrusted third party. The core of MergeGuard is a new post-training methodology for linearizing and merging fully connected layers which we show simultaneously improves model generalizability and performance. Our Proof of Concept evaluation on Transformer models demonstrates that MergeGuard maintains model accuracy while decreasing trojan attack success rate, outperforming commonly used (post-training) Trojan mitigation by fine-tuning methodologies.

摘要: 本文提出了MergeGuard，这是一种缓解人工智能特洛伊攻击的新型方法。对人工智能模型的特洛伊木马攻击导致嵌入触发器的输入被错误分类到对手的目标类别，对不受信任的第三方训练的模型可用性构成重大威胁。MergeGuard的核心是一种新的训练后方法，用于线性化和合并完全连接的层，我们证明它可以同时提高模型的可概括性和性能。我们对Transformer模型的概念验证评估表明，MergeGuard保持了模型的准确性，同时降低了特洛伊木马攻击的成功率，通过微调方法优于常用的（训练后）特洛伊木马缓解。



## **6. Towards Universal and Black-Box Query-Response Only Attack on LLMs with QROA**

采用QROA对LLM进行通用和黑匣子仅查询响应攻击 cs.CL

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2406.02044v3) [paper-pdf](http://arxiv.org/pdf/2406.02044v3)

**Authors**: Hussein Jawad, Yassine Chenik, Nicolas J. -B. Brunel

**Abstract**: The rapid adoption of Large Language Models (LLMs) has exposed critical security and ethical vulnerabilities, particularly their susceptibility to adversarial manipulations. This paper introduces QROA, a novel black-box jailbreak method designed to identify adversarial suffixes that can bypass LLM alignment safeguards when appended to a malicious instruction. Unlike existing suffix-based jailbreak approaches, QROA does not require access to the model's logit or any other internal information. It also eliminates reliance on human-crafted templates, operating solely through the standard query-response interface of LLMs. By framing the attack as an optimization bandit problem, QROA employs a surrogate model and token level optimization to efficiently explore suffix variations. Furthermore, we propose QROA-UNV, an extension that identifies universal adversarial suffixes for individual models, enabling one-query jailbreaks across a wide range of instructions. Testing on multiple models demonstrates Attack Success Rate (ASR) greater than 80\%. These findings highlight critical vulnerabilities, emphasize the need for advanced defenses, and contribute to the development of more robust safety evaluations for secure AI deployment. The code is made public on the following link: https://github.com/qroa/QROA

摘要: 大型语言模型（LLM）的迅速采用暴露了关键的安全和道德漏洞，特别是它们容易受到对抗性操纵的影响。本文介绍了QROA，这是一种新型黑匣子越狱方法，旨在识别对抗性后缀，这些后缀在附加到恶意指令时可以绕过LLM对齐保障措施。与现有的基于后缀的越狱方法不同，QROA不需要访问模型的logit或任何其他内部信息。它还消除了对人工模板的依赖，仅通过LLM的标准查询-响应界面操作。通过将攻击定义为优化强盗问题，QROA采用代理模型和令牌级优化来有效地探索后缀变体。此外，我们还提出了QROA-UNV，这是一种扩展，可以为各个模型识别通用的对抗性后缀，从而实现跨广泛指令的单查询越狱。对多个模型的测试表明攻击成功率（ASB）大于80%。这些发现凸显了关键漏洞，强调了对先进防御的需要，并有助于开发更强大的安全评估以实现安全的人工智能部署。该代码在以下链接上公开：https://github.com/qroa/QROA



## **7. Model-Targeted Data Poisoning Attacks against ITS Applications with Provable Convergence**

可证明收敛的面向模型的ITS应用数据中毒攻击 math.OC

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03966v1) [paper-pdf](http://arxiv.org/pdf/2505.03966v1)

**Authors**: Xin Wanga, Feilong Wang, Yuan Hong, R. Tyrrell Rockafellar, Xuegang, Ban

**Abstract**: The growing reliance of intelligent systems on data makes the systems vulnerable to data poisoning attacks. Such attacks could compromise machine learning or deep learning models by disrupting the input data. Previous studies on data poisoning attacks are subject to specific assumptions, and limited attention is given to learning models with general (equality and inequality) constraints or lacking differentiability. Such learning models are common in practice, especially in Intelligent Transportation Systems (ITS) that involve physical or domain knowledge as specific model constraints. Motivated by ITS applications, this paper formulates a model-target data poisoning attack as a bi-level optimization problem with a constrained lower-level problem, aiming to induce the model solution toward a target solution specified by the adversary by modifying the training data incrementally. As the gradient-based methods fail to solve this optimization problem, we propose to study the Lipschitz continuity property of the model solution, enabling us to calculate the semi-derivative, a one-sided directional derivative, of the solution over data. We leverage semi-derivative descent to solve the bi-level optimization problem, and establish the convergence conditions of the method to any attainable target model. The model and solution method are illustrated with a simulation of a poisoning attack on the lane change detection using SVM.

摘要: 智能系统对数据的日益依赖使得系统容易受到数据中毒攻击。这种攻击可能会破坏输入数据，从而危及机器学习或深度学习模型。以往关于数据中毒攻击的研究都受到特定假设的限制，对具有一般（等式和不等式）约束或缺乏可微性的学习模型的关注有限。这种学习模型在实践中很常见，特别是在涉及物理或领域知识作为特定模型约束的智能交通系统（ITS）中。受ITS应用的启发，本文将模型-目标数据中毒攻击描述为具有约束较低层问题的双层优化问题，旨在通过增量修改训练数据将模型解引导到对手指定的目标解。由于基于梯度的方法无法解决这个优化问题，我们建议研究模型解的Lipschitz连续性，使我们能够计算解对数据的半导（单边方向导）。我们利用半导下降来解决双层优化问题，并建立该方法对任何可达到的目标模型的收敛条件。通过对使用支持者对车道变更检测的中毒攻击进行模拟，说明了该模型和解决方法。



## **8. Sustainable Smart Farm Networks: Enhancing Resilience and Efficiency with Decision Theory-Guided Deep Reinforcement Learning**

可持续智能农场网络：通过决策理论指导的深度强化学习增强韧性和效率 cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03721v1) [paper-pdf](http://arxiv.org/pdf/2505.03721v1)

**Authors**: Dian Chen, Zelin Wan, Dong Sam Ha, Jin-Hee Cho

**Abstract**: Solar sensor-based monitoring systems have become a crucial agricultural innovation, advancing farm management and animal welfare through integrating sensor technology, Internet-of-Things, and edge and cloud computing. However, the resilience of these systems to cyber-attacks and their adaptability to dynamic and constrained energy supplies remain largely unexplored. To address these challenges, we propose a sustainable smart farm network designed to maintain high-quality animal monitoring under various cyber and adversarial threats, as well as fluctuating energy conditions. Our approach utilizes deep reinforcement learning (DRL) to devise optimal policies that maximize both monitoring effectiveness and energy efficiency. To overcome DRL's inherent challenge of slow convergence, we integrate transfer learning (TL) and decision theory (DT) to accelerate the learning process. By incorporating DT-guided strategies, we optimize monitoring quality and energy sustainability, significantly reducing training time while achieving comparable performance rewards. Our experimental results prove that DT-guided DRL outperforms TL-enhanced DRL models, improving system performance and reducing training runtime by 47.5%.

摘要: 基于太阳能传感器的监测系统已成为一项重要的农业创新，通过集成传感器技术、物联网、边缘和云计算，促进了农场管理和动物福利。然而，这些系统对网络攻击的弹性及其对动态和受限能源供应的适应性在很大程度上仍未得到探索。为了应对这些挑战，我们提出了一个可持续的智能农场网络，旨在在各种网络和对抗性威胁以及波动的能源条件下保持高质量的动物监测。我们的方法利用深度强化学习（DRL）来设计最佳策略，以最大限度地提高监测效率和能源效率。为了克服DRL固有的收敛速度慢的挑战，我们集成了迁移学习（TL）和决策理论（DT）来加速学习过程。通过结合DT引导的策略，我们优化了质量和能源可持续性的监控，显着减少培训时间，同时实现相当的绩效奖励。我们的实验结果证明，DT引导的DRL优于TL增强的DRL模型，提高了系统性能，并将训练运行时间减少了47.5%。



## **9. Adversarial Robustness of Deep Learning Models for Inland Water Body Segmentation from SAR Images**

SAR图像内陆水体分割深度学习模型的对抗鲁棒性 eess.IV

21 pages, 15 figures, 2 tables

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.01884v2) [paper-pdf](http://arxiv.org/pdf/2505.01884v2)

**Authors**: Siddharth Kothari, Srinivasan Murali, Sankalp Kothari, Ujjwal Verma, Jaya Sreevalsan-Nair

**Abstract**: Inland water body segmentation from Synthetic Aperture Radar (SAR) images is an important task needed for several applications, such as flood mapping. While SAR sensors capture data in all-weather conditions as high-resolution images, differentiating water and water-like surfaces from SAR images is not straightforward. Inland water bodies, such as large river basins, have complex geometry, which adds to the challenge of segmentation. U-Net is a widely used deep learning model for land-water segmentation of SAR images. In practice, manual annotation is often used to generate the corresponding water masks as ground truth. Manual annotation of the images is prone to label noise owing to data poisoning attacks, especially due to complex geometry. In this work, we simulate manual errors in the form of adversarial attacks on the U-Net model and study the robustness of the model to human errors in annotation. Our results indicate that U-Net can tolerate a certain level of corruption before its performance drops significantly. This finding highlights the crucial role that the quality of manual annotations plays in determining the effectiveness of the segmentation model. The code and the new dataset, along with adversarial examples for robust training, are publicly available. (GitHub link - https://github.com/GVCL/IWSeg-SAR-Poison.git)

摘要: 从合成口径雷达（SAR）图像中分割内陆水体是洪水绘图等多种应用所需的重要任务。虽然SAR传感器将全天候条件下的数据捕获为高分辨率图像，但区分水和类水表面与SAR图像并不简单。内陆水体（例如大型流域）具有复杂的几何形状，这增加了分段的挑战。U-Net是一种广泛使用的深度学习模型，用于SAR图像的海陆分割。在实践中，通常使用手动注释来生成相应的水面具作为地面真相。由于数据中毒攻击，特别是由于复杂的几何形状，图像的手动注释容易出现标签噪音。在这项工作中，我们以对抗攻击的形式模拟了对U-Net模型的手动错误，并研究了该模型对注释中人为错误的鲁棒性。我们的结果表明，U-Net在性能显着下降之前可以容忍一定程度的腐败。这一发现凸显了手动注释的质量在决定分割模型的有效性方面所发挥的关键作用。代码和新数据集，以及用于稳健训练的对抗示例都已公开。（GitHub链接-https：//github.com/GVCL/IWSeg-SAR-Poison.git）



## **10. Data-Driven Falsification of Cyber-Physical Systems**

数据驱动的网络物理系统证伪 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03863v1) [paper-pdf](http://arxiv.org/pdf/2505.03863v1)

**Authors**: Atanu Kundu, Sauvik Gon, Rajarshi Ray

**Abstract**: Cyber-Physical Systems (CPS) are abundant in safety-critical domains such as healthcare, avionics, and autonomous vehicles. Formal verification of their operational safety is, therefore, of utmost importance. In this paper, we address the falsification problem, where the focus is on searching for an unsafe execution in the system instead of proving their absence. The contribution of this paper is a framework that (a) connects the falsification of CPS with the falsification of deep neural networks (DNNs) and (b) leverages the inherent interpretability of Decision Trees for faster falsification of CPS. This is achieved by: (1) building a surrogate model of the CPS under test, either as a DNN model or a Decision Tree, (2) application of various DNN falsification tools to falsify CPS, and (3) a novel falsification algorithm guided by the explanations of safety violations of the CPS model extracted from its Decision Tree surrogate. The proposed framework has the potential to exploit a repertoire of \emph{adversarial attack} algorithms designed to falsify robustness properties of DNNs, as well as state-of-the-art falsification algorithms for DNNs. Although the presented methodology is applicable to systems that can be executed/simulated in general, we demonstrate its effectiveness, particularly in CPS. We show that our framework, implemented as a tool \textsc{FlexiFal}, can detect hard-to-find counterexamples in CPS that have linear and non-linear dynamics. Decision tree-guided falsification shows promising results in efficiently finding multiple counterexamples in the ARCH-COMP 2024 falsification benchmarks~\cite{khandait2024arch}.

摘要: 网络物理系统（CPS）广泛应用于医疗保健、航空电子设备和自动驾驶汽车等安全关键领域。因此，对其运营安全性的正式验证至关重要。在本文中，我们解决了伪造问题，重点是搜索系统中不安全的执行，而不是证明它们的不存在。本文的贡献是一个框架，该框架（a）将CPS的伪造与深度神经网络（DNN）的伪造联系起来，并且（b）利用决策树的固有可解释性来更快地伪造CPS。这是通过以下方式实现的：（1）构建受测CPS的代理模型，无论是DNN模型还是决策树，（2）应用各种DNN伪造工具来伪造CPS，以及（3）以CPS模型的安全违规解释为指导的新型伪造算法从其决策树代理中提取。提出的框架有可能利用一系列旨在伪造DNN鲁棒性属性的\{对抗攻击}算法，以及DNN的最先进伪造算法。尽管所提出的方法适用于一般可以执行/模拟的系统，但我们证明了它的有效性，特别是在CPS中。我们表明，我们的框架，实现为一个工具，可以检测到很难找到的反例在CPS具有线性和非线性动态。决策树引导的证伪在有效地发现多个反例在ARCH-COMP 2024证伪基准测试中显示出有希望的结果。



## **11. ALMA: Aggregated Lipschitz Maximization Attack on Auto-encoders**

ALMA：对自动编码器的聚合Lipschitz最大化攻击 cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03646v1) [paper-pdf](http://arxiv.org/pdf/2505.03646v1)

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Eirini Ntoutsi

**Abstract**: Despite the extensive use of deep autoencoders (AEs) in critical applications, their adversarial robustness remains relatively underexplored compared to classification models. AE robustness is characterized by the Lipschitz bounds of its components. Existing robustness evaluation frameworks based on white-box attacks do not fully exploit the vulnerabilities of intermediate ill-conditioned layers in AEs. In the context of optimizing imperceptible norm-bounded additive perturbations to maximize output damage, existing methods struggle to effectively propagate adversarial loss gradients throughout the network, often converging to less effective perturbations. To address this, we propose a novel layer-conditioning-based adversarial optimization objective that effectively guides the adversarial map toward regions of local Lipschitz bounds by enhancing loss gradient information propagation during attack optimization. We demonstrate through extensive experiments on state-of-the-art AEs that our adversarial objective results in stronger attacks, outperforming existing methods in both universal and sample-specific scenarios. As a defense method against this attack, we introduce an inference-time adversarially trained defense plugin that mitigates the effects of adversarial examples.

摘要: 尽管深度自动编码器（AE）在关键应用中广泛使用，但与分类模型相比，其对抗鲁棒性仍然相对未充分研究。AE鲁棒性由其成分的Lipschitz界来描述。现有的基于白盒攻击的稳健性评估框架并未充分利用AE中中间病态层的漏洞。在优化不可感知的规范有界添加性扰动以最大化输出损害的背景下，现有方法很难在整个网络中有效传播对抗损失梯度，通常会收敛到效率较低的扰动。为了解决这个问题，我们提出了一种新型的基于层条件的对抗性优化目标，该目标通过在攻击优化期间增强损失梯度信息传播，有效地将对抗性地图引导到局部Lipschitz界限区域。我们通过对最先进AE的广泛实验证明，我们的对抗目标会导致更强的攻击，在通用和特定样本场景中都优于现有方法。作为针对这种攻击的防御方法，我们引入了一个推理时对抗训练的防御插件，该插件可以减轻对抗示例的影响。



## **12. The Adaptive Arms Race: Redefining Robustness in AI Security**

自适应军备竞赛：重新定义人工智能安全的稳健性 cs.AI

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2312.13435v3) [paper-pdf](http://arxiv.org/pdf/2312.13435v3)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world AI-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. Canonical robustness evaluation relies on adaptive attacks, which leverage complete knowledge of the defense and are tailored to bypass it. This work broadens the notion of adaptivity, which we employ to enhance both attacks and defenses, showing how they can benefit from mutual learning through interaction. We introduce a framework for adaptively optimizing black-box attacks and defenses under the competitive game they form. To assess robustness reliably, it is essential to evaluate against realistic and worst-case attacks. We thus enhance attacks and their evasive arsenal together using RL, apply the same principle to defenses, and evaluate them first independently and then jointly under a multi-agent perspective. We find that active defenses, those that dynamically control system responses, are an essential complement to model hardening against decision-based attacks; that these defenses can be circumvented by adaptive attacks, something that elicits defenses being adaptive too. Our findings, supported by an extensive theoretical and empirical investigation, confirm that adaptive adversaries pose a serious threat to black-box AI-based systems, rekindling the proverbial arms race. Notably, our approach outperforms the state-of-the-art black-box attacks and defenses, while bringing them together to render effective insights into the robustness of real-world deployed ML-based systems.

摘要: 尽管做出了相当大的努力来使其稳健性，但现实世界中的基于人工智能的系统仍然容易受到基于决策的攻击，因为迄今为止证明其操作稳健性的明确证据很难解决。典型稳健性评估依赖于自适应攻击，这种攻击利用了对防御的完整知识，并经过量身定制以绕过它。这项工作扩展了自适应性的概念，我们使用它来增强攻击和防御，展示了它们如何通过交互从相互学习中受益。我们引入了一个框架，用于在黑匣子形成的竞争游戏下自适应地优化黑匣子攻击和防御。为了可靠地评估稳健性，必须针对现实和最坏情况的攻击进行评估。因此，我们使用RL共同增强攻击及其规避武器库，将相同的原则应用于防御，并首先独立评估它们，然后在多智能体的角度下联合评估它们。我们发现，主动防御（动态控制系统响应的防御）是针对基于决策的攻击的模型强化的重要补充;这些防御可以被自适应攻击规避，这使得防御也具有自适应性。我们的研究结果在广泛的理论和实证调查的支持下证实，适应性对手对基于黑匣子的人工智能系统构成了严重威胁，重新点燃了众所周知的军备竞赛。值得注意的是，我们的方法优于最先进的黑匣子攻击和防御，同时将它们结合在一起，以有效洞察现实世界部署的基于ML的系统的稳健性。



## **13. Uncovering the Limitations of Model Inversion Evaluation: Benchmarks and Connection to Type-I Adversarial Attacks**

揭示模型反演评估的局限性：基准测试和与I型对抗攻击的联系 cs.LG

Our dataset and code are available in the Supp

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03519v1) [paper-pdf](http://arxiv.org/pdf/2505.03519v1)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information of private training data by exploiting access to machine learning models. The most common evaluation framework for MI attacks/defenses relies on an evaluation model that has been utilized to assess progress across almost all MI attacks and defenses proposed in recent years. In this paper, for the first time, we present an in-depth study of MI evaluation. Firstly, we construct the first comprehensive human-annotated dataset of MI attack samples, based on 28 setups of different MI attacks, defenses, private and public datasets. Secondly, using our dataset, we examine the accuracy of the MI evaluation framework and reveal that it suffers from a significant number of false positives. These findings raise questions about the previously reported success rates of SOTA MI attacks. Thirdly, we analyze the causes of these false positives, design controlled experiments, and discover the surprising effect of Type I adversarial features on MI evaluation, as well as adversarial transferability, highlighting a relationship between two previously distinct research areas. Our findings suggest that the performance of SOTA MI attacks has been overestimated, with the actual privacy leakage being significantly less than previously reported. In conclusion, we highlight critical limitations in the widely used MI evaluation framework and present our methods to mitigate false positive rates. We remark that prior research has shown that Type I adversarial attacks are very challenging, with no existing solution. Therefore, we urge to consider human evaluation as a primary MI evaluation framework rather than merely a supplement as in previous MI research. We also encourage further work on developing more robust and reliable automatic evaluation frameworks.

摘要: 模型倒置（MI）攻击旨在通过利用对机器学习模型的访问来重建私人训练数据的信息。MI攻击/防御最常见的评估框架依赖于一个评估模型，该模型已用于评估近年来提出的几乎所有MI攻击和防御的进展。本文首次对MI评估进行了深入的研究。首先，我们基于28个不同MI攻击、防御、私有和公共数据集的设置，构建了第一个全面的MI攻击样本的人类注释数据集。其次，使用我们的数据集，我们检查了MI评估框架的准确性，并发现它存在大量的误报。这些发现对先前报告的SOTA MI攻击成功率提出了质疑。第三，我们分析了这些假阳性的原因，设计了对照实验，并发现了I型对抗性特征对MI评估的惊人影响，以及对抗性可转移性，突出了两个以前不同的研究领域之间的关系。我们的研究结果表明，SOTA MI攻击的性能被高估了，实际的隐私泄露明显低于以前的报告。总之，我们强调了广泛使用的MI评估框架的关键局限性，并介绍了我们降低假阳性率的方法。我们指出，之前的研究表明，I型对抗性攻击非常具有挑战性，目前还没有解决方案。因此，我们敦促将人类评估视为主要的MI评估框架，而不仅仅是之前的MI研究的补充。我们还鼓励进一步开发更强大、更可靠的自动评估框架。



## **14. BadLingual: A Novel Lingual-Backdoor Attack against Large Language Models**

BadLingual：针对大型语言模型的新型语言后门攻击 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03501v1) [paper-pdf](http://arxiv.org/pdf/2505.03501v1)

**Authors**: Zihan Wang, Hongwei Li, Rui Zhang, Wenbo Jiang, Kangjie Chen, Tianwei Zhang, Qingchuan Zhao, Guowen Xu

**Abstract**: In this paper, we present a new form of backdoor attack against Large Language Models (LLMs): lingual-backdoor attacks. The key novelty of lingual-backdoor attacks is that the language itself serves as the trigger to hijack the infected LLMs to generate inflammatory speech. They enable the precise targeting of a specific language-speaking group, exacerbating racial discrimination by malicious entities. We first implement a baseline lingual-backdoor attack, which is carried out by poisoning a set of training data for specific downstream tasks through translation into the trigger language. However, this baseline attack suffers from poor task generalization and is impractical in real-world settings. To address this challenge, we design BadLingual, a novel task-agnostic lingual-backdoor, capable of triggering any downstream tasks within the chat LLMs, regardless of the specific questions of these tasks. We design a new approach using PPL-constrained Greedy Coordinate Gradient-based Search (PGCG) based adversarial training to expand the decision boundary of lingual-backdoor, thereby enhancing the generalization ability of lingual-backdoor across various tasks. We perform extensive experiments to validate the effectiveness of our proposed attacks. Specifically, the baseline attack achieves an ASR of over 90% on the specified tasks. However, its ASR reaches only 37.61% across six tasks in the task-agnostic scenario. In contrast, BadLingual brings up to 37.35% improvement over the baseline. Our study sheds light on a new perspective of vulnerabilities in LLMs with multilingual capabilities and is expected to promote future research on the potential defenses to enhance the LLMs' robustness

摘要: 在本文中，我们提出了一种针对大型语言模型（LLM）的新形式后门攻击：语言后门攻击。语言后门攻击的关键新颖之处在于，语言本身充当了劫持受感染LLM以产生煽动性言语的触发器。它们能够准确瞄准特定语言群体，加剧恶意实体的种族歧视。我们首先实施基线语言后门攻击，通过翻译成触发语言来毒害特定下游任务的一组训练数据来执行该攻击。然而，这种基线攻击的任务概括性较差，并且在现实世界环境中不切实际。为了应对这一挑战，我们设计了BadLingual，这是一种新型的任务不可知语言后门，能够触发聊天LLM内的任何下游任务，无论这些任务的具体问题如何。我们设计了一种使用PPL约束的基于贪婪协调搜索（PGCG）的对抗训练的新方法，以扩大语言后门的决策边界，从而增强语言后门在各种任务中的概括能力。我们进行了广泛的实验来验证我们提出的攻击的有效性。具体来说，基线攻击在指定任务上实现了超过90%的ASB。然而，在任务不可知的场景中，其六项任务的ASB仅达到37.61%。相比之下，BadLingual较基线提高了37.35%。我们的研究揭示了具有多语言功能的LLM漏洞的新视角，并预计将促进未来对潜在防御措施的研究，以增强LLM的稳健性



## **15. Mitigating Backdoor Triggered and Targeted Data Poisoning Attacks in Voice Authentication Systems**

缓解语音认证系统中后门触发和有针对性的数据中毒攻击 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03455v1) [paper-pdf](http://arxiv.org/pdf/2505.03455v1)

**Authors**: Alireza Mohammadi, Keshav Sood, Dhananjay Thiruvady, Asef Nazari

**Abstract**: Voice authentication systems remain susceptible to two major threats: backdoor triggered attacks and targeted data poisoning attacks. This dual vulnerability is critical because conventional solutions typically address each threat type separately, leaving systems exposed to adversaries who can exploit both attacks simultaneously. We propose a unified defense framework that effectively addresses both BTA and TDPA. Our framework integrates a frequency focused detection mechanism that flags covert pitch boosting and sound masking backdoor attacks in near real time, followed by a convolutional neural network that addresses TDPA. This dual layered defense approach utilizes multidimensional acoustic features to isolate anomalous signals without requiring costly model retraining. In particular, our PBSM detection mechanism can seamlessly integrate into existing voice authentication pipelines and scale effectively for large scale deployments. Experimental results on benchmark datasets and their compression with the state of the art algorithm demonstrate that our PBSM detection mechanism outperforms the state of the art. Our framework reduces attack success rates to as low as five to fifteen percent while maintaining a recall rate of up to ninety five percent in recognizing TDPA.

摘要: 语音认证系统仍然容易受到两种主要威胁：后门触发攻击和有针对性的数据中毒攻击。这种双重漏洞至关重要，因为传统的解决方案通常会单独解决每种威胁类型，从而使系统暴露在可以同时利用这两种攻击的对手手中。我们提出了一个有效解决MTA和TDPA的统一防御框架。我们的框架集成了以频率为中心的检测机制，该机制近乎实时地标记隐蔽音调增强和声音掩蔽后门攻击，然后是解决TDPA的卷积神经网络。这种双层防御方法利用多维声学特征来隔离异常信号，而无需昂贵的模型再训练。特别是，我们的PBSM检测机制可以无缝集成到现有的语音认证管道中，并有效扩展以适应大规模部署。对基准数据集及其使用最先进算法进行压缩的实验结果表明，我们的PBSM检测机制优于最先进的技术。我们的框架将攻击成功率降低至低至百分之五到百分之十五，同时在识别TDPA时保持高达百分之九十五的召回率。



## **16. Robustness in AI-Generated Detection: Enhancing Resistance to Adversarial Attacks**

人工智能生成检测的鲁棒性：增强对抗性攻击的抵抗力 cs.CV

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03435v1) [paper-pdf](http://arxiv.org/pdf/2505.03435v1)

**Authors**: Sun Haoxuan, Hong Yan, Zhan Jiahui, Chen Haoxing, Lan Jun, Zhu Huijia, Wang Weiqiang, Zhang Liqing, Zhang Jianfu

**Abstract**: The rapid advancement of generative image technology has introduced significant security concerns, particularly in the domain of face generation detection. This paper investigates the vulnerabilities of current AI-generated face detection systems. Our study reveals that while existing detection methods often achieve high accuracy under standard conditions, they exhibit limited robustness against adversarial attacks. To address these challenges, we propose an approach that integrates adversarial training to mitigate the impact of adversarial examples. Furthermore, we utilize diffusion inversion and reconstruction to further enhance detection robustness. Experimental results demonstrate that minor adversarial perturbations can easily bypass existing detection systems, but our method significantly improves the robustness of these systems. Additionally, we provide an in-depth analysis of adversarial and benign examples, offering insights into the intrinsic characteristics of AI-generated content. All associated code will be made publicly available in a dedicated repository to facilitate further research and verification.

摘要: 生成图像技术的快速发展带来了重大的安全问题，特别是在面部生成检测领域。本文研究了当前人工智能生成的人脸检测系统的漏洞。我们的研究表明，虽然现有的检测方法通常在标准条件下实现高准确性，但它们对对抗攻击的鲁棒性有限。为了应对这些挑战，我们提出了一种整合对抗性训练的方法，以减轻对抗性示例的影响。此外，我们利用扩散倒置和重建来进一步增强检测鲁棒性。实验结果表明，微小的对抗性扰动可以轻松绕过现有的检测系统，但我们的方法显着提高了这些系统的鲁棒性。此外，我们还对对抗性和良性示例进行深入分析，深入分析人工智能生成内容的内在特征。所有相关代码都将在专用存储库中公开，以促进进一步的研究和验证。



## **17. Attention-aggregated Attack for Boosting the Transferability of Facial Adversarial Examples**

提高面部对抗示例可移植性的注意力聚集攻击 cs.CV

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03383v1) [paper-pdf](http://arxiv.org/pdf/2505.03383v1)

**Authors**: Jian-Wei Li, Wen-Ze Shao

**Abstract**: Adversarial examples have revealed the vulnerability of deep learning models and raised serious concerns about information security. The transfer-based attack is a hot topic in black-box attacks that are practical to real-world scenarios where the training datasets, parameters, and structure of the target model are unknown to the attacker. However, few methods consider the particularity of class-specific deep models for fine-grained vision tasks, such as face recognition (FR), giving rise to unsatisfactory attacking performance. In this work, we first investigate what in a face exactly contributes to the embedding learning of FR models and find that both decisive and auxiliary facial features are specific to each FR model, which is quite different from the biological mechanism of human visual system. Accordingly we then propose a novel attack method named Attention-aggregated Attack (AAA) to enhance the transferability of adversarial examples against FR, which is inspired by the attention divergence and aims to destroy the facial features that are critical for the decision-making of other FR models by imitating their attentions on the clean face images. Extensive experiments conducted on various FR models validate the superiority and robust effectiveness of the proposed method over existing methods.

摘要: 对抗性的例子揭示了深度学习模型的脆弱性，并引发了人们对信息安全的严重担忧。基于传输的攻击是黑匣子攻击中的热门话题，这种攻击对于攻击者未知目标模型的训练数据集、参数和结构的现实场景很实用。然而，很少有方法考虑针对细粒度视觉任务（例如人脸识别（FR））的特定类别深度模型的特殊性，从而导致攻击性能不令人满意。在这项工作中，我们首先研究了面部中的哪些因素对FR模型的嵌入学习做出了贡献，发现决定性和辅助面部特征都是每个FR模型特有的，这与人类视觉系统的生物学机制截然不同。因此，我们提出了一种名为注意力聚集攻击（AAA）的新型攻击方法，以增强对抗性示例针对FR的可移植性，该方法受到注意力分歧的启发，旨在通过模仿其他FR模型对干净面部图像的注意力来破坏对决策至关重要的面部特征。在各种FR模型上进行的大量实验验证了所提出的方法相对于现有方法的优越性和鲁棒性。



## **18. A Chaos Driven Metric for Backdoor Attack Detection**

一种基于混沌驱动的后门攻击检测方法 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03208v1) [paper-pdf](http://arxiv.org/pdf/2505.03208v1)

**Authors**: Hema Karnam Surendrababu, Nithin Nagaraj

**Abstract**: The advancement and adoption of Artificial Intelligence (AI) models across diverse domains have transformed the way we interact with technology. However, it is essential to recognize that while AI models have introduced remarkable advancements, they also present inherent challenges such as their vulnerability to adversarial attacks. The current work proposes a novel defense mechanism against one of the most significant attack vectors of AI models - the backdoor attack via data poisoning of training datasets. In this defense technique, an integrated approach that combines chaos theory with manifold learning is proposed. A novel metric - Precision Matrix Dependency Score (PDS) that is based on the conditional variance of Neurochaos features is formulated. The PDS metric has been successfully evaluated to distinguish poisoned samples from non-poisoned samples across diverse datasets.

摘要: 人工智能（AI）模型在不同领域的进步和采用改变了我们与技术互动的方式。然而，必须认识到，虽然人工智能模型带来了显着的进步，但它们也面临着固有的挑战，例如容易受到对抗攻击。当前的工作提出了一种新颖的防御机制，以对抗人工智能模型最重要的攻击载体之一--通过训练数据集的数据中毒进行的后门攻击。在这种防御技术中，提出了一种将混乱理论与多维学习相结合的集成方法。提出了一种基于Neurochaos特征的条件方差的新型指标--精确矩阵依赖性得分（DDS）。PDC指标已成功评估，可在不同数据集中区分有毒样本与非有毒样本。



## **19. Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models**

使用机械可解释性来应对大型语言模型的对抗攻击 cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2503.06269v2) [paper-pdf](http://arxiv.org/pdf/2503.06269v2)

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at https://github.com/Sckathach/subspace-rerouting.

摘要: 用于针对LLM创建对抗性扰动的传统白盒方法通常仅依赖于目标模型的梯度计算，而忽略了负责攻击成功或失败的内部机制。相反，分析这些内部机制的可解释性研究缺乏运行时干预之外的实际应用。我们通过引入一种新颖的白盒方法来弥合这一差距，该方法利用机械解释性技术来制作实用的对抗性输入。具体来说，我们首先识别接受子空间--不会触发模型拒绝机制的特征载体集--然后使用基于梯度的优化将嵌入从拒绝子空间重新路由到接受子空间，有效地实现越狱。与经常失败或需要数小时计算的现有技术相比，这种有针对性的方法显着降低了计算成本，在几分钟甚至几秒钟内就实现了对Gemma 2、Llama3.2和Qwen 2.5等最先进模型80- 95%的攻击成功率。我们相信这种方法为攻击研究和防御开发开辟了新的方向。此外，它展示了机械解释性的实际应用，而其他方法效率较低，这凸显了它的实用性。代码和生成的数据集可在https://github.com/Sckathach/subspace-rerouting上获取。



## **20. PEEK: Phishing Evolution Framework for Phishing Generation and Evolving Pattern Analysis using Large Language Models**

TEK：使用大型语言模型进行网络钓鱼生成和演变模式分析的网络钓鱼进化框架 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2411.11389v2) [paper-pdf](http://arxiv.org/pdf/2411.11389v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), in particular, deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains detection effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems and people vulnerable to an ever-growing array of attacks. We propose the first Phishing Evolution FramEworK (PEEK) for augmenting phishing email datasets with respect to quality and diversity, and analyzing changing phishing patterns for detection to adapt to updated phishing attacks. Specifically, we integrate large language models (LLMs) into the process of adversarial training to enhance the performance of the generated dataset and leverage persuasion principles in a recurrent framework to facilitate the understanding of changing phishing strategies. PEEK raises the proportion of usable phishing samples from 21.4% to 84.8%, surpassing existing works that rely on prompting and fine-tuning LLMs. The phishing datasets provided by PEEK, with evolving phishing patterns, outperform the other two available LLM-generated phishing email datasets in improving detection robustness. PEEK phishing boosts detectors' accuracy to over 88% and reduces adversarial sensitivity by up to 70%, still maintaining 70% detection accuracy against adversarial attacks.

摘要: 网络钓鱼仍然是一种普遍存在的网络威胁，因为攻击者制作了欺骗性电子邮件来引诱受害者泄露敏感信息。虽然人工智能（AI），特别是深度学习，已成为防御网络钓鱼攻击的关键组成部分，但这些方法面临着严重的局限性。主要由于隐私问题，公开可用的、多样化的和更新的数据的稀缺限制了检测有效性。随着网络钓鱼策略的迅速发展，在有限、过时的数据上训练的模型很难检测到新的、复杂的欺骗策略，从而使系统和人们容易受到越来越多的攻击。我们提出了第一个网络钓鱼Evolution FramEworK（TEK），用于增强网络钓鱼电子邮件数据集的质量和多样性，并分析不断变化的网络钓鱼模式进行检测，以适应更新的网络钓鱼攻击。具体来说，我们将大型语言模型（LLM）集成到对抗训练过程中，以增强生成的数据集的性能，并在循环框架中利用说服原则，以促进对不断变化的网络钓鱼策略的理解。TEK将可用网络钓鱼样本的比例从21.4%提高到84.8%，超过了依赖提示和微调LLM的现有作品。TEK提供的网络钓鱼数据集具有不断变化的网络钓鱼模式，在提高检测稳健性方面优于其他两个可用的LLM生成的网络钓鱼电子邮件数据集。TEK网络钓鱼将检测器的准确性提高至88%以上，并将对抗敏感性降低高达70%，针对对抗攻击仍保持70%的检测准确性。



## **21. Adversarial Sample Generation for Anomaly Detection in Industrial Control Systems**

用于工业控制系统异常检测的对抗样本生成 cs.CR

Accepted in the 1st Workshop on Modeling and Verification for Secure  and Performant Cyber-Physical Systems in conjunction with Cyber-Physical  Systems and Internet-of-Things Week, Irvine, USA, May 6-9, 2025

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03120v1) [paper-pdf](http://arxiv.org/pdf/2505.03120v1)

**Authors**: Abdul Mustafa, Muhammad Talha Khan, Muhammad Azmi Umer, Zaki Masood, Chuadhry Mujeeb Ahmed

**Abstract**: Machine learning (ML)-based intrusion detection systems (IDS) are vulnerable to adversarial attacks. It is crucial for an IDS to learn to recognize adversarial examples before malicious entities exploit them. In this paper, we generated adversarial samples using the Jacobian Saliency Map Attack (JSMA). We validate the generalization and scalability of the adversarial samples to tackle a broad range of real attacks on Industrial Control Systems (ICS). We evaluated the impact by assessing multiple attacks generated using the proposed method. The model trained with adversarial samples detected attacks with 95% accuracy on real-world attack data not used during training. The study was conducted using an operational secure water treatment (SWaT) testbed.

摘要: 基于机器学习（ML）的入侵检测系统（IDS）容易受到对抗攻击。对于IDS来说，在恶意实体利用它们之前学会识别对抗性示例至关重要。在本文中，我们使用雅可比显着地图攻击（JSM）生成对抗样本。我们验证了对抗样本的一般性和可扩展性，以应对对工业控制系统（ICS）的广泛实际攻击。我们通过评估使用所提出的方法产生的多个攻击来评估影响。用对抗样本训练的模型在训练期间未使用的现实世界攻击数据上以95%的准确率检测到攻击。该研究使用可操作的安全水处理（SWaT）测试台进行。



## **22. Adversarial Attacks in Multimodal Systems: A Practitioner's Survey**

多模式系统中的对抗性攻击：从业者的调查 cs.LG

Accepted in IEEE COMPSAC 2025

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03084v1) [paper-pdf](http://arxiv.org/pdf/2505.03084v1)

**Authors**: Shashank Kapoor, Sanjay Surendranath Girija, Lakshit Arora, Dipen Pradhan, Ankit Shetgaonkar, Aman Raj

**Abstract**: The introduction of multimodal models is a huge step forward in Artificial Intelligence. A single model is trained to understand multiple modalities: text, image, video, and audio. Open-source multimodal models have made these breakthroughs more accessible. However, considering the vast landscape of adversarial attacks across these modalities, these models also inherit vulnerabilities of all the modalities, and ultimately, the adversarial threat amplifies. While broad research is available on possible attacks within or across these modalities, a practitioner-focused view that outlines attack types remains absent in the multimodal world. As more Machine Learning Practitioners adopt, fine-tune, and deploy open-source models in real-world applications, it's crucial that they can view the threat landscape and take the preventive actions necessary. This paper addresses the gap by surveying adversarial attacks targeting all four modalities: text, image, video, and audio. This survey provides a view of the adversarial attack landscape and presents how multimodal adversarial threats have evolved. To the best of our knowledge, this survey is the first comprehensive summarization of the threat landscape in the multimodal world.

摘要: 多模式模型的引入是人工智能向前迈出的一大步。单个模型经过训练以理解多种模式：文本、图像、视频和音频。开源多模式模型使这些突破变得更容易实现。然而，考虑到这些模式中对抗性攻击的广阔格局，这些模型也继承了所有模式的脆弱性，最终，对抗性威胁会被放大。虽然对这些模式内部或跨这些模式的可能攻击进行了广泛的研究，但在多模式世界中仍然缺乏以攻击者为中心、概述攻击类型的观点。随着越来越多的机器学习实践者在现实世界的应用程序中采用、微调和部署开源模型，他们能够查看威胁格局并采取必要的预防措施至关重要。本文通过调查针对所有四种模式（文本、图像、视频和音频）的对抗攻击来解决这一差距。这项调查提供了对抗性攻击格局的视图，并展示了多模式对抗性威胁的演变方式。据我们所知，这项调查是对多模式世界威胁格局的首次全面总结。



## **23. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

大型语言模型作为软件分析中稳健的数据生成器：我们已经做到了吗？ cs.SE

Accepted to the AI Model/Data Track of the Evaluation and Assessment  in Software Engineering (EASE) 2025 Conference

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2411.10565v3) [paper-pdf](http://arxiv.org/pdf/2411.10565v3)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.

摘要: 大型语言模型（LLM）生成的数据越来越多地用于软件分析，但目前尚不清楚该数据与人类编写的数据相比如何，特别是当模型暴露于对抗场景时。对抗性攻击可能会损害软件系统的可靠性和安全性，因此，与作为模型性能基准的人类编写数据相比，了解LLM生成的数据在这些条件下的表现如何，可以为LLM生成的数据是否提供类似的稳健性和有效性提供有价值的见解。为了解决这一差距，我们系统地评估和比较人类编写的数据和LLM生成的数据的质量，以便在对抗性攻击的背景下微调稳健的预训练模型（Ptms）。我们评估了六种广泛使用的PtM的稳健性，这些PtM在对抗性攻击之前和之后根据人类编写和LLM生成的数据进行了微调。该评估在三个流行的软件分析任务中使用了九种最先进的（SOTA）对抗性攻击技术：克隆检测，代码摘要和代码审查讨论中的情感分析。此外，我们使用11个相似性度量来分析生成的对抗性示例的质量。我们的研究结果表明，虽然对LLM生成的数据进行微调的PTM与对人类编写的数据进行微调的PTM具有竞争力，但它们在软件分析任务中对对抗性攻击的鲁棒性较低。我们的研究强调了进一步探索提高LLM生成的训练数据质量的必要性，以开发高性能且能够抵御软件分析中的对抗性攻击的模型。



## **24. Adversarial Robustness Analysis of Vision-Language Models in Medical Image Segmentation**

医学图像分割中视觉语言模型的对抗鲁棒性分析 cs.CV

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02971v1) [paper-pdf](http://arxiv.org/pdf/2505.02971v1)

**Authors**: Anjila Budathoki, Manish Dhakal

**Abstract**: Adversarial attacks have been fairly explored for computer vision and vision-language models. However, the avenue of adversarial attack for the vision language segmentation models (VLSMs) is still under-explored, especially for medical image analysis.   Thus, we have investigated the robustness of VLSMs against adversarial attacks for 2D medical images with different modalities with radiology, photography, and endoscopy. The main idea of this project was to assess the robustness of the fine-tuned VLSMs specially in the medical domain setting to address the high risk scenario.   First, we have fine-tuned pre-trained VLSMs for medical image segmentation with adapters.   Then, we have employed adversarial attacks -- projected gradient descent (PGD) and fast gradient sign method (FGSM) -- on that fine-tuned model to determine its robustness against adversaries.   We have reported models' performance decline to analyze the adversaries' impact.   The results exhibit significant drops in the DSC and IoU scores after the introduction of these adversaries. Furthermore, we also explored universal perturbation but were not able to find for the medical images.   \footnote{https://github.com/anjilab/secure-private-ai}

摘要: 计算机视觉和视觉语言模型的对抗性攻击已经得到了充分的探索。然而，视觉语言分割模型（VLSM）的对抗攻击途径仍然没有得到充分的探索，尤其是对于医学图像分析。   因此，我们研究了VLSM对放射学、摄影和内窥镜检查等不同模式的2D医学图像对抗攻击的稳健性。该项目的主要想法是评估微调VLSM的稳健性，特别是在医疗领域环境中，以应对高风险场景。   首先，我们对预训练的VLSM进行了微调，用于使用适配器进行医学图像分割。   然后，我们对该微调模型采用了对抗攻击--投影梯度下降（PVD）和快速梯度符号法（FGSM）--以确定其对对手的鲁棒性。   我们报告了模型的性能下降，以分析对手的影响。   结果显示，引入这些对手后，DSA和IoU分数显着下降。此外，我们还探索了普遍扰动，但未能找到医学图像。   \脚注{https：//github.com/anjilab/secure-private-ai}



## **25. Constrained Adversarial Learning for Automated Software Testing: a literature review**

用于自动化软件测试的约束对抗学习：文献综述 cs.SE

36 pages, 4 tables, 2 figures, Discover Applied Sciences journal

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2303.07546v2) [paper-pdf](http://arxiv.org/pdf/2303.07546v2)

**Authors**: João Vitorino, Tiago Dias, Tiago Fonseca, Eva Maia, Isabel Praça

**Abstract**: It is imperative to safeguard computer applications and information systems against the growing number of cyber-attacks. Automated software testing tools can be developed to quickly analyze many lines of code and detect vulnerabilities by generating function-specific testing data. This process draws similarities to the constrained adversarial examples generated by adversarial machine learning methods, so there could be significant benefits to the integration of these methods in testing tools to identify possible attack vectors. Therefore, this literature review is focused on the current state-of-the-art of constrained data generation approaches applied for adversarial learning and software testing, aiming to guide researchers and developers to enhance their software testing tools with adversarial testing methods and improve the resilience and robustness of their information systems. The found approaches were systematized, and the advantages and limitations of those specific for white-box, grey-box, and black-box testing were analyzed, identifying research gaps and opportunities to automate the testing tools with data generated by adversarial attacks.

摘要: 保护计算机应用程序和信息系统免受日益增多的网络攻击至关重要。可以开发自动化软件测试工具来快速分析多行代码并通过生成特定于功能的测试数据来检测漏洞。该过程与对抗性机器学习方法生成的受约束对抗示例具有相似之处，因此将这些方法集成到测试工具中以识别可能的攻击载体可能会带来显着的好处。因此，本次文献综述重点关注当前应用于对抗性学习和软件测试的约束数据生成方法的最新发展水平，旨在指导研究人员和开发人员通过对抗性测试方法增强其软件测试工具，并提高其信息系统的弹性和稳健性。对所发现的方法进行了系统化，并分析了白盒、灰盒和黑盒测试方法的优点和局限性，确定了研究差距和机会，以利用对抗性攻击生成的数据自动化测试工具。



## **26. Commitment Attacks on Ethereum's Reward Mechanism**

对以太坊奖励机制的承诺攻击 cs.CR

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2407.19479v2) [paper-pdf](http://arxiv.org/pdf/2407.19479v2)

**Authors**: Roozbeh Sarenche, Ertem Nusret Tas, Barnabe Monnot, Caspar Schwarz-Schilling, Bart Preneel

**Abstract**: Validators in permissionless, large-scale blockchains, such as Ethereum, are typically payoff-maximizing, rational actors. Ethereum relies on in-protocol incentives, like rewards for correct and timely votes, to induce honest behavior and secure the blockchain. However, external incentives, such as the block proposer's opportunity to capture maximal extractable value (MEV), may tempt validators to deviate from honest protocol participation.   We show a series of commitment attacks on LMD GHOST, a core part of Ethereum's consensus mechanism. We demonstrate how a single adversarial block proposer can orchestrate long-range chain reorganizations by manipulating Ethereum's reward system for timely votes. These attacks disrupt the intended balance of power between proposers and voters: by leveraging credible threats, the adversarial proposer can coerce voters from previous slots into supporting blocks that conflict with the honest chain, enabling a chain reorganization.   In response, we introduce a novel reward mechanism that restores the voters' role as a check against proposer power. Our proposed mitigation is fairer and more decentralized, not only in the context of these attacks, but also practical for implementation in Ethereum.

摘要: 以太坊等无需许可的大型区块链中的验证者通常是回报最大化的理性参与者。以太坊依赖于协议内激励，例如对正确和及时投票的奖励，来诱导诚实行为并保护区块链。然而，外部激励，例如区块提议者捕获最大可提取值（MEV）的机会，可能会引诱验证者偏离诚实协议参与。   我们展示了对LMD Ghost（以太坊共识机制的核心部分）的一系列承诺攻击。我们展示了单个对抗性区块提案者如何通过操纵以太坊的及时投票奖励系统来策划长期连锁重组。这些攻击破坏了提议者和选民之间预期的权力平衡：通过利用可信的威胁，对抗性提议者可以强迫选民从之前的位置进入与诚实链冲突的支持区块，从而实现链重组。   作为回应，我们引入了一种新颖的奖励机制，恢复选民对提案人权力的制衡作用。我们提出的缓解措施更加公平、更加分散，不仅在这些攻击的背景下，而且对于在以太坊中的实施也是可行的。



## **27. Robustness questions the interpretability of graph neural networks: what to do?**

鲁棒性质疑图神经网络的可解释性：该怎么办？ cs.LG

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02566v1) [paper-pdf](http://arxiv.org/pdf/2505.02566v1)

**Authors**: Kirill Lukyanov, Georgii Sazonov, Serafim Boyarsky, Ilya Makarov

**Abstract**: Graph Neural Networks (GNNs) have become a cornerstone in graph-based data analysis, with applications in diverse domains such as bioinformatics, social networks, and recommendation systems. However, the interplay between model interpretability and robustness remains poorly understood, especially under adversarial scenarios like poisoning and evasion attacks. This paper presents a comprehensive benchmark to systematically analyze the impact of various factors on the interpretability of GNNs, including the influence of robustness-enhancing defense mechanisms.   We evaluate six GNN architectures based on GCN, SAGE, GIN, and GAT across five datasets from two distinct domains, employing four interpretability metrics: Fidelity, Stability, Consistency, and Sparsity. Our study examines how defenses against poisoning and evasion attacks, applied before and during model training, affect interpretability and highlights critical trade-offs between robustness and interpretability. The framework will be published as open source.   The results reveal significant variations in interpretability depending on the chosen defense methods and model architecture characteristics. By establishing a standardized benchmark, this work provides a foundation for developing GNNs that are both robust to adversarial threats and interpretable, facilitating trust in their deployment in sensitive applications.

摘要: 图形神经网络（GNN）已成为基于图形的数据分析的基石，应用于生物信息学、社交网络和推荐系统等各个领域。然而，模型可解释性和稳健性之间的相互作用仍然知之甚少，尤其是在中毒和规避攻击等对抗场景下。本文提出了一个全面的基准来系统地分析各种因素对GNN可解释性的影响，包括鲁棒性增强防御机制的影响。   我们在来自两个不同领域的五个数据集上评估了基于GCN、SAGE、GIN和GAT的六种GNN架构，采用四种可解释性指标：富达性、稳定性、一致性和稀疏性。我们的研究考察了模型训练之前和期间应用的针对中毒和规避攻击的防御措施如何影响可解释性，并强调了稳健性和可解释性之间的关键权衡。该框架将以开源形式发布。   结果揭示了可解释性的显着差异，具体取决于所选择的防御方法和模型架构特征。通过建立标准化的基准，这项工作为开发既对对抗威胁稳健又可解释的GNN提供了基础，从而促进了对其在敏感应用程序中部署的信任。



## **28. Bayesian Robust Aggregation for Federated Learning**

用于联邦学习的Bayesian稳健聚集 cs.LG

14 pages, 4 figures, 8 tables

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02490v1) [paper-pdf](http://arxiv.org/pdf/2505.02490v1)

**Authors**: Aleksandr Karakulev, Usama Zafar, Salman Toor, Prashant Singh

**Abstract**: Federated Learning enables collaborative training of machine learning models on decentralized data. This scheme, however, is vulnerable to adversarial attacks, when some of the clients submit corrupted model updates. In real-world scenarios, the total number of compromised clients is typically unknown, with the extent of attacks potentially varying over time. To address these challenges, we propose an adaptive approach for robust aggregation of model updates based on Bayesian inference. The mean update is defined by the maximum of the likelihood marginalized over probabilities of each client to be `honest'. As a result, the method shares the simplicity of the classical average estimators (e.g., sample mean or geometric median), being independent of the number of compromised clients. At the same time, it is as effective against attacks as methods specifically tailored to Federated Learning, such as Krum. We compare our approach with other aggregation schemes in federated setting on three benchmark image classification data sets. The proposed method consistently achieves state-of-the-art performance across various attack types with static and varying number of malicious clients.

摘要: 联合学习能够在去中心化数据上对机器学习模型进行协作训练。然而，当一些客户端提交损坏的模型更新时，该计划很容易受到对抗攻击。在现实世界的场景中，受攻击客户端的总数通常是未知的，攻击的程度可能会随着时间的推移而变化。为了解决这些挑战，我们提出了一种基于Bayesian推理的模型更新稳健聚合的自适应方法。平均更新由每个客户“诚实”的可能性的最大值定义。因此，该方法具有经典平均估计量的简单性（例如，样本平均值或几何中位数），与受影响客户的数量无关。与此同时，它与专门为联邦学习量身定制的方法（例如Krum）一样有效。我们在三个基准图像分类数据集上将我们的方法与联邦环境中的其他聚合方案进行了比较。所提出的方法在具有静态和不同数量的恶意客户端的各种攻击类型中始终实现最先进的性能。



## **29. Economic Security of Multiple Shared Security Protocols**

多个共享安全协议的经济安全 cs.CR

21 pages, 6 figures

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.03843v1) [paper-pdf](http://arxiv.org/pdf/2505.03843v1)

**Authors**: Abhimanyu Nag, Dhruv Bodani, Abhishek Kumar

**Abstract**: As restaking protocols gain adoption across blockchain ecosystems, there is a need for Actively Validated Services (AVSs) to span multiple Shared Security Providers (SSPs). This leads to stake fragmentation which introduces new complications where an adversary may compromise an AVS by targeting its weakest SSP. In this paper, we formalize the Multiple SSP Problem and analyze two architectures : an isolated fragmented model called Model $\mathbb{M}$ and a shared unified model called Model $\mathbb{S}$, through a convex optimization and game-theoretic lens. We derive utility bounds, attack cost conditions, and market equilibrium that describes protocol security for both models. Our results show that while Model $\mathbb{M}$ offers deployment flexibility, it inherits lowest-cost attack vulnerabilities, whereas Model $\mathbb{S}$ achieves tighter security guarantees through single validator sets and aggregated slashing logic. We conclude with future directions of work including an incentive-compatible stake rebalancing allocation in restaking ecosystems.

摘要: 随着重新质押协议在区块链生态系统中的采用，需要主动验证服务（AVS）跨越多个共享安全提供商（SSP）。这导致股权碎片化，这引入了新的复杂性，其中对手可能通过瞄准其最弱的SSP来损害AVS。在本文中，我们形式化的多SSP问题和分析两个架构：一个孤立的碎片模型称为模型$\mathbb{M}$和一个共享的统一模型称为模型$\mathbb{S}$，通过凸优化和博弈论的镜头。我们推导出效用界，攻击成本条件，和市场均衡，描述了这两种模式的协议安全性。我们的结果表明，虽然模型$\mathbb{M}$提供了部署灵活性，但它继承了成本最低的攻击漏洞，而模型$\mathbb{S}$通过单个验证器集和聚合削减逻辑实现了更严格的安全保证。我们总结了未来的工作方向，包括激励兼容的股权重新平衡重新押注生态系统的分配。



## **30. Catastrophic Overfitting, Entropy Gap and Participation Ratio: A Noiseless $l^p$ Norm Solution for Fast Adversarial Training**

灾难性的过度匹配、熵差和参与率：快速对抗训练的无声$l & p$ Norm解决方案 cs.LG

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02360v1) [paper-pdf](http://arxiv.org/pdf/2505.02360v1)

**Authors**: Fares B. Mehouachi, Saif Eddin Jabari

**Abstract**: Adversarial training is a cornerstone of robust deep learning, but fast methods like the Fast Gradient Sign Method (FGSM) often suffer from Catastrophic Overfitting (CO), where models become robust to single-step attacks but fail against multi-step variants. While existing solutions rely on noise injection, regularization, or gradient clipping, we propose a novel solution that purely controls the $l^p$ training norm to mitigate CO.   Our study is motivated by the empirical observation that CO is more prevalent under the $l^{\infty}$ norm than the $l^2$ norm. Leveraging this insight, we develop a framework for generalized $l^p$ attack as a fixed point problem and craft $l^p$-FGSM attacks to understand the transition mechanics from $l^2$ to $l^{\infty}$. This leads to our core insight: CO emerges when highly concentrated gradients where information localizes in few dimensions interact with aggressive norm constraints. By quantifying gradient concentration through Participation Ratio and entropy measures, we develop an adaptive $l^p$-FGSM that automatically tunes the training norm based on gradient information. Extensive experiments demonstrate that this approach achieves strong robustness without requiring additional regularization or noise injection, providing a novel and theoretically-principled pathway to mitigate the CO problem.

摘要: 对抗性训练是稳健深度学习的基石，但像快速梯度符号法（FGSM）这样的快速方法经常受到灾难性过适应（CO）的影响，即模型对单步攻击变得稳健，但对多步变体却失败。虽然现有的解决方案依赖于噪音注入、正规化或梯度限幅，但我们提出了一种新颖的解决方案，该解决方案纯粹控制$l ' p$训练规范以减轻CO。   我们的研究的动机是经验观察，即CO在$l &{\infty}$规范下比在$l & 2 $规范下更普遍。利用这一见解，我们开发了一个将广义$l ' p$攻击作为定点问题的框架，并精心设计了$l ' p$-FGSM攻击，以了解从$l ' 2 $到$l '#'#'的转变机制。这引出了我们的核心见解：当信息局部化在少数维度上的高度集中的梯度与激进的规范约束相互作用时，CO就会出现。通过通过参与率和信息量量化梯度集中度，我们开发了一种自适应的$l ' p$-FGSM，它可以根据梯度信息自动调整训练规范。大量实验表明，这种方法在不需要额外的正规化或噪音注入的情况下实现了很强的鲁棒性，提供了一种新颖且有理论原则的途径来缓解CO问题。



## **31. Bayes-Nash Generative Privacy Against Membership Inference Attacks**

针对会员推断攻击的Bayes-Nash生成隐私 cs.CR

arXiv admin note: substantial text overlap with arXiv:2406.01811

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2410.07414v4) [paper-pdf](http://arxiv.org/pdf/2410.07414v4)

**Authors**: Tao Zhang, Rajagopal Venkatesaramani, Rajat K. De, Bradley A. Malin, Yevgeniy Vorobeychik

**Abstract**: Membership inference attacks (MIAs) expose significant privacy risks by determining whether an individual's data is in a dataset. While differential privacy (DP) mitigates such risks, it has several limitations in achieving an optimal balance between utility and privacy, include limited resolution in expressing this tradeoff in only a few privacy parameters, and intractable sensitivity calculations that may be necessary to provide tight privacy guarantees. We propose a game-theoretic framework that models privacy protection from MIA as a Bayesian game between a defender and an attacker. In this game, a dataset is the defender's private information, with privacy loss to the defender (which is gain to the attacker) captured in terms of the attacker's ability to infer membership of individuals in the dataset. To address the strategic complexity of this game, we represent the mixed strategy of the defender as a neural network generator which maps a private dataset to its public representation (for example, noisy summary statistics), while the mixed strategy of the attacker is captured by a discriminator which makes membership inference claims. We refer to the resulting computational approach as a general-sum Generative Adversarial Network, which is trained iteratively by alternating generator and discriminator updates akin to conventional GANs. We call the defender's data sharing policy thereby obtained Bayes-Nash Generative Privacy (BNGP). The BNGP strategy avoids sensitivity calculations, supports compositions of correlated mechanisms, is robust to the attacker's heterogeneous preferences over true and false positives, and yields provable differential privacy guarantees, albeit in an idealized setting.

摘要: 成员资格推断攻击（MIA）通过确定个人的数据是否在数据集中暴露了重大的隐私风险。虽然差异隐私（DP）可以减轻此类风险，但它在实现效用和隐私之间的最佳平衡方面存在一些局限性，包括仅用少数隐私参数表达这种权衡的分辨率有限，以及提供严格隐私保证可能需要的棘手敏感性计算。我们提出了一个博弈论框架，将MIA的隐私保护建模为防御者和攻击者之间的Bayesian博弈。在这个游戏中，数据集是防御者的私人信息，防御者的隐私损失（这是攻击者的收益）根据攻击者推断数据集中个人成员资格的能力来捕捉。为了解决这个游戏的策略复杂性，我们将防御者的混合策略表示为神经网络生成器，该生成器将私人数据集映射到其公共表示（例如，有噪的摘要统计数据），而攻击者的混合策略则由发起成员资格推断的搜索器捕获。我们将由此产生的计算方法称为通用和生成对抗网络，它通过类似于传统GAN的交替生成器和RST更新进行迭代训练。我们将由此获得的防御者的数据共享政策称为Bayes-Nash生成隐私（BNGP）。BCGP策略避免了敏感性计算，支持相关机制的组合，对攻击者对真阳性和假阳性的不同偏好具有鲁棒性，并产生可证明的差异隐私保证（尽管是在理想化的环境中）。



## **32. Open Challenges in Multi-Agent Security: Towards Secure Systems of Interacting AI Agents**

多代理安全面临的开放挑战：迈向交互式人工智能代理的安全系统 cs.CR

**SubmitDate**: 2025-05-04    [abs](http://arxiv.org/abs/2505.02077v1) [paper-pdf](http://arxiv.org/pdf/2505.02077v1)

**Authors**: Christian Schroeder de Witt

**Abstract**: Decentralized AI agents will soon interact across internet platforms, creating security challenges beyond traditional cybersecurity and AI safety frameworks. Free-form protocols are essential for AI's task generalization but enable new threats like secret collusion and coordinated swarm attacks. Network effects can rapidly spread privacy breaches, disinformation, jailbreaks, and data poisoning, while multi-agent dispersion and stealth optimization help adversaries evade oversightcreating novel persistent threats at a systemic level. Despite their critical importance, these security challenges remain understudied, with research fragmented across disparate fields including AI security, multi-agent learning, complex systems, cybersecurity, game theory, distributed systems, and technical AI governance. We introduce \textbf{multi-agent security}, a new field dedicated to securing networks of decentralized AI agents against threats that emerge or amplify through their interactionswhether direct or indirect via shared environmentswith each other, humans, and institutions, and characterize fundamental security-performance trade-offs. Our preliminary work (1) taxonomizes the threat landscape arising from interacting AI agents, (2) surveys security-performance tradeoffs in decentralized AI systems, and (3) proposes a unified research agenda addressing open challenges in designing secure agent systems and interaction environments. By identifying these gaps, we aim to guide research in this critical area to unlock the socioeconomic potential of large-scale agent deployment on the internet, foster public trust, and mitigate national security risks in critical infrastructure and defense contexts.

摘要: 去中心化的人工智能代理很快将在互联网平台上互动，从而带来超越传统网络安全和人工智能安全框架的安全挑战。自由形式的协议对于人工智能的任务概括至关重要，但也会产生秘密共谋和协同群攻击等新威胁。网络效应可以迅速传播隐私泄露、虚假信息、越狱和数据中毒，而多代理分散和隐形优化帮助对手逃避疏忽，从而在系统层面上产生新颖的持续威胁。尽管这些安全挑战至关重要，但研究仍然不足，研究分散在不同领域，包括人工智能安全、多代理学习、复杂系统、网络安全、博弈论、分布式系统和技术人工智能治理。我们引入了\textBF{多代理安全}，这是一个新领域，致力于保护去中心化人工智能代理网络免受通过相互作用而出现或放大的威胁（无论是直接还是间接通过与彼此、人类和机构的共享环境），并描述了基本的安全性能权衡。我们的初步工作（1）对交互人工智能代理产生的威胁格局进行分类，（2）调查去中心化人工智能系统中的安全性能权衡，（3）提出了一个统一的研究议程，解决设计安全代理系统和交互环境中的开放挑战。通过识别这些差距，我们的目标是指导这一关键领域的研究，以释放互联网上大规模代理部署的社会经济潜力，促进公众信任，并减轻关键基础设施和国防环境中的国家安全风险。



## **33. Lightweight Defense Against Adversarial Attacks in Time Series Classification**

时间序列分类中对抗攻击的轻量级防御 cs.LG

13 pages, 8 figures. Accepted at RAFDA Workshop, PAKDD 2025  (Springer, EI & Scopus indexed). Code:  https://github.com/Yi126/Lightweight-Defence

**SubmitDate**: 2025-05-04    [abs](http://arxiv.org/abs/2505.02073v1) [paper-pdf](http://arxiv.org/pdf/2505.02073v1)

**Authors**: Yi Han

**Abstract**: As time series classification (TSC) gains prominence, ensuring robust TSC models against adversarial attacks is crucial. While adversarial defense is well-studied in Computer Vision (CV), the TSC field has primarily relied on adversarial training (AT), which is computationally expensive. In this paper, five data augmentation-based defense methods tailored for time series are developed, with the most computationally intensive method among them increasing the computational resources by only 14.07% compared to the original TSC model. Moreover, the deployment process for these methods is straightforward. By leveraging these advantages of our methods, we create two combined methods. One of these methods is an ensemble of all the proposed techniques, which not only provides better defense performance than PGD-based AT but also enhances the generalization ability of TSC models. Moreover, the computational resources required for our ensemble are less than one-third of those required for PGD-based AT. These methods advance robust TSC in data mining. Furthermore, as foundation models are increasingly explored for time series feature learning, our work provides insights into integrating data augmentation-based adversarial defense with large-scale pre-trained models in future research.

摘要: 随着时间序列分类（TSC）的日益突出，确保强大的TSC模型抵御对抗性攻击至关重要。虽然对抗性防御在计算机视觉（CV）中得到了很好的研究，但TSC领域主要依赖于对抗性训练（AT），这在计算上是昂贵的。本文提出了五种基于数据增强的时间序列防御方法，其中计算量最大的方法与原始TSC模型相比仅增加了14.07%的计算资源。此外，这些方法的部署过程很简单。通过利用我们方法的这些优势，我们创建了两种组合方法。其中一种方法是所有提出的技术的集成，它不仅提供比基于PGD的AT更好的防御性能，而且还增强了OSC模型的概括能力。此外，我们的集成所需的计算资源还不到基于PGD的AT所需的三分之一。这些方法在数据挖掘中推进了稳健的OSC。此外，随着基础模型被越来越多地探索用于时间序列特征学习，我们的工作为在未来的研究中将基于数据增强的对抗性防御与大规模预训练模型集成提供了见解。



## **34. A Comprehensive Analysis of Adversarial Attacks against Spam Filters**

针对垃圾邮件过滤器的对抗性攻击综合分析 cs.CR

**SubmitDate**: 2025-05-04    [abs](http://arxiv.org/abs/2505.03831v1) [paper-pdf](http://arxiv.org/pdf/2505.03831v1)

**Authors**: Esra Hotoğlu, Sevil Sen, Burcu Can

**Abstract**: Deep learning has revolutionized email filtering, which is critical to protect users from cyber threats such as spam, malware, and phishing. However, the increasing sophistication of adversarial attacks poses a significant challenge to the effectiveness of these filters. This study investigates the impact of adversarial attacks on deep learning-based spam detection systems using real-world datasets. Six prominent deep learning models are evaluated on these datasets, analyzing attacks at the word, character sentence, and AI-generated paragraph-levels. Novel scoring functions, including spam weights and attention weights, are introduced to improve attack effectiveness. This comprehensive analysis sheds light on the vulnerabilities of spam filters and contributes to efforts to improve their security against evolving adversarial threats.

摘要: 深度学习彻底改变了电子邮件过滤，这对于保护用户免受垃圾邮件、恶意软件和网络钓鱼等网络威胁至关重要。然而，对抗攻击的日益复杂，对这些过滤器的有效性构成了重大挑战。本研究使用现实世界数据集调查了对抗攻击对基于深度学习的垃圾邮件检测系统的影响。在这些数据集上评估了六个著名的深度学习模型，分析单词、字符句和人工智能生成的段落级别的攻击。引入了新颖的评分功能，包括垃圾邮件权重和注意力权重，以提高攻击有效性。这项全面的分析揭示了垃圾邮件过滤器的漏洞，并有助于提高其安全性，以应对不断变化的对抗威胁。



## **35. CAMOUFLAGE: Exploiting Misinformation Detection Systems Through LLM-driven Adversarial Claim Transformation**

CAMOUFLAGE：通过LLM驱动的对抗性主张转换开发错误信息检测系统 cs.CL

**SubmitDate**: 2025-05-03    [abs](http://arxiv.org/abs/2505.01900v1) [paper-pdf](http://arxiv.org/pdf/2505.01900v1)

**Authors**: Mazal Bethany, Nishant Vishwamitra, Cho-Yu Jason Chiang, Peyman Najafirad

**Abstract**: Automated evidence-based misinformation detection systems, which evaluate the veracity of short claims against evidence, lack comprehensive analysis of their adversarial vulnerabilities. Existing black-box text-based adversarial attacks are ill-suited for evidence-based misinformation detection systems, as these attacks primarily focus on token-level substitutions involving gradient or logit-based optimization strategies, which are incapable of fooling the multi-component nature of these detection systems. These systems incorporate both retrieval and claim-evidence comparison modules, which requires attacks to break the retrieval of evidence and/or the comparison module so that it draws incorrect inferences. We present CAMOUFLAGE, an iterative, LLM-driven approach that employs a two-agent system, a Prompt Optimization Agent and an Attacker Agent, to create adversarial claim rewritings that manipulate evidence retrieval and mislead claim-evidence comparison, effectively bypassing the system without altering the meaning of the claim. The Attacker Agent produces semantically equivalent rewrites that attempt to mislead detectors, while the Prompt Optimization Agent analyzes failed attack attempts and refines the prompt of the Attacker to guide subsequent rewrites. This enables larger structural and stylistic transformations of the text rather than token-level substitutions, adapting the magnitude of changes based on previous outcomes. Unlike existing approaches, CAMOUFLAGE optimizes its attack solely based on binary model decisions to guide its rewriting process, eliminating the need for classifier logits or extensive querying. We evaluate CAMOUFLAGE on four systems, including two recent academic systems and two real-world APIs, with an average attack success rate of 46.92\% while preserving textual coherence and semantic equivalence to the original claims.

摘要: 自动化的基于证据的错误信息检测系统根据证据评估简短主张的真实性，但缺乏对其对抗漏洞的全面分析。现有的基于黑匣子文本的对抗攻击不适合基于证据的错误信息检测系统，因为这些攻击主要集中在涉及梯度或基于逻辑的优化策略的标记级替换上，而这些策略无法愚弄这些检测系统的多组件性质。这些系统结合了检索和主张证据比较模块，这需要攻击破坏证据检索和/或比较模块，以便得出错误的推论。我们提出了CAMOUFLAGE，这是一种迭代的、LLM驱动的方法，它采用双代理系统、即时优化代理和攻击代理来创建对抗性主张重写，从而操纵证据检索并误导主张证据比较，有效地绕过系统而不改变主张的含义。Attacker Agent生成试图误导检测器的语义等效重写，而Prompt Optimization Agent分析失败的攻击尝试并改进Attacker的提示以指导后续重写。这使得文本能够进行更大的结构和风格转换，而不是标记级别的替换，从而根据先前的结果调整变化的幅度。与现有方法不同，CAMOUFLAGE仅基于二进制模型决策来优化其攻击，以指导其重写过程，从而消除了对分类器logits或广泛查询的需要。我们在四个系统上评估了CAMOUFLAGE，包括两个最近的学术系统和两个真实世界的API，平均攻击成功率为46.92%，同时保持了文本的连贯性和语义等价性。



## **36. PQS-BFL: A Post-Quantum Secure Blockchain-based Federated Learning Framework**

PQS-BFL：后量子安全的基于区块链的联邦学习框架 cs.CR

**SubmitDate**: 2025-05-03    [abs](http://arxiv.org/abs/2505.01866v1) [paper-pdf](http://arxiv.org/pdf/2505.01866v1)

**Authors**: Daniel Commey, Garth V. Crosby

**Abstract**: Federated Learning (FL) enables collaborative model training while preserving data privacy, but its classical cryptographic underpinnings are vulnerable to quantum attacks. This vulnerability is particularly critical in sensitive domains like healthcare. This paper introduces PQS-BFL (Post-Quantum Secure Blockchain-based Federated Learning), a framework integrating post-quantum cryptography (PQC) with blockchain verification to secure FL against quantum adversaries. We employ ML-DSA-65 (a FIPS 204 standard candidate, formerly Dilithium) signatures to authenticate model updates and leverage optimized smart contracts for decentralized validation. Extensive evaluations on diverse datasets (MNIST, SVHN, HAR) demonstrate that PQS-BFL achieves efficient cryptographic operations (average PQC sign time: 0.65 ms, verify time: 0.53 ms) with a fixed signature size of 3309 Bytes. Blockchain integration incurs a manageable overhead, with average transaction times around 4.8 s and gas usage per update averaging 1.72 x 10^6 units for PQC configurations. Crucially, the cryptographic overhead relative to transaction time remains minimal (around 0.01-0.02% for PQC with blockchain), confirming that PQC performance is not the bottleneck in blockchain-based FL. The system maintains competitive model accuracy (e.g., over 98.8% for MNIST with PQC) and scales effectively, with round times showing sublinear growth with increasing client numbers. Our open-source implementation and reproducible benchmarks validate the feasibility of deploying long-term, quantum-resistant security in practical FL systems.

摘要: 联邦学习（FL）支持协作模型训练，同时保护数据隐私，但其经典的密码学基础容易受到量子攻击。这种漏洞在医疗保健等敏感领域尤为严重。本文介绍了PQS-BFL（后量子安全基于区块链的联合学习），这是一个将后量子密码学（PQC）与区块链验证相结合的框架，以保护FL免受量子对手的攻击。我们使用ML-DSA-65（FIPS 204标准候选者，以前称为Dilithium）签名来验证模型更新，并利用优化的智能合约进行分散验证。对不同数据集（MNIST、SVHN、HAR）的广泛评估表明，PQS-BFL实现了高效的加密操作（平均PQC签名时间：0.65 ms，验证时间：0.53 ms），固定签名大小为3309。区块链集成会产生可管理的费用，PQC配置的平均交易时间约为4.8秒，每次更新的气体使用量平均为1.72 x 106单位。至关重要的是，相对于交易时间的加密费用仍然最小（对于采用区块链的PQC，约为0.01-0.02%），这证实了PQC性能并不是基于区块链的FL的瓶颈。该系统保持有竞争力的模型准确性（例如，配备PQC的MNIST超过98.8%），并且有效扩展，随着客户数量的增加，整周时间显示出亚线性增长。我们的开源实施和可重复基准验证了在实际FL系统中部署长期、抗量子安全性的可行性。



## **37. Rogue Cell: Adversarial Attack and Defense in Untrusted O-RAN Setup Exploiting the Traffic Steering xApp**

Rogue Cell：利用流量引导xApp的不受信任O-RAN设置中的对抗攻击和防御 cs.CR

**SubmitDate**: 2025-05-03    [abs](http://arxiv.org/abs/2505.01816v1) [paper-pdf](http://arxiv.org/pdf/2505.01816v1)

**Authors**: Eran Aizikovich, Dudu Mimran, Edita Grolman, Yuval Elovici, Asaf Shabtai

**Abstract**: The Open Radio Access Network (O-RAN) architecture is revolutionizing cellular networks with its open, multi-vendor design and AI-driven management, aiming to enhance flexibility and reduce costs. Although it has many advantages, O-RAN is not threat-free. While previous studies have mainly examined vulnerabilities arising from O-RAN's intelligent components, this paper is the first to focus on the security challenges and vulnerabilities introduced by transitioning from single-operator to multi-operator RAN architectures. This shift increases the risk of untrusted third-party operators managing different parts of the network. To explore these vulnerabilities and their potential mitigation, we developed an open-access testbed environment that integrates a wireless network simulator with the official O-RAN Software Community (OSC) RAN intelligent component (RIC) cluster. This environment enables realistic, live data collection and serves as a platform for demonstrating APATE (adversarial perturbation against traffic efficiency), an evasion attack in which a malicious cell manipulates its reported key performance indicators (KPIs) and deceives the O-RAN traffic steering to gain unfair allocations of user equipment (UE). To ensure that O-RAN's legitimate activity continues, we introduce MARRS (monitoring adversarial RAN reports), a detection framework based on a long-short term memory (LSTM) autoencoder (AE) that learns contextual features across the network to monitor malicious telemetry (also demonstrated in our testbed). Our evaluation showed that by executing APATE, an attacker can obtain a 248.5% greater UE allocation than it was supposed to in a benign scenario. In addition, the MARRS detection method was also shown to successfully classify malicious cell activity, achieving accuracy of 99.2% and an F1 score of 0.978.

摘要: 开放无线电接入网络（O-RAN）架构凭借其开放的多供应商设计和人工智能驱动的管理，正在彻底改变蜂窝网络，旨在增强灵活性并降低成本。尽管O-RAN有很多优势，但它并非没有威胁。虽然之前的研究主要研究了O-RAN智能组件产生的漏洞，但本文首次关注从单运营商RAN架构过渡到多运营商RAN架构所带来的安全挑战和漏洞。这种转变增加了不受信任的第三方运营商管理网络不同部分的风险。为了探索这些漏洞及其潜在的缓解措施，我们开发了一个开放访问测试台环境，该环境将无线网络模拟器与官方O-RAN软件社区（OSC）RAN智能组件（RIC）集群集成。该环境能够实现真实的实时数据收集，并充当演示APATE（针对流量效率的对抗性扰动）的平台，APATE是一种规避攻击，其中恶意蜂窝操纵其报告的关键性能指标（KPI）并欺骗O-RAN流量引导以获得用户设备（UE）的不公平分配。为了确保O-RAN的合法活动继续进行，我们引入了MARRS（监控对抗性RAN报告），这是一种基于长短期记忆（LSTM）自动编码器（AE）的检测框架，可以学习整个网络的上下文特征以监控恶意遥感（也在我们的测试床上演示）。我们的评估表明，通过执行APATE，攻击者可以获得比良性情况下预期多248.5%的UE分配。此外，MARRS检测方法还被证明可以成功分类恶意细胞活动，准确率为99.2%，F1评分为0.978。



## **38. LeapFrog: The Rowhammer Instruction Skip Attack**

LeapFrog：Rowhammer指令跳过攻击 cs.CR

Accepted at EuroS&P 2025 and Hardware.io 2024,

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2404.07878v3) [paper-pdf](http://arxiv.org/pdf/2404.07878v3)

**Authors**: Andrew Adiletta, M. Caner Tol, Kemal Derya, Berk Sunar, Saad Islam

**Abstract**: Since its inception, Rowhammer exploits have rapidly evolved into increasingly sophisticated threats compromising data integrity and the control flow integrity of victim processes. Nevertheless, it remains a challenge for an attacker to identify vulnerable targets (i.e., Rowhammer gadgets), understand the outcome of the attempted fault, and formulate an attack that yields useful results.   In this paper, we present a new type of Rowhammer gadget, called a LeapFrog gadget, which, when present in the victim code, allows an adversary to subvert code execution to bypass a critical piece of code (e.g., authentication check logic, encryption rounds, padding in security protocols). The LeapFrog gadget manifests when the victim code stores the Program Counter (PC) value in the user or kernel stack (e.g., a return address during a function call) which, when tampered with, repositions the return address to a location that bypasses a security-critical code pattern.   This research also presents a systematic process to identify LeapFrog gadgets. This methodology enables the automated detection of susceptible targets and the determination of optimal attack parameters. We first show the attack on a decision tree algorithm to show the potential implications. Secondly, we employ the attack on OpenSSL to bypass the encryption and reveal the plaintext. We then use our tools to scan the Open Quantum Safe library and report on the number of LeapFrog gadgets in the code. Lastly, we demonstrate this new attack vector through a practical demonstration in a client/server TLS handshake scenario, successfully inducing an instruction skip in a client application. Our findings extend the impact of Rowhammer attacks on control flow and contribute to developing more robust defenses against these increasingly sophisticated threats.

摘要: 自成立以来，Rowhammer漏洞利用已迅速演变为日益复杂的威胁，损害了受害者流程的数据完整性和控制流完整性。然而，攻击者识别易受攻击的目标（即，Rowhammer小工具），了解尝试错误的结果，并制定产生有用结果的攻击。   在本文中，我们提出了一种新型Rowhammer小工具，称为LeapFrog小工具，当它出现在受害者代码中时，它允许对手颠覆代码执行以绕过关键代码段（例如，身份验证检查逻辑、加密回合、安全协议中的填充）。当受害者代码将程序计数器（PC）值存储在用户或内核堆栈中时，LeapFrog小工具就会显现（例如，函数调用期间的返回地址），当被篡改时，会将返回地址重新定位到绕过安全关键代码模式的位置。   这项研究还提出了一个识别LeapFrog小工具的系统过程。该方法能够自动检测易感目标并确定最佳攻击参数。我们首先展示对决策树算法的攻击，以展示潜在的影响。其次，我们利用对OpenSSL的攻击来绕过加密并揭示明文。然后，我们使用我们的工具扫描Open Quantum Safe库并报告代码中LeapFrog小工具的数量。最后，我们通过在客户端/服务器TLS握手场景中的实际演示来演示这种新的攻击向量，成功地在客户端应用程序中诱导指令跳过。我们的研究结果扩展了Rowhammer攻击对控制流的影响，并有助于开发更强大的防御措施来应对这些日益复杂的威胁。



## **39. Modeling Behavioral Preferences of Cyber Adversaries Using Inverse Reinforcement Learning**

使用反向强化学习建模网络对手的行为偏好 cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.03817v1) [paper-pdf](http://arxiv.org/pdf/2505.03817v1)

**Authors**: Aditya Shinde, Prashant Doshi

**Abstract**: This paper presents a holistic approach to attacker preference modeling from system-level audit logs using inverse reinforcement learning (IRL). Adversary modeling is an important capability in cybersecurity that lets defenders characterize behaviors of potential attackers, which enables attribution to known cyber adversary groups. Existing approaches rely on documenting an ever-evolving set of attacker tools and techniques to track known threat actors. Although attacks evolve constantly, attacker behavioral preferences are intrinsic and less volatile. Our approach learns the behavioral preferences of cyber adversaries from forensics data on their tools and techniques. We model the attacker as an expert decision-making agent with unknown behavioral preferences situated in a computer host. We leverage attack provenance graphs of audit logs to derive a state-action trajectory of the attack. We test our approach on open datasets of audit logs containing real attack data. Our results demonstrate for the first time that low-level forensics data can automatically reveal an adversary's subjective preferences, which serves as an additional dimension to modeling and documenting cyber adversaries. Attackers' preferences tend to be invariant despite their different tools and indicate predispositions that are inherent to the attacker. As such, these inferred preferences can potentially serve as unique behavioral signatures of attackers and improve threat attribution.

摘要: 本文提出了一种使用反向强化学习（IRL）从系统级审计日志中进行攻击者偏好建模的整体方法。敌对者建模是网络安全领域的一项重要功能，可以让防御者描述潜在攻击者的行为，从而能够归因于已知的网络对手群体。现有的方法依赖于记录一组不断发展的攻击者工具和技术来跟踪已知的威胁参与者。尽管攻击不断发展，但攻击者的行为偏好是固有的，波动性较小。我们的方法从网络对手工具和技术的取证数据中学习网络对手的行为偏好。我们将攻击者建模为位于计算机主机中具有未知行为偏好的专家决策代理。我们利用审计日志的攻击来源图来推导攻击的状态动作轨迹。我们在包含真实攻击数据的审计日志开放数据集上测试我们的方法。我们的结果首次证明，低级取证数据可以自动揭示对手的主观偏好，这是建模和记录网络对手的额外维度。尽管攻击者的工具不同，但其偏好往往是不变的，并表明攻击者固有的倾向。因此，这些推断的偏好可能会作为攻击者的独特行为签名并改善威胁归因。



## **40. Machine Learning for Cyber-Attack Identification from Traffic Flows**

从流量中识别网络攻击的机器学习 cs.LG

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01489v1) [paper-pdf](http://arxiv.org/pdf/2505.01489v1)

**Authors**: Yujing Zhou, Marc L. Jacquet, Robel Dawit, Skyler Fabre, Dev Sarawat, Faheem Khan, Madison Newell, Yongxin Liu, Dahai Liu, Hongyun Chen, Jian Wang, Huihui Wang

**Abstract**: This paper presents our simulation of cyber-attacks and detection strategies on the traffic control system in Daytona Beach, FL. using Raspberry Pi virtual machines and the OPNSense firewall, along with traffic dynamics from SUMO and exploitation via the Metasploit framework. We try to answer the research questions: are we able to identify cyber attacks by only analyzing traffic flow patterns. In this research, the cyber attacks are focused particularly when lights are randomly turned all green or red at busy intersections by adversarial attackers. Despite challenges stemming from imbalanced data and overlapping traffic patterns, our best model shows 85\% accuracy when detecting intrusions purely using traffic flow statistics. Key indicators for successful detection included occupancy, jam length, and halting durations.

摘要: 本文介绍了我们使用Raspberry Pi虚拟机和OPNSSense防火墙对佛罗里达州代托纳海滩交通控制系统的网络攻击和检测策略的模拟，以及SUMO的流量动态和通过Metasploit框架的开发。我们试图回答研究问题：我们是否能够仅通过分析流量模式来识别网络攻击。在这项研究中，网络攻击尤其是当敌对攻击者将繁忙十字路口的灯光随机全绿或变红时。尽管数据不平衡和重叠的流量模式带来了挑战，但当纯粹使用流量统计来检测入侵时，我们的最佳模型显示出85%的准确性。成功检测的关键指标包括占用率、堵塞长度和停止持续时间。



## **41. Synthesizing Grid Data with Cyber Resilience and Privacy Guarantees**

综合具有网络弹性和隐私保证的网格数据 eess.SY

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2503.14877v2) [paper-pdf](http://arxiv.org/pdf/2503.14877v2)

**Authors**: Shengyang Wu, Vladimir Dvorkin

**Abstract**: Differential privacy (DP) provides a principled approach to synthesizing data (e.g., loads) from real-world power systems while limiting the exposure of sensitive information. However, adversaries may exploit synthetic data to calibrate cyberattacks on the source grids. To control these risks, we propose new DP algorithms for synthesizing data that provide the source grids with both cyber resilience and privacy guarantees. The algorithms incorporate both normal operation and attack optimization models to balance the fidelity of synthesized data and cyber resilience. The resulting post-processing optimization is reformulated as a robust optimization problem, which is compatible with the exponential mechanism of DP to moderate its computational burden.

摘要: 差异隐私（DP）提供了一种有原则的方法来合成数据（例如，负载）来自现实世界的电力系统，同时限制敏感信息的暴露。然而，对手可能会利用合成数据来校准对源网格的网络攻击。为了控制这些风险，我们提出了新的DP算法来合成数据，为源网格提供网络弹性和隐私保证。这些算法结合了正常操作和攻击优化模型，以平衡合成数据的保真度和网络弹性。所得的后处理优化被重新表述为鲁棒优化问题，该问题与DP的指数机制兼容，以减轻其计算负担。



## **42. Deep Learning-Enabled System Diagnosis in Microgrids: A Feature-Feedback GAN Approach**

微电网中支持深度学习的系统诊断：激励反馈GAN方法 eess.SY

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01366v1) [paper-pdf](http://arxiv.org/pdf/2505.01366v1)

**Authors**: Swetha Rani Kasimalla, Kuchan Park, Junho Hong, Young-Jin Kim, HyoJong Lee

**Abstract**: The increasing integration of inverter-based resources (IBRs) and communication networks has brought both modernization and new vulnerabilities to the power system infrastructure. These vulnerabilities expose the system to internal faults and cyber threats, particularly False Data Injection (FDI) attacks, which can closely mimic real fault scenarios. Hence, this work presents a two-stage fault and cyberattack detection framework tailored for inverter-based microgrids. Stage 1 introduces an unsupervised learning model Feature Feedback Generative Adversarial Network (F2GAN), to distinguish between genuine internal faults and cyber-induced anomalies in microgrids. Compared to conventional GAN architectures, F2GAN demonstrates improved system diagnosis and greater adaptability to zero-day attacks through its feature-feedback mechanism. In Stage 2, supervised machine learning techniques, including Support Vector Machines (SVM), k-Nearest Neighbors (KNN), Decision Trees (DT), and Artificial Neural Networks (ANN) are applied to localize and classify faults within inverter switches, distinguishing between single-switch and multi-switch faults. The proposed framework is validated on a simulated microgrid environment, illustrating robust performance in detecting and classifying both physical and cyber-related disturbances in power electronic-dominated systems.

摘要: 基于逆变器的资源（IBR）和通信网络的日益集成给电力系统基础设施带来了现代化和新的脆弱性。这些漏洞使系统面临内部故障和网络威胁，特别是虚假数据注入（FDI）攻击，它可以密切模仿真实的故障场景。因此，这项工作提出了一个为基于逆变器的微电网量身定制的两阶段故障和网络攻击检测框架。第一阶段引入了无监督学习模型特征反馈生成对抗网络（F2 GAN），以区分微电网中真正的内部故障和网络引发的异常。与传统GAN架构相比，F2 GAN通过其特征反馈机制展示了改进的系统诊断和对零日攻击的更强适应性。在第二阶段，应用监督机器学习技术（包括支持向量机（支持量机）、k近邻（KNN）、决策树（DT）和人工神经网络（NN））来定位和分类逆变器开关内的故障，区分单开关和多开关故障。所提出的框架在模拟微电网环境上得到了验证，说明了在电力电子主导的系统中检测和分类物理和网络相关干扰方面的稳健性能。



## **43. Constrained Network Adversarial Attacks: Validity, Robustness, and Transferability**

约束网络对抗攻击：有效性、稳健性和可移植性 cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01328v1) [paper-pdf](http://arxiv.org/pdf/2505.01328v1)

**Authors**: Anass Grini, Oumaima Taheri, Btissam El Khamlichi, Amal El Fallah-Seghrouchni

**Abstract**: While machine learning has significantly advanced Network Intrusion Detection Systems (NIDS), particularly within IoT environments where devices generate large volumes of data and are increasingly susceptible to cyber threats, these models remain vulnerable to adversarial attacks. Our research reveals a critical flaw in existing adversarial attack methodologies: the frequent violation of domain-specific constraints, such as numerical and categorical limits, inherent to IoT and network traffic. This leads to up to 80.3% of adversarial examples being invalid, significantly overstating real-world vulnerabilities. These invalid examples, though effective in fooling models, do not represent feasible attacks within practical IoT deployments. Consequently, relying on these results can mislead resource allocation for defense, inflating the perceived susceptibility of IoT-enabled NIDS models to adversarial manipulation. Furthermore, we demonstrate that simpler surrogate models like Multi-Layer Perceptron (MLP) generate more valid adversarial examples compared to complex architectures such as CNNs and LSTMs. Using the MLP as a surrogate, we analyze the transferability of adversarial severity to other ML/DL models commonly used in IoT contexts. This work underscores the importance of considering both domain constraints and model architecture when evaluating and designing robust ML/DL models for security-critical IoT and network applications.

摘要: 虽然机器学习极大地提高了网络入侵检测系统（NIDS），特别是在设备生成大量数据并且越来越容易受到网络威胁的物联网环境中，但这些模型仍然容易受到对抗性攻击。我们的研究揭示了现有对抗性攻击方法中的一个关键缺陷：经常违反特定领域的限制，例如物联网和网络流量固有的数字和类别限制。这导致高达80.3%的对抗示例无效，大大夸大了现实世界的漏洞。这些无效的示例虽然可以有效地欺骗模型，但并不代表实际物联网部署中的可行攻击。因此，依赖这些结果可能会误导国防资源分配，从而扩大支持物联网的NIDS模型对对抗操纵的感知敏感性。此外，我们还证明，与CNN和LSTM等复杂架构相比，多层感知器（MLP）等更简单的代理模型会生成更有效的对抗性示例。使用MLP作为替代品，我们分析了对抗严重性到物联网环境中常用的其他ML/DL模型的可转移性。这项工作强调了在评估和设计用于安全关键型物联网和网络应用的稳健ML/DL模型时考虑域约束和模型架构的重要性。



## **44. Security Metrics for Uncertain Interconnected Systems under Stealthy Data Injection Attacks**

隐形数据注入攻击下不确定互连系统的安全收件箱 eess.SY

6 pages, 5 figures, accepted to the 10th IFAC Conference on Networked  Systems, Hongkong 2025

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01233v1) [paper-pdf](http://arxiv.org/pdf/2505.01233v1)

**Authors**: Anh Tung Nguyen, Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: This paper quantifies the security of uncertain interconnected systems under stealthy data injection attacks. In particular, we consider a large-scale system composed of a certain subsystem interconnected with an uncertain subsystem, where only the input-output channels are accessible. An adversary is assumed to inject false data to maximize the performance loss of the certain subsystem while remaining undetected. By abstracting the uncertain subsystem as a class of admissible systems satisfying an $\mathcal{L}_2$ gain constraint, the worst-case performance loss is obtained as the solution to a convex semi-definite program depending only on the certain subsystem dynamics and such an $\mathcal{L}_2$ gain constraint. This solution is proved to serve as an upper bound for the actual worst-case performance loss when the model of the entire system is fully certain. The results are demonstrated through numerical simulations of the power transmission grid spanning Sweden and Northern Denmark.

摘要: 本文量化了隐形数据注入攻击下不确定互连系统的安全性。特别是，我们考虑一个由某个子系统与一个不确定子系统互连组成的大规模系统，其中只有输入输出通道是可访问的。假设对手注入错误数据，以最大限度地降低特定子系统的性能损失，同时保持未被检测到。通过将不确定子系统抽象为一类满足$\mathcal{L}_2$收益约束的可允许系统，最坏情况的性能损失被获得为仅取决于某些子系统动态和这样的$\mathcal{L}_2$收益约束的凸半定规划的解。当整个系统的模型完全确定时，该解决方案被证明可以作为实际最坏情况下性能损失的上界。结果表明，通过跨越瑞典和丹麦北部的输电网的数值模拟。



## **45. Bilateral Cognitive Security Games in Networked Control Systems under Stealthy Injection Attacks**

隐形注入攻击下网络控制系统中的双边认知安全博弈 eess.SY

8 pages, 3 figures, conference submission

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01232v1) [paper-pdf](http://arxiv.org/pdf/2505.01232v1)

**Authors**: Anh Tung Nguyen, Quanyan Zhu, André Teixeira

**Abstract**: This paper studies a strategic security problem in networked control systems under stealthy false data injection attacks. The security problem is modeled as a bilateral cognitive security game between a defender and an adversary, each possessing cognitive reasoning abilities. The adversary with an adversarial cognitive ability strategically attacks some interconnections of the system with the aim of disrupting the network performance while remaining stealthy to the defender. Meanwhile, the defender with a defense cognitive ability strategically monitors some nodes to impose the stealthiness constraint with the purpose of minimizing the worst-case disruption caused by the adversary. Within the proposed bilateral cognitive security framework, the preferred cognitive levels of the two strategic agents are formulated in terms of two newly proposed concepts, cognitive mismatch and cognitive resonance. Moreover, we propose a method to compute the policies for the defender and the adversary with arbitrary cognitive abilities. A sufficient condition is established under which an increase in cognitive levels does not alter the policies for the defender and the adversary, ensuring convergence. The obtained results are validated through numerical simulations.

摘要: 本文研究了网络控制系统在隐性虚假数据注入攻击下的战略安全问题。安全问题被建模为防御者和对手之间的双边认知安全游戏，双方都拥有认知推理能力。具有对抗性认知能力的对手会战略性地攻击系统的一些互连，目的是破坏网络性能，同时对防御者保持潜行。与此同时，具有防御认知能力的防御者会策略性地监控一些节点，以施加隐蔽性约束，以最大限度地减少对手造成的最坏情况破坏。在拟议的双边认知安全框架内，两个战略主体的首选认知水平是根据两个新提出的概念：认知不匹配和认知共振来制定的。此外，我们还提出了一种计算具有任意认知能力的防御者和对手策略的方法。建立了一个充分条件，在此条件下，认知水平的提高不会改变防御者和对手的政策，从而确保趋同。通过数值模拟验证了所得结果。



## **46. Secure Cluster-Based Hierarchical Federated Learning in Vehicular Networks**

车辆网络中安全的基于机器人的分层联邦学习 cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01186v1) [paper-pdf](http://arxiv.org/pdf/2505.01186v1)

**Authors**: M. Saeid HaghighiFard, Sinem Coleri

**Abstract**: Hierarchical Federated Learning (HFL) has recently emerged as a promising solution for intelligent decision-making in vehicular networks, helping to address challenges such as limited communication resources, high vehicle mobility, and data heterogeneity. However, HFL remains vulnerable to adversarial and unreliable vehicles, whose misleading updates can significantly compromise the integrity and convergence of the global model. To address these challenges, we propose a novel defense framework that integrates dynamic vehicle selection with robust anomaly detection within a cluster-based HFL architecture, specifically designed to counter Gaussian noise and gradient ascent attacks. The framework performs a comprehensive reliability assessment for each vehicle by evaluating historical accuracy, contribution frequency, and anomaly records. Anomaly detection combines Z-score and cosine similarity analyses on model updates to identify both statistical outliers and directional deviations in model updates. To further refine detection, an adaptive thresholding mechanism is incorporated into the cosine similarity metric, dynamically adjusting the threshold based on the historical accuracy of each vehicle to enforce stricter standards for consistently high-performing vehicles. In addition, a weighted gradient averaging mechanism is implemented, which assigns higher weights to gradient updates from more trustworthy vehicles. To defend against coordinated attacks, a cross-cluster consistency check is applied to identify collaborative attacks in which multiple compromised clusters coordinate misleading updates. Together, these mechanisms form a multi-level defense strategy to filter out malicious contributions effectively. Simulation results show that the proposed algorithm significantly reduces convergence time compared to benchmark methods across both 1-hop and 3-hop topologies.

摘要: 分层联邦学习（HFL）最近成为车载网络智能决策的一种有前途的解决方案，有助于解决通信资源有限、车辆移动性高和数据异构等挑战。然而，HFL仍然容易受到敌对和不可靠的车辆的影响，这些车辆的误导性更新可能会严重损害全局模型的完整性和收敛性。为了解决这些挑战，我们提出了一种新的防御框架，该框架将动态车辆选择与基于集群的HFL架构中的强大异常检测相结合，专门用于对抗高斯噪声和梯度上升攻击。该框架通过评估历史准确性、贡献频率和异常记录，对每辆车进行全面的可靠性评估。异常检测结合了对模型更新的Z分数和余弦相似性分析，以识别模型更新中的统计离群值和方向偏差。为了进一步完善检测，自适应阈值机制被纳入余弦相似性度量，根据每辆车的历史准确性动态调整阈值，以执行更严格的标准，始终高性能的车辆。此外，实现了加权梯度平均机制，该机制为来自更值得信赖的车辆的梯度更新分配更高的权重。为了抵御协同攻击，跨集群一致性检查应用于识别协作攻击，其中多个受损集群协调误导性更新。这些机制共同构成了多层防御策略，以有效过滤恶意贡献。仿真结果表明，与1跳和3跳布局的基准方法相比，所提出的算法显着缩短了收敛时间。



## **47. Explainable AI Based Diagnosis of Poisoning Attacks in Evolutionary Swarms**

基于可解释人工智能的进化群中毒攻击诊断 cs.AI

To appear in short form in Genetic and Evolutionary Computation  Conference (GECCO '25 Companion), 2025

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01181v1) [paper-pdf](http://arxiv.org/pdf/2505.01181v1)

**Authors**: Mehrdad Asadi, Roxana Rădulescu, Ann Nowé

**Abstract**: Swarming systems, such as for example multi-drone networks, excel at cooperative tasks like monitoring, surveillance, or disaster assistance in critical environments, where autonomous agents make decentralized decisions in order to fulfill team-level objectives in a robust and efficient manner. Unfortunately, team-level coordinated strategies in the wild are vulnerable to data poisoning attacks, resulting in either inaccurate coordination or adversarial behavior among the agents. To address this challenge, we contribute a framework that investigates the effects of such data poisoning attacks, using explainable AI methods. We model the interaction among agents using evolutionary intelligence, where an optimal coalition strategically emerges to perform coordinated tasks. Then, through a rigorous evaluation, the swarm model is systematically poisoned using data manipulation attacks. We showcase the applicability of explainable AI methods to quantify the effects of poisoning on the team strategy and extract footprint characterizations that enable diagnosing. Our findings indicate that when the model is poisoned above 10%, non-optimal strategies resulting in inefficient cooperation can be identified.

摘要: 集群系统（例如多无人机网络）擅长执行关键环境中的监控、监视或灾难援助等协作任务，其中自主代理做出分散决策，以便以稳健有效的方式实现团队级目标。不幸的是，野外团队级协调策略很容易受到数据中毒攻击，导致代理之间协调不准确或敌对行为。为了应对这一挑战，我们提供了一个框架，使用可解释的人工智能方法调查此类数据中毒攻击的影响。我们使用进化智能对代理之间的交互进行建模，其中战略性地出现最佳联盟来执行协调的任务。然后，通过严格的评估，使用数据操纵攻击对群模型进行系统性毒害。我们展示了可解释的人工智能方法的适用性，以量化中毒对团队策略的影响，并提取能够诊断的足迹特征。我们的研究结果表明，当模型中毒超过10%时，可以识别导致合作效率低下的非最优策略。



## **48. Harmonizing Intra-coherence and Inter-divergence in Ensemble Attacks for Adversarial Transferability**

协调对抗性可转让性的集合攻击中的内部一致性和内部分歧性 cs.LG

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01168v1) [paper-pdf](http://arxiv.org/pdf/2505.01168v1)

**Authors**: Zhaoyang Ma, Zhihao Wu, Wang Lu, Xin Gao, Jinghang Yue, Taolin Zhang, Lipo Wang, Youfang Lin, Jing Wang

**Abstract**: The development of model ensemble attacks has significantly improved the transferability of adversarial examples, but this progress also poses severe threats to the security of deep neural networks. Existing methods, however, face two critical challenges: insufficient capture of shared gradient directions across models and a lack of adaptive weight allocation mechanisms. To address these issues, we propose a novel method Harmonized Ensemble for Adversarial Transferability (HEAT), which introduces domain generalization into adversarial example generation for the first time. HEAT consists of two key modules: Consensus Gradient Direction Synthesizer, which uses Singular Value Decomposition to synthesize shared gradient directions; and Dual-Harmony Weight Orchestrator which dynamically balances intra-domain coherence, stabilizing gradients within individual models, and inter-domain diversity, enhancing transferability across models. Experimental results demonstrate that HEAT significantly outperforms existing methods across various datasets and settings, offering a new perspective and direction for adversarial attack research.

摘要: 模型集成攻击的发展显着提高了对抗性示例的可移植性，但这一进步也对深度神经网络的安全性构成了严重威胁。然而，现有的方法面临着两个关键挑战：模型之间的共享梯度方向的捕获不足以及缺乏自适应权重分配机制。为了解决这些问题，我们提出了一种新颖的方法对抗性可移植协调集合（HEAT），该方法首次将领域概括引入对抗性示例生成中。HEAT由两个关键模块组成：共识梯度方向合成器，它使用奇异值分解来合成共享的梯度方向;和双和谐权重合成器，它动态平衡域内一致性，稳定单个模型内的梯度和域间多样性，增强模型之间的可移植性。实验结果表明，HEAT在各种数据集和环境中的表现显着优于现有方法，为对抗性攻击研究提供了新的视角和方向。



## **49. Active Sybil Attack and Efficient Defense Strategy in IPFS DHT**

IPFS IDT中的主动Sybil攻击与高效防御策略 cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01139v1) [paper-pdf](http://arxiv.org/pdf/2505.01139v1)

**Authors**: V. H. M. Netto, T. Cholez, C. L. Ignat

**Abstract**: The InterPlanetary File System (IPFS) is a decentralized peer-to-peer (P2P) storage that relies on Kademlia, a Distributed Hash Table (DHT) structure commonly used in P2P systems for its proved scalability. However, DHTs are known to be vulnerable to Sybil attacks, in which a single entity controls multiple malicious nodes. Recent studies have shown that IPFS is affected by a passive content eclipse attack, leveraging Sybils, in which adversarial nodes hide received indexed information from other peers, making the content appear unavailable. Fortunately, the latest mitigation strategy coupling an attack detection based on statistical tests and a wider publication strategy upon detection was able to circumvent it.   In this work, we present a new active attack, with malicious nodes responding with semantically correct but intentionally false data, exploiting both an optimized placement of Sybils to stay below the detection threshold and an early trigger of the content discovery termination in Kubo, the main IPFS implementation. Our attack achieves to completely eclipse content on the latest Kubo release. When evaluated against the most recent known mitigation, it successfully denies access to the target content in approximately 80\% of lookup attempts.   To address this vulnerability, we propose a new mitigation called SR-DHT-Store, which enables efficient, Sybil-resistant content publication without relying on attack detection but instead on a systematic and precise use of region-based queries, defined by a dynamically computed XOR distance to the target ID. SR-DHT-Store can be combined with other defense mechanisms resulting in a defense strategy that completely mitigates both passive and active Sybil attacks at a lower overhead, while allowing an incremental deployment.

摘要: 星际文件系统（IPFS）是一种去中心化的点对点（P2P）存储，它依赖于Kademlia，这是一种分布式哈希表（IDT）结构，由于其已证明的可扩展性，通常用于P2P系统。然而，众所周知，IDT容易受到Sybil攻击，其中单个实体控制多个恶意节点。最近的研究表明，IPFS受到利用Sybils的被动内容日食攻击的影响，其中对抗性节点隐藏从其他对等点接收的索引信息，使内容看起来不可用。幸运的是，最新的缓解策略将基于统计测试的攻击检测和检测后更广泛的发布策略结合起来，能够规避它。   在这项工作中，我们提出了一种新的主动攻击，恶意节点以语义正确但故意错误的数据进行响应，利用Sybils的优化放置以保持低于检测阈值，以及Kubo中内容发现终止的早期触发，主要的IPFS实现。我们的攻击完全掩盖了最新Kubo版本的内容。当对照最近的已知缓解措施进行评估时，它在大约80%的查找尝试中成功拒绝对目标内容的访问。   为了解决这个漏洞，我们提出了一种名为SR-DHL-Store的新缓解措施，它可以在不依赖攻击检测的情况下实现高效、耐Sybil的内容发布，而是系统性且精确地使用基于区域的查询，由动态计算的到目标ID的异或距离定义。SR-IDT-Store可以与其他防御机制相结合，形成一种以较低的费用完全减轻被动和主动Sybil攻击的防御策略，同时允许增量部署。



## **50. Risk Analysis and Design Against Adversarial Actions**

针对对抗行为的风险分析和设计 cs.LG

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01130v1) [paper-pdf](http://arxiv.org/pdf/2505.01130v1)

**Authors**: Marco C. Campi, Algo Carè, Luis G. Crespo, Simone Garatti, Federico A. Ramponi

**Abstract**: Learning models capable of providing reliable predictions in the face of adversarial actions has become a central focus of the machine learning community in recent years. This challenge arises from observing that data encountered at deployment time often deviate from the conditions under which the model was trained. In this paper, we address deployment-time adversarial actions and propose a versatile, well-principled framework to evaluate the model's robustness against attacks of diverse types and intensities. While we initially focus on Support Vector Regression (SVR), the proposed approach extends naturally to the broad domain of learning via relaxed optimization techniques. Our results enable an assessment of the model vulnerability without requiring additional test data and operate in a distribution-free setup. These results not only provide a tool to enhance trust in the model's applicability but also aid in selecting among competing alternatives. Later in the paper, we show that our findings also offer useful insights for establishing new results within the out-of-distribution framework.

摘要: 近年来，能够在面对对抗行为时提供可靠预测的学习模型已成为机器学习界的中心焦点。这一挑战源于观察部署时遇到的数据经常偏离模型训练的条件。在本文中，我们讨论了部署时对抗动作，并提出了一个通用、原则良好的框架来评估该模型针对不同类型和强度攻击的稳健性。虽然我们最初专注于支持量回归（SVR），但所提出的方法通过宽松的优化技术自然扩展到广泛的学习领域。我们的结果可以评估模型漏洞，而不需要额外的测试数据，并在无分发设置中运行。这些结果不仅提供了增强对模型适用性信任的工具，而且还有助于在竞争的替代方案中进行选择。在本文的后面，我们表明我们的研究结果还为在非分布框架内建立新结果提供了有用的见解。



