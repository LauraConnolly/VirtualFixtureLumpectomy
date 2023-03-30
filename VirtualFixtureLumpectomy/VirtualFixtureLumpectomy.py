import logging
import os

import vtk
import qt

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import numpy as np


#
# VirtualFixtureLumpectomy
#

class VirtualFixtureLumpectomy(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "VirtualFixtureLumpectomy"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#VirtualFixtureLumpectomy">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # VirtualFixtureLumpectomy1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='VirtualFixtureLumpectomy',
        sampleName='VirtualFixtureLumpectomy1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'VirtualFixtureLumpectomy1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='VirtualFixtureLumpectomy1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='VirtualFixtureLumpectomy1'
    )

    # VirtualFixtureLumpectomy2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='VirtualFixtureLumpectomy',
        sampleName='VirtualFixtureLumpectomy2',
        thumbnailFileName=os.path.join(iconsPath, 'VirtualFixtureLumpectomy2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='VirtualFixtureLumpectomy2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='VirtualFixtureLumpectomy2'
    )


#
# VirtualFixtureLumpectomyWidget
#

class VirtualFixtureLumpectomyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/VirtualFixtureLumpectomy.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = VirtualFixtureLumpectomyLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Setup connections
        self.ui.setupFixtureButton.connect('clicked(bool)', self.onFixtureSetup)
        self.ui.releaseArmButton.connect('clicked(bool)', self.onReleaseArm)
        self.ui.dropFiducialButton.connect('clicked(bool)', self.onFiducialDropButton)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
        self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
        self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")


        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)

    def onFixtureSetup(self):

        self.logic.initializeVF()

    def onReleaseArm(self):

        self.logic.releaseArm()

    def onFiducialDropButton(self):

        self.logic.dropFiducial()


#
# VirtualFixtureLumpectomyLogic
#

class VirtualFixtureLumpectomyLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.breachWarningNode = None
        self.previousTimestamp = 0

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


    def initializeVF(self):

        # Get the ros2 node node
        self.ros2Node = slicer.mrmlScene.GetFirstNodeByName("ros2:node:slicer")
        self.ros2Node.CreateAndAddSubscriber("vtkMRMLROS2SubscriberPoseStampedNode", "/arm/measured_cp")
        self.ros2Node.CreateAndAddPublisher("vtkMRMLROS2PublisherPoseStampedNode", "/arm/servo_cp")
        self.ros2Node.CreateAndAddPublisher("vtkMRMLROS2PublisherWrenchStampedNode", "/arm/servo_cf")
        self.ciPub = self.ros2Node.CreateAndAddPublisher("vtkMRMLROS2PublisherCartesianImpedanceGainsNode", "/arm/servo_ci")

        self.sub = slicer.mrmlScene.GetFirstNodeByName("ros2:sub:/arm/measured_cp")
        self.pub = slicer.mrmlScene.GetFirstNodeByName("ros2:pub:/arm/servo_cp")
        self.wrenchpub = slicer.mrmlScene.GetFirstNodeByName("ros2:pub:/arm/servo_cf")
        #
        # Create the list of fiducials
        slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'tipTracking')
        self.tipTracking = slicer.mrmlScene.GetFirstNodeByName('tipTracking')

        slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'tipTracking-GT')
        self.tipTrackingGT = slicer.mrmlScene.GetFirstNodeByName('tipTracking-GT')
        self.tipTrackingGT.SetDisplayVisibility(False)

        # Get the breach warning node and check if there is a collision
        self.breachWarningNode = slicer.mrmlScene.GetFirstNodeByName("BreachWarning")
        self.forceTransform = slicer.vtkMRMLLinearTransformNode()
        self.forceTransform.SetName("ForceTransform")
        slicer.mrmlScene.AddNode(self.forceTransform)
        slicer.modules.createmodels.widgetRepresentation().OnCreateCoordinateClicked()  # dependency on SlicerIGT
        coordinateModel = slicer.mrmlScene.GetFirstNodeByName("CoordinateModel")
        coordinateModel.SetAndObserveTransformNodeID(self.forceTransform.GetID())

        timer = qt.QTimer()
        timer.setInterval(20) # 20 ms = 50 hz
        # timer.connect('timeout()', self.breachDetection(None, None))
        timer.start()

        self.increment = 20
        for increment in range(0, 10000):
          timeValue = self.increment*increment
          # timer.singleShot(timeValue, lambda: self.breachDetection(None, None))
          timer.singleShot(timeValue, lambda: self.breachDetectionCartesianImpedance(None, None))


    def breachDetection(self, caller, event):

        if (self.breachWarningNode.IsToolTipInsideModel()):
            pose = vtk.vtkMatrix4x4()
            self.sub.GetLastMessage(pose)
            self.pub.Publish(pose)
            self.dropFiducialGT()
        else:
            darr = vtk.vtkDoubleArray()
            self.wrenchpub.Publish(darr)

    def breachDetectionCartesianImpedance(self, caller, event):
        print('cartesian impede')
        if (self.breachWarningNode.IsToolTipInsideModel()):
            pose = vtk.vtkMatrix4x4()
            # self.sub.GetLastMessage(pose)
            lineToClosestPoint = self.breachWarningNode.GetLineToClosestPointNode()
            l1 = [0,0,0]
            l2 = [0,0,0]
            lineToClosestPoint.GetNthControlPointPosition(0, l1)
            lineToClosestPoint.GetNthControlPointPosition(1, l2)
            z = [(l2[0]-l1[0]), (l2[1] - l1[1]), (l2[2]-l1[2])]
            z_norm =  z / np.linalg.norm(z)
            x_norm = self.perpendicular_vector(z_norm)
            y_norm = np.cross(z_norm, x_norm)
            print("vector:")
            print(z_norm)
            print(x_norm)
            print(y_norm)
            # get x y z from the closest point on the surface
            pos_xyz = self.breachWarningNode.GetClosestPointOnModel()

            # Set up the matrix
            pose.SetElement(0, 0, x_norm[0])
            pose.SetElement(1, 0, x_norm[1])  # not sure of order of i, j - double check
            pose.SetElement(2, 0, x_norm[2])

            pose.SetElement(0, 1, y_norm[0])
            pose.SetElement(1, 1, y_norm[1])
            pose.SetElement(2, 1, y_norm[2])

            pose.SetElement(0, 2, z_norm[0])
            pose.SetElement(1, 2, z_norm[1])
            pose.SetElement(2, 2, z_norm[2])

            pose.SetElement(0, 3, pos_xyz[0])
            pose.SetElement(1, 3, pos_xyz[1])
            pose.SetElement(2, 3, pos_xyz[2])

            self.forceTransform.SetMatrixTransformToParent(pose)


            self.ciPub.Publish(pose)
            # how do I tell it where z is? Will get this from breach warning vector
            # need to deconstruct into the x y z and quaternion
            # self.pub.Publish(pose)
            # self.dropFiducialGT()
        else:
            darr = vtk.vtkDoubleArray()
            self.wrenchpub.Publish(darr)

    def releaseArm(self):

        self.ros2Node = slicer.mrmlScene.GetFirstNodeByName("ros2:node:slicer")
        self.ros2Node.CreateAndAddPublisher("vtkMRMLROS2PublisherWrenchStampedNode", "/arm/servo_cf")
        self.wrenchpub = slicer.mrmlScene.GetFirstNodeByName("ros2:pub:/arm/servo_cf")
        # Need a publisher for a wrench
        darr = vtk.vtkDoubleArray()
        self.wrenchpub.Publish(darr)

    def perpendicular_vector(self, v):
        if v[1] == 0 and v[2] == 0:
            if v[0] == 0:
                raise ValueError('zero vector')
            else:
                return np.cross(v, [0, 1, 0])
        return np.cross(v, [1, 0, 0])


    def dropFiducial(self):

        d = slicer.mrmlScene.GetNodeByID('vtkMRMLMarkupsLineNode1')
        pos = d.GetNthControlPointPosition(0)
        self.tipTracking.AddFiducial(pos[0], pos[1], pos[2])

    def dropFiducialGT(self):
        d = slicer.mrmlScene.GetNodeByID('vtkMRMLMarkupsLineNode1')
        pos = d.GetNthControlPointPosition(0)
        self.tipTrackingGT.AddFiducial(pos[0], pos[1], pos[2])
        self.tipTrackingGT.SetNthFiducialLabel(self.tipTrackingGT.GetNumberOfControlPoints(), '')
        # self.ros2Node.CreateAndAddSubscriber("vtkMRMLROS2SubscriberJoyNode", "/arm/button1")
        # self.buttonSub = slicer.mrmlScene.GetFirstNodeByName("ros2:sub:/arm/button1")
        #
        # buttonMess = self.buttonSub.GetLastMessageYAML()
        # lhs, rhs = buttonMess.split('nanosec:', 1)
        # timestamp, end = rhs.split('frame_id', 1)
        #
        # print(timestamp)
        #
        # if int(timestamp) > self.previousTimestamp:
        #     self.previousTimestamp = int(timestamp)
        #     print("fiducial dropped")

        # parse the button message and drop a fiducial at the tip of the transform
        # get the tip from the xyz in the tip to whatever transform





#
# VirtualFixtureLumpectomyTest
#

class VirtualFixtureLumpectomyTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_VirtualFixtureLumpectomy1()

    def test_VirtualFixtureLumpectomy1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('VirtualFixtureLumpectomy1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = VirtualFixtureLumpectomyLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
