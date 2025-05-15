#include <atomic>
#include <ros/ros.h>
#include <rviz/panel.h>
#include <ui_panel.h> // generated from "panel.ui" using CMAKE_AUTOMOC

#include "osx_skills/osx_skill_server.h"
#include <chrono>
#include <sensor_msgs/Image.h>
#include <ur_msgs/IOStates.h>

namespace osx_rviz
{

  // Camera state enum
  enum CameraState
  {
    UNAVAILABLE = 0, // Camera is not available
    UNACTIVE = 1,    // Camera is available but not active
    ACTIVE = 2       // Camera is available and active
  };

  class osxSetupPanel : public rviz::Panel
  {
    Q_OBJECT
  public:
    osxSetupPanel(QWidget *parent = nullptr);

    // slots
  private slots:
    // auto-connected slots: on_<object_name>_<signal_name>(<signal parameters>)

    // buttons
    void on_button_activate_ros_control_clicked();
    void on_button_robots_home_clicked();

    void on_button_home_a_bot_clicked();
    void on_button_open_gripper_a_clicked();
    void on_button_close_gripper_a_clicked();

    void on_button_home_b_bot_clicked();
    void on_button_open_gripper_b_clicked();
    void on_button_close_gripper_b_clicked();

    void on_button_open_base_fixation_clicked();
    void on_button_close_base_fixation_clicked();

    void on_button_a_bot_inside_camera_clicked();
    void on_button_a_bot_outside_camera_clicked();
    void on_button_b_bot_inside_camera_clicked();
    void on_button_b_bot_outside_camera_clicked();

    // methods
  private:
    void set_button_active(QPushButton *button, const bool active);

    void b_bot_inside_camera_callback(const sensor_msgs::Image &image);
    void b_bot_outside_camera_callback(const sensor_msgs::Image &image);
    void a_bot_inside_camera_callback(const sensor_msgs::Image &image);
    void a_bot_outside_camera_callback(const sensor_msgs::Image &image);
    void update_status(const ros::TimerEvent &);

    // Helper function to update camera button state
    void update_camera_button(QPushButton *button, CameraState &cam_state, CameraState &prev_cam_state);

    // Helper function to determine camera state
    CameraState determine_camera_state(const std::chrono::time_point<std::chrono::system_clock> &last_message_time,
                                       const std::string &camera_name, const std::string &active_camera);

    // The below 2 functions are taken from "planning_scene_display.h" *(MoveIt)

    // TODO: Add queue for background jobs (or just live with the buttons not
    // being perfect)
    // /** Queue this function call for execution within the background thread
    // All jobs are queued and processed in order by a single background thread.
    // */ void addBackgroundJob(const boost::function<void()>& job, const
    // std::string& name);

    /** Directly spawn a (detached) background thread for execution of this
       function call Should be used, when order of processing is not relevant /
       job can run in parallel. Must be used, when job will be blocking. Using
       addBackgroundJob() in this case will block other queued jobs as well */
    void spawnBackgroundJob(const boost::function<void()> &job);

    void robots_go_home();

    void open_gripper_wrapper(std::string robot);
    void close_gripper_wrapper(std::string robot);
    void go_named_pose_wrapper(std::string robot_name, std::string pose);
    void equip_unequip_wrapper(std::string robot_name, std::string tool,
                               std::string equip_or_unequip);

    // members
  private:
    Ui::osxSetupPanel ui;

    SkillServer ss_;
    ros::NodeHandle n;

    ros::Subscriber sub_io_states;
    ros::Subscriber sub_a_bot_inside_cam;
    ros::Subscriber sub_a_bot_outside_cam;
    ros::Subscriber sub_b_bot_inside_cam;
    ros::Subscriber sub_b_bot_outside_cam;

    std::chrono::time_point<std::chrono::system_clock>
        last_a_bot_inside_cam_message_time, last_a_bot_outside_cam_message_time,
        last_b_bot_inside_cam_message_time, last_b_bot_outside_cam_message_time;
    ros::Timer update_status_timer_;

    QString red_label_style = "QLabel { background-color : red; color : black; }";
    QString yellow_label_style =
        "QLabel { background-color : yellow; color : yellow; }";
    QString green_label_style =
        "QLabel { background-color : green; color : green; }";
    QString grey_label_style =
        "QLabel { background-color : grey; color : black; }";

    QString red_button_style =
        "QPushButton { background-color : red; color : black; }";
    QString yellow_button_style =
        "QPushButton { background-color : yellow; color : black; }";
    QString green_button_style =
        "QPushButton { background-color : green; color : green; }";
    QString grey_button_style =
        "QPushButton { background-color : grey; color : black; }";

    bool m3_suction_on, m4_suction_on;

    // Camera status tracking
    std::string active_camera;
    CameraState a_bot_inside_cam_state = UNAVAILABLE;
    CameraState a_bot_outside_cam_state = UNAVAILABLE;
    CameraState b_bot_inside_cam_state = UNAVAILABLE;
    CameraState b_bot_outside_cam_state = UNAVAILABLE;

    // Previous states for UI update optimization
    std::string prev_active_camera = "";
    CameraState prev_a_bot_inside_cam_state = UNAVAILABLE;
    CameraState prev_a_bot_outside_cam_state = UNAVAILABLE;
    CameraState prev_b_bot_inside_cam_state = UNAVAILABLE;
    CameraState prev_b_bot_outside_cam_state = UNAVAILABLE;

    // Track previous ROS control active state
    bool prev_ros_control_active = false;
  };

  osxSetupPanel::osxSetupPanel(QWidget *parent)
      : rviz::Panel(parent)
  {
    sub_a_bot_inside_cam =
        n.subscribe("/a_bot_inside_camera/aligned_depth_to_color/image_raw", 1,
                    &osxSetupPanel::a_bot_inside_camera_callback, this);
    sub_a_bot_outside_cam =
        n.subscribe("/a_bot_outside_camera/aligned_depth_to_color/image_raw", 1,
                    &osxSetupPanel::a_bot_outside_camera_callback, this);
    sub_b_bot_inside_cam =
        n.subscribe("/b_bot_inside_camera/aligned_depth_to_color/image_raw", 1,
                    &osxSetupPanel::b_bot_inside_camera_callback, this);
    sub_b_bot_outside_cam =
        n.subscribe("/b_bot_outside_camera/aligned_depth_to_color/image_raw", 1,
                    &osxSetupPanel::b_bot_outside_camera_callback, this);

    // load ui form and auto-connect slots
    ui.setupUi(this);

    // Default status is RED
    ui.label_robots_online->setStyleSheet(red_label_style);

    // disable buttons
    ui.button_a_bot_inside_camera->setEnabled(false);
    ui.button_a_bot_outside_camera->setEnabled(false);
    ui.button_b_bot_inside_camera->setEnabled(false);
    ui.button_b_bot_outside_camera->setEnabled(false);

    // Initialize camera message times
    last_a_bot_inside_cam_message_time = std::chrono::system_clock::now();
    last_a_bot_outside_cam_message_time = std::chrono::system_clock::now();
    last_b_bot_inside_cam_message_time = std::chrono::system_clock::now();
    last_b_bot_outside_cam_message_time = std::chrono::system_clock::now();

    // Check for active camera and update UI accordingly
    std::string current_active_camera;

    update_status_timer_ =
        n.createTimer(ros::Duration(0.1), &osxSetupPanel::update_status, this);
  }

  void osxSetupPanel::spawnBackgroundJob(const boost::function<void()> &job)
  {
    boost::thread t(job);
  }

  void osxSetupPanel::update_camera_button(QPushButton *button, CameraState &cam_state, CameraState &prev_cam_state)
  {
    // Only update UI if state has changed
    if (cam_state != prev_cam_state)
    {
      ROS_INFO_STREAM("Camera state changed from " << prev_cam_state << " to " << cam_state);
      switch (cam_state)
      {
      case UNAVAILABLE:
        button->setStyleSheet(grey_button_style);
        button->setEnabled(false);
        button->setText("-");
        break;
      case UNACTIVE:
        button->setEnabled(true);
        button->setStyleSheet(yellow_button_style);
        button->setText("-");
        break;
      case ACTIVE:
        button->setEnabled(true);
        button->setStyleSheet(green_button_style);
        button->setText("O");
        break;
      }
      prev_cam_state = cam_state;
    }
  }

  CameraState osxSetupPanel::determine_camera_state(
      const std::chrono::time_point<std::chrono::system_clock> &last_message_time,
      const std::string &camera_name, const std::string &active_camera)
  {
    // Check if camera is available (received a message within 500ms)
    auto now = std::chrono::system_clock::now();
    auto time_since_last_msg = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_message_time);
    bool is_available = (time_since_last_msg < std::chrono::milliseconds{500});

    if (!is_available)
    {
      return UNAVAILABLE;
    }
    else if (active_camera == camera_name)
    {
      return ACTIVE;
    }
    else
    {
      return UNACTIVE;
    }
  }

  void osxSetupPanel::update_status(const ros::TimerEvent &event)
  {
    // Check if ROS control active state has changed
    bool ros_control_active = ss_.a_bot_ros_control_active_ && ss_.b_bot_ros_control_active_;

    // Only update the label if the state has changed
    if (ros_control_active != prev_ros_control_active)
    {
      ROS_INFO_STREAM("ROS control state changed to: " << (ros_control_active ? "active" : "inactive"));
      if (ros_control_active)
        ui.label_robots_online->setStyleSheet(green_label_style);
      else
        ui.label_robots_online->setStyleSheet(red_label_style);

      // Update previous states
      prev_ros_control_active = ros_control_active;
    }

    // Get current active camera
    active_camera = ss_.getCurrentActiveCamera();

    // Update camera states
    a_bot_inside_cam_state = determine_camera_state(last_a_bot_inside_cam_message_time, "a_bot_inside_camera", active_camera);
    a_bot_outside_cam_state = determine_camera_state(last_a_bot_outside_cam_message_time, "a_bot_outside_camera", active_camera);
    b_bot_inside_cam_state = determine_camera_state(last_b_bot_inside_cam_message_time, "b_bot_inside_camera", active_camera);
    b_bot_outside_cam_state = determine_camera_state(last_b_bot_outside_cam_message_time, "b_bot_outside_camera", active_camera);

    // Update camera buttons
    update_camera_button(ui.button_a_bot_inside_camera, a_bot_inside_cam_state, prev_a_bot_inside_cam_state);
    update_camera_button(ui.button_a_bot_outside_camera, a_bot_outside_cam_state, prev_a_bot_outside_cam_state);
    update_camera_button(ui.button_b_bot_inside_camera, b_bot_inside_cam_state, prev_b_bot_inside_cam_state);
    update_camera_button(ui.button_b_bot_outside_camera, b_bot_outside_cam_state, prev_b_bot_outside_cam_state);

    // Update previous active camera
    prev_active_camera = active_camera;
  }

  void osxSetupPanel::a_bot_inside_camera_callback(
      const sensor_msgs::Image &image)
  {
    last_a_bot_inside_cam_message_time = std::chrono::system_clock::now();
  }

  void osxSetupPanel::a_bot_outside_camera_callback(
      const sensor_msgs::Image &image)
  {
    last_a_bot_outside_cam_message_time = std::chrono::system_clock::now();
  }

  void osxSetupPanel::b_bot_inside_camera_callback(
      const sensor_msgs::Image &image)
  {
    last_b_bot_inside_cam_message_time = std::chrono::system_clock::now();
  }

  void osxSetupPanel::b_bot_outside_camera_callback(
      const sensor_msgs::Image &image)
  {
    last_b_bot_outside_cam_message_time = std::chrono::system_clock::now();
  }

  // ======

  void osxSetupPanel::on_button_activate_ros_control_clicked()
  {
    ss_.activateROSControlOnUR("a_bot");
    ss_.activateROSControlOnUR("b_bot");
  }

  void osxSetupPanel::on_button_robots_home_clicked()
  {
    spawnBackgroundJob(boost::bind(&osxSetupPanel::robots_go_home,
                                   this)); // Required for background processing
  }
  void osxSetupPanel::robots_go_home()
  {
    ss_.goToNamedPose("home", "a_bot", 0.2, 0.2);
    ss_.goToNamedPose("home", "b_bot", 0.2, 0.2);
  }

  void osxSetupPanel::on_button_home_a_bot_clicked()
  {
    spawnBackgroundJob(boost::bind(&osxSetupPanel::go_named_pose_wrapper, this,
                                   "a_bot",
                                   "home")); // Required for background processing
  }
  void osxSetupPanel::on_button_open_gripper_a_clicked()
  {
    spawnBackgroundJob(
        boost::bind(&osxSetupPanel::open_gripper_wrapper, this, "a_bot"));
  }
  void osxSetupPanel::on_button_close_gripper_a_clicked()
  {
    spawnBackgroundJob(
        boost::bind(&osxSetupPanel::close_gripper_wrapper, this, "a_bot"));
  }

  void osxSetupPanel::on_button_home_b_bot_clicked()
  {
    spawnBackgroundJob(boost::bind(&osxSetupPanel::go_named_pose_wrapper, this,
                                   "b_bot",
                                   "home")); // Required for background processing
  }
  void osxSetupPanel::on_button_open_gripper_b_clicked()
  {
    spawnBackgroundJob(
        boost::bind(&osxSetupPanel::open_gripper_wrapper, this, "b_bot"));
  }
  void osxSetupPanel::on_button_close_gripper_b_clicked()
  {
    spawnBackgroundJob(
        boost::bind(&osxSetupPanel::close_gripper_wrapper, this, "b_bot"));
  }

  void osxSetupPanel::on_button_open_base_fixation_clicked()
  {
    ss_.setSuctionEjection("base_plate_lock", false);
    ss_.setSuctionEjection("base_plate_release", true);
  }
  void osxSetupPanel::on_button_close_base_fixation_clicked()
  {
    ss_.setSuctionEjection("base_plate_release", false);
    ss_.setSuctionEjection("base_plate_lock", true);
  }
  void osxSetupPanel::on_button_a_bot_inside_camera_clicked()
  {
    ss_.activateCamera("a_bot_inside_camera");
  }

  void osxSetupPanel::on_button_a_bot_outside_camera_clicked()
  {
    ss_.activateCamera("a_bot_outside_camera");
  }

  void osxSetupPanel::on_button_b_bot_inside_camera_clicked()
  {
    ss_.activateCamera("b_bot_inside_camera");
  }

  void osxSetupPanel::on_button_b_bot_outside_camera_clicked()
  {
    ss_.activateCamera("b_bot_outside_camera");
  }

  // Wrappers to allow background processing
  void osxSetupPanel::open_gripper_wrapper(std::string robot)
  {
    ss_.openGripper(robot);
  }
  void osxSetupPanel::close_gripper_wrapper(std::string robot)
  {
    ss_.closeGripper(robot);
  }
  void osxSetupPanel::go_named_pose_wrapper(std::string robot_name,
                                            std::string pose)
  {
    ss_.goToNamedPose(pose, robot_name, 0.2, 0.2);
  }

  // ---

  void osxSetupPanel::set_button_active(QPushButton *button, const bool active)
  {
    const QString style = active ? "QPushButton {color: red;}" : "QPushButton {}";
    button->setStyleSheet(style);
  }

} // namespace osx_rviz

#include "osx_rviz.moc"

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(osx_rviz::osxSetupPanel, rviz::Panel)
