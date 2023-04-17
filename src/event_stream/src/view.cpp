/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <metavision/sdk/core/utils/colors.h>

#include "event_utils.h"
#include "view.h"

View::View(Metavision::Camera &camera, Viewer::EventBuffer &event_buffer, const Parameters &parameters,
           const cv::Size &ui_size, const std::string &window_name) :
    camera_(camera),
    event_buffer_(event_buffer),
    parameters_(parameters),
    window_name_((window_name.empty() ? "Metavision Player" : window_name)),
    window_size_(getCameraSize(camera) + ui_size),
    frame_data(getCameraSize(camera), CV_8SC1),
    frame_show(getCameraSize(camera), CV_8UC3),
    setup_(false),
    show_help_(false),
    palette_(Metavision::ColorPalette::Light) {
    
        std::string camera_name("");
        nh_.getParam("camera_name", camera_name);
        const std::string topic_event_data  = "/prophesee/" + camera_name + "/event_data";
        const std::string topic_event_frame  = "/prophesee/" + camera_name + "/event_frame";

        image_transport::ImageTransport it(nh_);
        image_pub = nh_.advertise<rospy_tutorials::Floats>(topic_event_data, 1);
        show_pub = it.advertise(topic_event_frame,1);
    }

View::View(const cv::Size &ui_size, const View &view) :
    camera_(view.camera_),
    event_buffer_(view.event_buffer_),
    parameters_(view.parameters_),
    window_name_(view.window_name_),
    window_size_(getCameraSize(view.camera_) + ui_size),
    frame_data(view.frame_data.clone()),
    frame_show(view.frame_show.clone()),
    setup_(false),
    show_help_(view.show_help_),
    palette_(view.palette_) {}

View::~View() {
}

const std::string &View::windowName() const {
    return window_name_;
}

bool View::helpVisible() const {
    return show_help_;
}

void View::toggleHelpVisibility() {
    show_help_ = !show_help_;
}

void View::showHelp(cv::Mat &frame) {
    const auto &palette        = colorPalette(); 
    const cv::Scalar aux_color = getCVColor(palette, Metavision::ColorType::Auxiliary);
    frame -= cv::Scalar::all(255 * 0.8);

    std::vector<std::string> msgs = getHelpMessages();
    const int LetterWidth = 8, LineHeight = 20;
    cv::Size size;
    for (auto &msg : msgs) {
        if (msg.size() * LetterWidth > static_cast<size_t>(size.width))
            size.width = msg.size() * LetterWidth;
        size.height += LineHeight;
    }

    int y_offset = 0;
    for (auto &msg : msgs) {
        addText(frame, msg, cv::Point((frame.cols - size.width) / 2, (frame.rows - size.height) / 2 + y_offset),
                aux_color);
        y_offset += LineHeight;
    }
}

void View::addTextBox(const std::string &text, const cv::Scalar &color, const cv::Rect &rect, const cv::Point &pos) {
    cv::Mat box = frame_show(rect);
    box -= cv::Scalar::all(255 * 0.8);
    addText(box, text, pos, color);
}

int View::update() {
    if (!setup_) {
        setup_ = true;
        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        cv::imshow(window_name_, frame_show);
        show_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame_show).toImageMsg();

        int h = frame_data.size().height, w = frame_data.size().width, c = frame_data.channels();
        // ROS_INFO(std::string(c));
        // msg.channel = c, msg.height = h, msg.width = w;
        msg.data = std::vector<float>(frame_data.reshape(1, h * w * c));
        image_pub.publish(msg);
        show_pub.publish(show_msg);
        cv::resizeWindow(window_name_, window_size_.width, window_size_.height);
        // setup();
        cv::waitKey(1);
    }

    int key_pressed = cv::waitKey(1);
    // switch (key_pressed) {
    // case 'h':
    //     toggleHelpVisibility();
    //     break;
    // case 'c':
    //     cycleColorPalette();
    //     break;
    // default:
    //     break;
    // }

    // Draw image and compute statistic message
    auto &buffer             = eventBuffer();
    Metavision::timestamp ts = currentTimeUs();
    size_t num_events = makeSliceTwoImage(frame_show, frame_data, buffer.begin(), buffer.end(), ts, accumulationTimeUs(), framePeriodUs(),
                                       Viewer::FRAME_RATE, colorPalette());

    update(frame_show);
    if (helpVisible()) {
        showHelp(frame_show);
    } else {
        const auto &msg =
            makeSliceImageOverlayText(num_events, ts, accumulationTimeUs(), framePeriodUs(), Viewer::FRAME_RATE);
        addTextBox(msg, cv::Scalar::all(255), cv::Rect(0, 0, frame_data.cols, 20), cv::Point(5, 14));
    }
    if (!status_msg_.empty()) {
        const auto &palette = colorPalette();
        auto now            = std::chrono::high_resolution_clock::now();
        long delay          = std::chrono::duration_cast<std::chrono::milliseconds>(now - status_msg_time_).count();
        if (delay < status_msg_delay_ms_) {
            addTextBox(status_msg_, getCVColor(palette, Metavision::ColorType::Auxiliary),
                       cv::Rect(0, frame_data.rows - 20, frame_data.cols, 20), cv::Point(5, 14));
        } else {
            status_msg_ = std::string();
        }
    }
    cv::imshow(window_name_, frame_show);
    show_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame_show).toImageMsg();
    int h = frame_data.size().height, w = frame_data.size().width, c = frame_data.channels();
    // msg.channel = c, msg.height = h, msg.width = w;
    msg.data = std::vector<float>(frame_data.reshape(1, h * w * c));
    image_pub.publish(msg);
    show_pub.publish(show_msg);
    return key_pressed;
}

void View::cycleColorPalette() {
    palette_ = static_cast<Metavision::ColorPalette>((static_cast<int>(palette_) + 1) % 3);
}

Metavision::ColorPalette View::colorPalette() const {
    return palette_;
}

void View::setStatusMessage(const std::string &msg, int delay_msecs) {
    status_msg_          = msg;
    status_msg_time_     = std::chrono::high_resolution_clock::now();
    status_msg_delay_ms_ = delay_msecs;
}

Metavision::Camera &View::camera() {
    return camera_;
}

const Metavision::Camera &View::camera() const {
    return camera_;
}

const Viewer::EventBuffer &View::eventBuffer() const {
    return event_buffer_;
}

const Parameters &View::parameters() const {
    return parameters_;
}

int View::accumulationTimeUs() const {
    return compute_accumulation_time(accumulationRatio(), framePeriodUs());
}

int View::framePeriodUs() const {
    return compute_frame_period(fps());
}
