-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Nov 03, 2025 at 08:17 PM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.2.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `econsulta`
--

DELIMITER $$
--
-- Procedures
--
CREATE DEFINER=`root`@`localhost` PROCEDURE `update_machine_learning` (IN `screening_id` INT)   BEGIN
    DECLARE diagnosis_text TEXT;
    DECLARE disease_name VARCHAR(255);
    DECLARE schedule_date DATE;
    DECLARE year_val INT;
    DECLARE month_name VARCHAR(50);
    DECLARE month_num INT;
    DECLARE weather_val VARCHAR(10);

    
    SELECT d.diagnosis, s.schedule_time
    INTO diagnosis_text, schedule_date
    FROM doctor_notes d
    JOIN screening_data s ON s.id = d.screening_id
    WHERE d.screening_id = screening_id
    LIMIT 1;

    IF diagnosis_text IS NOT NULL AND schedule_date IS NOT NULL THEN
        SET disease_name = SUBSTRING_INDEX(diagnosis_text, '|', 1);
        SET year_val = YEAR(schedule_date);
        SET month_name = MONTHNAME(schedule_date);
        SET month_num = MONTH(schedule_date);

        
        IF month_num BETWEEN 6 AND 11 THEN
            SET weather_val = 'Rainy';
        ELSE
            SET weather_val = 'Sunny';
        END IF;

        
        INSERT INTO machine_learning (year, month, month_num, weather, disease, cases)
        VALUES (year_val, month_name, month_num, weather_val, disease_name, 1)
        ON DUPLICATE KEY UPDATE
            cases = machine_learning.cases + 1;
    END IF;
END$$

DELIMITER ;

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `id` int(11) NOT NULL,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`id`, `username`, `email`, `password`) VALUES
(1, 'sammy', 'sammy@gmail.com', 'Sammy@2025');

-- --------------------------------------------------------

--
-- Table structure for table `announcements`
--

CREATE TABLE `announcements` (
  `id` int(11) NOT NULL,
  `title` varchar(255) NOT NULL,
  `content` text NOT NULL,
  `image_path` varchar(255) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `chat_messages`
--

CREATE TABLE `chat_messages` (
  `id` int(11) NOT NULL,
  `sender_id` int(11) NOT NULL,
  `sender_type` enum('user','admin') NOT NULL,
  `receiver_id` int(11) NOT NULL,
  `receiver_type` enum('user','admin') NOT NULL,
  `message` text NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT current_timestamp(),
  `is_read` tinyint(1) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `chat_messages`
--

INSERT INTO `chat_messages` (`id`, `sender_id`, `sender_type`, `receiver_id`, `receiver_type`, `message`, `timestamp`, `is_read`) VALUES
(9, 5, 'user', 1, 'admin', 'Hello po, may concern lang po sa system', '2025-11-01 06:34:54', 0),
(10, 1, 'admin', 5, 'user', 'Problema mo?', '2025-11-01 06:35:38', 0);

-- --------------------------------------------------------

--
-- Table structure for table `doctors`
--

CREATE TABLE `doctors` (
  `id` int(11) NOT NULL,
  `first_name` varchar(50) NOT NULL,
  `last_name` varchar(50) NOT NULL,
  `specialization` varchar(100) DEFAULT NULL,
  `email` varchar(100) NOT NULL,
  `password` varchar(255) NOT NULL,
  `status` enum('active','inactive') DEFAULT 'active'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `doctors`
--

INSERT INTO `doctors` (`id`, `first_name`, `last_name`, `specialization`, `email`, `password`, `status`) VALUES
(1, 'John Michael', 'Celo', 'Physician', 'john@gmail.com', '$2y$10$80Rp/2A88RCwTfBJPJ8pIuphLqP7tc3beCMeUfVitLNhKefTdHbjy', 'active'),
(2, 'Nicole', 'Mallari', 'Dentist', 'nicole@gmail.com', '$2y$10$SO5puHd2PhWKHmlT0lTMPO7SRZmuJwTsGzDYu9l8xZ./eFFee3OPq', 'active'),
(3, 'Samantha', 'Montefalco', 'Dentist', 'Samantha@gmail.com', '$2y$10$.dVnWxbIiC3tRnXaGFpj5eVrFAfGtcO3Wuiv6K2N25aYyVQPSiQny', 'active');

-- --------------------------------------------------------

--
-- Table structure for table `doctor_notes`
--

CREATE TABLE `doctor_notes` (
  `id` int(11) NOT NULL,
  `screening_id` int(11) NOT NULL,
  `schedule_time` datetime DEFAULT NULL,
  `doctor_id` int(11) NOT NULL,
  `findings` text DEFAULT NULL,
  `diagnosis` text DEFAULT NULL,
  `prescription` text DEFAULT NULL,
  `recommendations` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `doctor_notes`
--

INSERT INTO `doctor_notes` (`id`, `screening_id`, `schedule_time`, `doctor_id`, `findings`, `diagnosis`, `prescription`, `recommendations`, `created_at`) VALUES
(1, 21, '2025-09-24 14:30:00', 1, 'Patient presents with chills and recurrent vomiting (non-bloody, non-bilious), onset ~8–12 hrs prior to consult. Reports low-grade fever and nausea; denies severe abdominal pain, diarrhea, hematemesis, melena, dysuria, flank pain, headache, or rash. Able to tolerate small sips of fluid. Urine output acceptable.\r\n\r\nVitals: T 38.2°C, HR 96, BP 110/70, RR 18, SpO2 98% RA.\r\nExam: Alert, no acute distress. Mucous membranes slightly dry; cap refill <2s. Abdomen soft, non-distended, non-tender; no guarding/rebound; no CVA tenderness. Neuro grossly intact; skin warm, no petechiae.\r\nImpression: Mild dehydration; otherwise stable.', 'acute_gastroenteritis|K52.9', 'uploads/prescriptions/rx-21-1758697060.jpg', 'Rest; avoid strenuous activity.', '2025-09-24 06:57:40'),
(2, 23, '2025-09-24 16:30:00', 1, 'red, swollen tonsils, swollen lymph nodes', 'asthma_exacerbation|J45', 'uploads/prescriptions/rx-23-1758707176.jpg', 'Most sore throats are viral and often accompanied by cough, runny nose, and hoarseness.', '2025-09-24 09:46:16'),
(3, 24, '2025-09-26 15:30:00', 1, 'Patient reports sudden onset chills beginning 1–2 days prior to consult. Associated subjective fever, body aches, mild headache. No cough, no sore throat, no diarrhea, no dysuria, no rash, no recent travel or known sick contacts reported.\r\nVitals (today): Temp 38.4 °C (febrile), BP 110/70 mmHg, HR 96 bpm, RR 18/min.\r\nGeneral: Ill-appearing but non-toxic, oriented, good skin turgor, no respiratory distress.\r\nHEENT: Non-erythematous oropharynx, no tonsillar exudates.\r\nChest/Lungs: Clear to auscultation bilaterally.\r\nCV: Regular rate/rhythm, no murmurs.\r\nAbdomen: Soft, non-tender, no guarding.\r\nSkin: No petechiae, no rashes.\r\nAssessment: Febrile illness with chills, no localizing signs on exam.', 'fever_unspecified|R50.9', 'uploads/prescriptions/rx-24-1758870714.jpg', 'Supportive care: Oral rehydration (water/ORS), light meals, adequate rest.\r\n\r\nAntipyretic: Paracetamol 10–15 mg/kg/dose (max 1 g per dose, 4 g/day for adults) every 6–8 hours as needed for fever/chills.\r\n\r\nNon-pharmacologic: Tepid sponging for high fever, avoid overdressing.', '2025-09-26 07:11:54'),
(4, 25, '2025-09-30 14:30:00', 1, 'Foul-smelling vomit: suggests a distal intestinal obstruction. \r\nProjectile vomiting: can indicate increased intracranial pressure.', 'fever_unspecified|R50.9', 'uploads/prescriptions/rx-25-1758951220.jpg', 'Prevent Dehydration: Sip small amounts of clear liquids, oral rehydration solutions, or electrolyte drinks.\r\nDiet: Once vomiting subsides, gradually reintroduce a bland diet (BRAT diet: bananas, rice, applesauce, toast).', '2025-09-27 05:33:40'),
(6, 40, '2025-10-20 09:00:00', 1, 'Dengue Fever', 'dengue_suspected|A97.9', NULL, 'Encourage fluid intake, monitor platelets', '2025-10-17 12:22:29'),
(7, 41, '2025-10-20 13:00:00', 1, 'Bronchial Asthma', 'Bronchial Asthma', NULL, 'Prescribed salbutamol inhaler', '2025-10-17 12:29:16'),
(8, 42, '2025-10-20 10:00:00', 1, 'Dengue Fever', 'dengue_suspected|A97.9', NULL, 'Advise hydration and platelet monitoring', '2025-10-17 12:33:57'),
(9, 43, '2024-01-17 10:00:00', 1, 'Headache, dizziness', 'Essential Hypertension', NULL, 'Lifestyle change, start meds', '2025-10-17 12:50:57'),
(10, 4, '2020-01-20 13:30:00', 1, 'Chills, Cough', 'Cough and Colds', NULL, 'Rest', '2025-10-18 03:29:37'),
(11, 21, '2025-09-24 14:30:00', 1, NULL, 'acute_gastroenteritis|K52.9', NULL, NULL, '2025-10-18 05:02:57'),
(12, 44, '2025-10-13 13:30:00', 1, 'High Fever', 'dengue_suspected|A97.9', NULL, 'Advise hydration and platelet monitoring', '2025-10-18 07:13:12'),
(13, 45, '2025-10-15 10:00:00', 1, 'High Fever', 'dengue_suspected|A97.9', NULL, 'Encourage fluid intake, monitor platelets', '2025-10-18 07:20:02'),
(14, 46, '2025-10-16 13:00:00', 1, 'High Fever', 'dengue_suspected|A97.9', NULL, 'Advise hydration and platelet monitoring', '2025-10-18 07:25:54'),
(15, 47, '2025-10-20 15:00:00', 1, 'Chills', 'Flu', NULL, 'Rest', '2025-10-18 07:28:08'),
(16, 48, '2025-10-21 09:30:00', 1, 'Fever', 'Flu', NULL, 'Rest', '2025-10-18 07:30:02');

--
-- Triggers `doctor_notes`
--
DELIMITER $$
CREATE TRIGGER `after_doctor_notes_insert` AFTER INSERT ON `doctor_notes` FOR EACH ROW BEGIN
    DECLARE diagnosis_text TEXT;
    DECLARE disease_name VARCHAR(255);
    DECLARE schedule_date DATE;
    DECLARE year_val INT;
    DECLARE month_name VARCHAR(50);
    DECLARE month_num INT;
    DECLARE weather_val VARCHAR(10);

    SELECT schedule_time INTO schedule_date
    FROM screening_data
    WHERE id = NEW.screening_id
    LIMIT 1;

    IF NEW.diagnosis IS NOT NULL AND schedule_date IS NOT NULL THEN
        SET disease_name = SUBSTRING_INDEX(NEW.diagnosis, '|', 1);
        SET year_val = YEAR(schedule_date);
        SET month_name = MONTHNAME(schedule_date);
        SET month_num = MONTH(schedule_date);

        IF month_num BETWEEN 6 AND 11 THEN
            SET weather_val = 'Rainy';
        ELSE
            SET weather_val = 'Sunny';
        END IF;

        INSERT INTO machine_learning (year, month, month_num, weather, disease, cases)
        VALUES (year_val, month_name, month_num, weather_val, disease_name, 1)
        ON DUPLICATE KEY UPDATE
            cases = machine_learning.cases + 1;
    END IF;
END
$$
DELIMITER ;
DELIMITER $$
CREATE TRIGGER `after_doctor_notes_update` AFTER UPDATE ON `doctor_notes` FOR EACH ROW BEGIN
    DECLARE old_disease VARCHAR(255);
    DECLARE new_disease VARCHAR(255);
    DECLARE year_val INT;
    DECLARE month_name VARCHAR(50);
    DECLARE month_num INT;
    DECLARE weather_val VARCHAR(10);
    DECLARE schedule_date DATE;

    SELECT schedule_time INTO schedule_date
    FROM screening_data
    WHERE id = NEW.screening_id
    LIMIT 1;

    IF schedule_date IS NOT NULL THEN
        SET year_val = YEAR(schedule_date);
        SET month_name = MONTHNAME(schedule_date);
        SET month_num = MONTH(schedule_date);

        IF month_num BETWEEN 6 AND 11 THEN
            SET weather_val = 'Rainy';
        ELSE
            SET weather_val = 'Sunny';
        END IF;

        SET old_disease = SUBSTRING_INDEX(OLD.diagnosis, '|', 1);
        SET new_disease = SUBSTRING_INDEX(NEW.diagnosis, '|', 1);

        IF old_disease <> new_disease THEN
            
            UPDATE machine_learning
            SET cases = GREATEST(cases - 1, 0)
            WHERE year = year_val AND month_num = month_num AND disease = old_disease;

            
            INSERT INTO machine_learning (year, month, month_num, weather, disease, cases)
            VALUES (year_val, month_name, month_num, weather_val, new_disease, 1)
            ON DUPLICATE KEY UPDATE
                cases = machine_learning.cases + 1;
        END IF;
    END IF;
END
$$
DELIMITER ;
DELIMITER $$
CREATE TRIGGER `trg_doctor_notes_copy_schedule` BEFORE INSERT ON `doctor_notes` FOR EACH ROW BEGIN
  DECLARE sched_time DATETIME;

  SELECT schedule_time
  INTO sched_time
  FROM screening_data
  WHERE id = NEW.screening_id
  LIMIT 1;

  SET NEW.schedule_time = sched_time;
END
$$
DELIMITER ;

-- --------------------------------------------------------

--
-- Table structure for table `doctor_schedule`
--

CREATE TABLE `doctor_schedule` (
  `id` int(11) NOT NULL,
  `doctor_id` int(11) NOT NULL,
  `day_of_week` enum('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday') NOT NULL,
  `start_time` time NOT NULL,
  `end_time` time NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `doctor_schedule`
--

INSERT INTO `doctor_schedule` (`id`, `doctor_id`, `day_of_week`, `start_time`, `end_time`) VALUES
(3, 1, 'Monday', '09:00:00', '12:00:00'),
(4, 1, 'Monday', '13:00:00', '17:00:00'),
(5, 1, 'Tuesday', '09:00:00', '12:00:00'),
(6, 1, 'Tuesday', '13:00:00', '17:00:00'),
(7, 1, 'Wednesday', '09:00:00', '12:00:00'),
(8, 1, 'Wednesday', '13:00:00', '17:00:00'),
(9, 1, 'Thursday', '09:00:00', '12:00:00'),
(10, 1, 'Thursday', '13:00:00', '17:00:00'),
(11, 1, 'Friday', '09:00:00', '12:00:00'),
(12, 1, 'Friday', '13:00:00', '17:00:00');

-- --------------------------------------------------------

--
-- Table structure for table `family_planning`
--

CREATE TABLE `family_planning` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `full_name` varchar(255) DEFAULT NULL,
  `age` int(11) DEFAULT NULL,
  `gender` varchar(10) DEFAULT NULL,
  `marital_status` varchar(50) DEFAULT NULL,
  `contact_number` varchar(20) DEFAULT NULL,
  `using_method` varchar(10) DEFAULT NULL,
  `current_method` varchar(255) DEFAULT NULL,
  `preferred_method` varchar(50) DEFAULT NULL,
  `health_concerns` text DEFAULT NULL,
  `method_reason` text DEFAULT NULL,
  `consent_given` enum('Yes','No') DEFAULT NULL,
  `pregnant` enum('Yes','No') DEFAULT NULL,
  `last_menstrual_period` date DEFAULT NULL,
  `number_of_children` int(11) DEFAULT NULL,
  `preferred_schedule` datetime DEFAULT NULL,
  `created_at` datetime DEFAULT current_timestamp(),
  `status` enum('pending','approved','rejected','expired') DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `family_planning`
--

INSERT INTO `family_planning` (`id`, `user_id`, `full_name`, `age`, `gender`, `marital_status`, `contact_number`, `using_method`, `current_method`, `preferred_method`, `health_concerns`, `method_reason`, `consent_given`, `pregnant`, `last_menstrual_period`, `number_of_children`, `preferred_schedule`, `created_at`, `status`) VALUES
(2, 3, 'Charlotte Bella LUNA. Nova', 29, 'Female', 'Married', '09557321458', 'No', '', 'Injection', '', '', 'Yes', 'No', '2025-06-16', 2, '2025-09-21 08:00:00', '2025-07-16 00:20:40', 'expired'),
(3, 1, 'Sean Andrew OWEN. Jackson', 20, 'Male', 'Married', '09557436598', 'No', '', 'Condom', '', 'Para saafe', 'Yes', 'No', '0000-00-00', 10, '2025-08-30 10:00:00', '2025-07-16 13:18:35', 'rejected'),
(4, 4, 'Sofia Isabella  FORB. Connor', 30, 'Male', 'Married', '091757783421', 'No', '', 'Pills', '', '', 'Yes', 'No', '2025-06-19', 1, '2025-09-06 08:00:00', '2025-07-19 18:52:54', 'expired'),
(5, 3, 'Charlotte Bella LUNA. Nova', 29, 'Female', 'Married', '09557321458', 'No', NULL, 'Injection', NULL, NULL, 'Yes', 'No', '2025-07-08', 2, '2025-07-29 10:00:00', '2025-07-28 22:01:46', 'expired'),
(6, 3, 'Charlotte Bella LUNA. Nova', 29, 'Female', NULL, '09557321458', 'Yes', 'Injection', 'Injection', '', '', 'Yes', '', '0000-00-00', 2, NULL, '2025-07-28 23:11:05', 'pending'),
(7, 3, 'Charlotte Bella LUNA. Nova', 29, 'Female', NULL, '09557321458', '', '', 'Pills', '', '', 'Yes', 'No', '0000-00-00', 2, NULL, '2025-07-30 15:39:38', 'pending'),
(8, 7, 'Lenaj ROLF. Rolf', 30, 'Female', NULL, '09544662627', 'No', '', 'Injection', '', '', 'Yes', 'Yes', '0000-00-00', 3, '2025-08-06 15:00:00', '2025-08-02 00:09:13', 'expired'),
(10, 13, 'Angel A.. Cruz', 30, 'Female', NULL, '09670984729', 'Yes', 'Pills', 'Injection', '', 'Reduced cancer risk', 'Yes', 'No', '0000-00-00', 5, '2025-11-10 10:30:00', '2025-11-03 23:09:30', 'approved'),
(11, 14, 'Cristina F.. Albindo', 30, 'Female', NULL, '09205057976', 'No', '', 'Pills', '', 'Easy method', 'Yes', 'No', '0000-00-00', 5, '2025-11-04 11:00:00', '2025-11-04 01:01:33', 'approved');

-- --------------------------------------------------------

--
-- Table structure for table `feedback`
--

CREATE TABLE `feedback` (
  `id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `subject` varchar(255) NOT NULL,
  `message` text NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `feedback`
--

INSERT INTO `feedback` (`id`, `name`, `email`, `subject`, `message`, `created_at`) VALUES
(1, 'Angel Cruz', 'angel@gmail.com', 'Consultation', 'Satisfied to your service!', '2025-11-03 15:11:48');

-- --------------------------------------------------------

--
-- Table structure for table `infants_consultation`
--

CREATE TABLE `infants_consultation` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `infant_name` varchar(100) NOT NULL,
  `infant_birthday` date DEFAULT NULL,
  `infant_age` int(11) NOT NULL,
  `guardian_name` varchar(100) NOT NULL,
  `preferred_schedule` datetime DEFAULT NULL,
  `consult_type` text DEFAULT NULL,
  `vax_status` text DEFAULT NULL,
  `others_text` varchar(255) DEFAULT NULL,
  `follow_up` enum('Yes','No') NOT NULL,
  `guardian_first_name` varchar(50) DEFAULT NULL,
  `guardian_middle_initial` varchar(10) DEFAULT NULL,
  `guardian_last_name` varchar(50) DEFAULT NULL,
  `guardian_suffix` varchar(10) DEFAULT NULL,
  `guardian_relationship` varchar(30) DEFAULT NULL,
  `guardian_contact_no` varchar(20) DEFAULT NULL,
  `guardian_email` varchar(100) DEFAULT NULL,
  `guardian_home_address` text DEFAULT NULL,
  `first_time` enum('Yes','No') DEFAULT NULL,
  `full_term` enum('Yes','No') DEFAULT NULL,
  `weeks_pregnancy` varchar(50) DEFAULT NULL,
  `has_allergies` enum('Yes','No') DEFAULT NULL,
  `allergy_details` varchar(255) DEFAULT NULL,
  `symptoms` text DEFAULT NULL,
  `other_symptoms` varchar(255) DEFAULT NULL,
  `hospitalization_details` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `status` enum('pending','approved','rejected','expired') NOT NULL DEFAULT 'pending',
  `doctor_id` int(11) DEFAULT NULL,
  `consultation_status` enum('pending','in_progress','completed') DEFAULT 'pending',
  `findings` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `infants_consultation`
--

INSERT INTO `infants_consultation` (`id`, `user_id`, `infant_name`, `infant_birthday`, `infant_age`, `guardian_name`, `preferred_schedule`, `consult_type`, `vax_status`, `others_text`, `follow_up`, `guardian_first_name`, `guardian_middle_initial`, `guardian_last_name`, `guardian_suffix`, `guardian_relationship`, `guardian_contact_no`, `guardian_email`, `guardian_home_address`, `first_time`, `full_term`, `weeks_pregnancy`, `has_allergies`, `allergy_details`, `symptoms`, `other_symptoms`, `hospitalization_details`, `created_at`, `status`, `doctor_id`, `consultation_status`, `findings`) VALUES
(1, 3, 'Mary Joy Levi Watsons', NULL, 3, 'Charlotte Bella Luna', '2025-07-18 08:00:00', 'Vaccination', 'Not Yet Vaccinated', '', 'Yes', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'Yes', 'Yes', '', 'No', '', 'None', '', '', '2025-07-15 05:19:29', 'expired', NULL, 'pending', NULL),
(2, 3, 'Lauefey Nova', NULL, 3, 'Charlotte Bella Luna', '2025-08-05 08:00:00', 'Vaccination', 'Not Yet Vaccinated', '', 'Yes', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'Yes', 'Yes', '', 'No', '', 'None', '', '', '2025-07-28 08:13:21', 'expired', NULL, 'pending', NULL),
(3, 3, 'Lauefey Nova', NULL, 3, 'Charlotte Bella Luna', NULL, 'Sick Visit', 'Partially Vaccinated', '', 'No', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'No', 'Yes', '', '', '', 'Fever', '', '', '2025-07-28 08:34:22', 'rejected', NULL, 'pending', NULL),
(4, 7, 'Kai Isaac Rolf', NULL, 3, 'Lenaj Rolf', '2025-08-04 10:00:00', 'Vaccination', 'Not Yet Vaccinated', '', 'Yes', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'Yes', 'Yes', '', 'Yes', 'Skin rashes', 'None', '', '', '2025-08-01 15:58:48', 'expired', NULL, 'pending', NULL),
(6, 13, 'Via', '2025-06-03', 5, 'Angel Cruz', '2025-11-06 10:00:00', 'Vaccination', 'Not Yet Vaccinated', '', 'No', 'Angel', 'A.', 'Cruz', 'N/A', 'Mother', '09670984729', 'angel@gmail.com', 'Pasig City', 'No', 'Yes', '', 'No', '', 'Fever', '', '', '2025-11-03 15:06:34', 'approved', 1, 'pending', NULL),
(7, 14, 'Zaijan', '2025-01-06', 9, 'Cristina Albindo', '2025-11-04 09:30:00', 'Check-Up', 'Fully Vaccinated', '', 'No', 'Cristina', 'F.', 'Albindo', 'N/A', 'Mother', '09205057976', 'cristina@gmail.com', 'Pasig City', 'Yes', 'Yes', '', 'No', '', 'Cough, Rash', '', '', '2025-11-03 16:59:36', 'approved', 1, 'pending', NULL);

-- --------------------------------------------------------

--
-- Table structure for table `login_history`
--

CREATE TABLE `login_history` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `user_type` enum('admin','doctor') NOT NULL,
  `ip_address` varchar(100) DEFAULT NULL,
  `user_agent` text DEFAULT NULL,
  `login_time` datetime DEFAULT current_timestamp(),
  `logout_time` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `machine_learning`
--

CREATE TABLE `machine_learning` (
  `id` int(11) NOT NULL,
  `year` int(11) NOT NULL,
  `month` varchar(50) NOT NULL,
  `month_num` int(11) NOT NULL,
  `weather` enum('Sunny','Rainy') NOT NULL,
  `disease` varchar(100) NOT NULL,
  `cases` int(11) NOT NULL DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `machine_learning`
--

INSERT INTO `machine_learning` (`id`, `year`, `month`, `month_num`, `weather`, `disease`, `cases`) VALUES
(1, 2020, 'January', 1, 'Sunny', 'Cough and Colds', 21),
(2, 2020, 'January', 1, 'Sunny', 'Hypertension', 58),
(3, 2020, 'January', 1, 'Sunny', 'Fever', 96),
(4, 2020, 'January', 1, 'Sunny', 'Asthma', 61),
(5, 2020, 'January', 1, 'Sunny', 'Flu', 25),
(6, 2020, 'January', 1, 'Sunny', 'Skin Infection', 4),
(7, 2020, 'January', 1, 'Sunny', 'Diarrhea', 13),
(8, 2020, 'February', 2, 'Sunny', 'Flu', 43),
(9, 2020, 'February', 2, 'Sunny', 'Asthma', 41),
(10, 2020, 'February', 2, 'Sunny', 'Dengue', 71),
(11, 2020, 'February', 2, 'Sunny', 'Hypertension', 10),
(12, 2020, 'February', 2, 'Sunny', 'Toothache', 18),
(13, 2020, 'February', 2, 'Sunny', 'Fever', 14),
(14, 2020, 'February', 2, 'Sunny', 'Headache', 45),
(15, 2020, 'February', 2, 'Sunny', 'Cough and Colds', 13),
(16, 2020, 'March', 3, 'Sunny', 'Asthma', 21),
(17, 2020, 'March', 3, 'Sunny', 'Flu', 56),
(18, 2020, 'March', 3, 'Sunny', 'Hypertension', 32),
(19, 2020, 'March', 3, 'Sunny', 'Cough and Colds', 15),
(20, 2020, 'March', 3, 'Sunny', 'Fever', 73),
(21, 2020, 'March', 3, 'Sunny', 'Diarrhea', 40),
(22, 2020, 'March', 3, 'Sunny', 'Body Pain', 14),
(23, 2020, 'March', 3, 'Sunny', 'Dengue', 16),
(24, 2020, 'April', 4, 'Sunny', 'Skin Infection', 66),
(25, 2020, 'April', 4, 'Sunny', 'Tuberculosis (TB)', 32),
(26, 2020, 'April', 4, 'Sunny', 'Asthma', 126),
(27, 2020, 'April', 4, 'Sunny', 'Body Pain', 8),
(28, 2020, 'April', 4, 'Sunny', 'Cough and Colds', 28),
(29, 2020, 'May', 5, 'Sunny', 'Dengue', 21),
(30, 2020, 'May', 5, 'Sunny', 'Tuberculosis (TB)', 16),
(31, 2020, 'May', 5, 'Sunny', 'Toothache', 72),
(32, 2020, 'May', 5, 'Sunny', 'Skin Infection', 97),
(33, 2020, 'May', 5, 'Sunny', 'Hypertension', 8),
(34, 2020, 'May', 5, 'Sunny', 'Fever', 24),
(35, 2020, 'May', 5, 'Sunny', 'Cough and Colds', 14),
(36, 2020, 'May', 5, 'Sunny', 'Flu', 17),
(37, 2020, 'June', 6, 'Rainy', 'Asthma', 4),
(38, 2020, 'June', 6, 'Rainy', 'Flu', 103),
(39, 2020, 'June', 6, 'Rainy', 'Skin Infection', 32),
(40, 2020, 'June', 6, 'Rainy', 'Toothache', 31),
(41, 2020, 'June', 6, 'Rainy', 'Hypertension', 8),
(42, 2020, 'June', 6, 'Rainy', 'Fever', 14),
(43, 2020, 'June', 6, 'Rainy', 'Cough and Colds', 40),
(44, 2020, 'June', 6, 'Rainy', 'Tuberculosis (TB)', 32),
(45, 2020, 'July', 7, 'Rainy', 'Flu', 106),
(46, 2020, 'July', 7, 'Rainy', 'Diarrhea', 17),
(47, 2020, 'July', 7, 'Rainy', 'Toothache', 50),
(48, 2020, 'July', 7, 'Rainy', 'Cough and Colds', 52),
(49, 2020, 'July', 7, 'Rainy', 'Skin Infection', 23),
(50, 2020, 'July', 7, 'Rainy', 'Asthma', 14),
(51, 2020, 'July', 7, 'Rainy', 'Dengue', 22),
(52, 2020, 'August', 8, 'Rainy', 'Skin Infection', 4),
(53, 2020, 'August', 8, 'Rainy', 'Body Pain', 17),
(54, 2020, 'August', 8, 'Rainy', 'Hypertension', 92),
(55, 2020, 'August', 8, 'Rainy', 'Dengue', 51),
(56, 2020, 'August', 8, 'Rainy', 'Flu', 54),
(57, 2020, 'August', 8, 'Rainy', 'Diarrhea', 25),
(58, 2020, 'August', 8, 'Rainy', 'Asthma', 22),
(59, 2020, 'August', 8, 'Rainy', 'Toothache', 16),
(60, 2020, 'September', 9, 'Rainy', 'Cough and Colds', 130),
(61, 2020, 'September', 9, 'Rainy', 'Headache', 47),
(62, 2020, 'September', 9, 'Rainy', 'Flu', 8),
(63, 2020, 'September', 9, 'Rainy', 'Fever', 13),
(64, 2020, 'September', 9, 'Rainy', 'Asthma', 32),
(65, 2020, 'September', 9, 'Rainy', 'Hypertension', 13),
(66, 2020, 'September', 9, 'Rainy', 'Dengue', 22),
(67, 2020, 'November', 11, 'Rainy', 'Flu', 40),
(68, 2020, 'November', 11, 'Rainy', 'Fever', 43),
(69, 2020, 'November', 11, 'Rainy', 'Hypertension', 66),
(70, 2020, 'November', 11, 'Rainy', 'Sore Throat', 50),
(71, 2020, 'November', 11, 'Rainy', 'Asthma', 39),
(72, 2020, 'November', 11, 'Rainy', 'Dengue', 13),
(73, 2020, 'November', 11, 'Rainy', 'Diarrhea', 14),
(74, 2020, 'November', 11, 'Rainy', 'Cough and Colds', 23),
(75, 2020, 'December', 12, 'Sunny', 'Cough and Colds', 34),
(76, 2020, 'December', 12, 'Sunny', 'Flu', 47),
(77, 2020, 'December', 12, 'Sunny', 'Asthma', 97),
(78, 2020, 'December', 12, 'Sunny', 'Tuberculosis (TB)', 20),
(79, 2020, 'December', 12, 'Sunny', 'Body Pain', 14),
(80, 2020, 'December', 12, 'Sunny', 'Diarrhea', 66),
(81, 2020, 'December', 12, 'Sunny', 'Toothache', 18),
(82, 2021, 'January', 1, 'Sunny', 'Dengue', 54),
(83, 2021, 'January', 1, 'Sunny', 'Toothache', 44),
(84, 2021, 'January', 1, 'Sunny', 'Fever', 109),
(85, 2021, 'January', 1, 'Sunny', 'Flu', 11),
(86, 2021, 'January', 1, 'Sunny', 'Cough and Colds', 29),
(87, 2021, 'January', 1, 'Sunny', 'Sore Throat', 20),
(88, 2021, 'January', 1, 'Sunny', 'Skin Infection', 14),
(89, 2021, 'February', 2, 'Sunny', 'Sore Throat', 4),
(90, 2021, 'February', 2, 'Sunny', 'Flu', 27),
(91, 2021, 'February', 2, 'Sunny', 'Dengue', 81),
(92, 2021, 'February', 2, 'Sunny', 'Fever', 10),
(93, 2021, 'February', 2, 'Sunny', 'Hypertension', 22),
(94, 2021, 'February', 2, 'Sunny', 'Tuberculosis (TB)', 16),
(95, 2021, 'February', 2, 'Sunny', 'Toothache', 43),
(96, 2021, 'February', 2, 'Sunny', 'Asthma', 45),
(97, 2021, 'March', 3, 'Sunny', 'Diarrhea', 4),
(98, 2021, 'March', 3, 'Sunny', 'Flu', 24),
(99, 2021, 'March', 3, 'Sunny', 'Cough and Colds', 32),
(100, 2021, 'March', 3, 'Sunny', 'Dengue', 16),
(101, 2021, 'March', 3, 'Sunny', 'Asthma', 63),
(102, 2021, 'March', 3, 'Sunny', 'Toothache', 42),
(103, 2021, 'March', 3, 'Sunny', 'Skin Infection', 49),
(104, 2021, 'March', 3, 'Sunny', 'Fever', 40),
(105, 2021, 'April', 4, 'Sunny', 'Fever', 4),
(106, 2021, 'April', 4, 'Sunny', 'Hypertension', 16),
(107, 2021, 'April', 4, 'Sunny', 'Skin Infection', 32),
(108, 2021, 'April', 4, 'Sunny', 'Flu', 38),
(109, 2021, 'April', 4, 'Sunny', 'Diarrhea', 95),
(110, 2021, 'April', 4, 'Sunny', 'Asthma', 63),
(111, 2021, 'April', 4, 'Sunny', 'Cough and Colds', 16),
(112, 2021, 'May', 5, 'Sunny', 'Toothache', 5),
(113, 2021, 'May', 5, 'Sunny', 'Fever', 17),
(114, 2021, 'May', 5, 'Sunny', 'Cough and Colds', 32),
(115, 2021, 'May', 5, 'Sunny', 'Flu', 17),
(116, 2021, 'May', 5, 'Sunny', 'Dengue', 50),
(117, 2021, 'May', 5, 'Sunny', 'Body Pain', 8),
(118, 2021, 'May', 5, 'Sunny', 'Hypertension', 37),
(119, 2021, 'May', 5, 'Sunny', 'Asthma', 17),
(120, 2021, 'May', 5, 'Sunny', 'Tuberculosis (TB)', 17),
(121, 2021, 'May', 5, 'Sunny', 'Skin Infection', 41),
(122, 2021, 'May', 5, 'Sunny', 'Diarrhea', 32),
(123, 2021, 'June', 6, 'Rainy', 'Cough and Colds', 36),
(124, 2021, 'June', 6, 'Rainy', 'Headache', 16),
(125, 2021, 'June', 6, 'Rainy', 'Flu', 81),
(126, 2021, 'June', 6, 'Rainy', 'Fever', 15),
(127, 2021, 'June', 6, 'Rainy', 'Skin Infection', 67),
(128, 2021, 'June', 6, 'Rainy', 'Diarrhea', 23),
(129, 2021, 'June', 6, 'Rainy', 'Dengue', 30),
(130, 2021, 'July', 7, 'Rainy', 'Flu', 53),
(131, 2021, 'July', 7, 'Rainy', 'Diarrhea', 29),
(132, 2021, 'July', 7, 'Rainy', 'Asthma', 98),
(133, 2021, 'July', 7, 'Rainy', 'Toothache', 52),
(134, 2021, 'July', 7, 'Rainy', 'Cough and Colds', 14),
(135, 2021, 'July', 7, 'Rainy', 'Skin Infection', 22),
(136, 2021, 'July', 7, 'Rainy', 'Dengue', 17),
(137, 2021, 'August', 8, 'Rainy', 'Diarrhea', 4),
(138, 2021, 'August', 8, 'Rainy', 'Toothache', 17),
(139, 2021, 'August', 8, 'Rainy', 'Fever', 52),
(140, 2021, 'August', 8, 'Rainy', 'Dengue', 52),
(141, 2021, 'August', 8, 'Rainy', 'Cough and Colds', 63),
(142, 2021, 'August', 8, 'Rainy', 'Hypertension', 43),
(143, 2021, 'August', 8, 'Rainy', 'Tuberculosis (TB)', 33),
(144, 2021, 'August', 8, 'Rainy', 'Body Pain', 22),
(145, 2021, 'September', 9, 'Rainy', 'Flu', 36),
(146, 2021, 'September', 9, 'Rainy', 'Sore Throat', 16),
(147, 2021, 'September', 9, 'Rainy', 'Diarrhea', 63),
(148, 2021, 'September', 9, 'Rainy', 'Tuberculosis (TB)', 48),
(149, 2021, 'September', 9, 'Rainy', 'Dengue', 37),
(150, 2021, 'September', 9, 'Rainy', 'Fever', 8),
(151, 2021, 'September', 9, 'Rainy', 'Skin Infection', 11),
(152, 2021, 'September', 9, 'Rainy', 'Toothache', 35),
(153, 2021, 'September', 9, 'Rainy', 'Cough and Colds', 15),
(154, 2021, 'November', 11, 'Rainy', 'Skin Infection', 53),
(155, 2021, 'November', 11, 'Rainy', 'Flu', 99),
(156, 2021, 'November', 11, 'Rainy', 'Toothache', 83),
(157, 2021, 'November', 11, 'Rainy', 'Cough and Colds', 40),
(158, 2021, 'November', 11, 'Rainy', 'Asthma', 13),
(159, 2021, 'December', 12, 'Sunny', 'Flu', 131),
(160, 2021, 'December', 12, 'Sunny', 'Diarrhea', 101),
(161, 2021, 'December', 12, 'Sunny', 'Hypertension', 12),
(162, 2021, 'December', 12, 'Sunny', 'Fever', 4),
(163, 2021, 'December', 12, 'Sunny', 'Skin Infection', 21),
(164, 2021, 'December', 12, 'Sunny', 'Cough and Colds', 14),
(165, 2021, 'December', 12, 'Sunny', 'Tuberculosis (TB)', 15),
(166, 2022, 'January', 1, 'Sunny', 'Fever', 49),
(167, 2022, 'January', 1, 'Sunny', 'Hypertension', 30),
(168, 2022, 'January', 1, 'Sunny', 'Flu', 94),
(169, 2022, 'January', 1, 'Sunny', 'Diarrhea', 32),
(170, 2022, 'January', 1, 'Sunny', 'Skin Infection', 14),
(171, 2022, 'January', 1, 'Sunny', 'Headache', 50),
(172, 2022, 'January', 1, 'Sunny', 'Cough and Colds', 14),
(173, 2022, 'February', 2, 'Sunny', 'Cough and Colds', 39),
(174, 2022, 'February', 2, 'Sunny', 'Dengue', 47),
(175, 2022, 'February', 2, 'Sunny', 'Flu', 117),
(176, 2022, 'February', 2, 'Sunny', 'Headache', 10),
(177, 2022, 'February', 2, 'Sunny', 'Asthma', 13),
(178, 2022, 'February', 2, 'Sunny', 'Fever', 13),
(179, 2022, 'February', 2, 'Sunny', 'Toothache', 15),
(180, 2022, 'March', 3, 'Sunny', 'Flu', 55),
(181, 2022, 'March', 3, 'Sunny', 'Cough and Colds', 87),
(182, 2022, 'March', 3, 'Sunny', 'Hypertension', 16),
(183, 2022, 'March', 3, 'Sunny', 'Toothache', 8),
(184, 2022, 'March', 3, 'Sunny', 'Headache', 17),
(185, 2022, 'March', 3, 'Sunny', 'Fever', 56),
(186, 2022, 'March', 3, 'Sunny', 'Asthma', 33),
(187, 2022, 'April', 4, 'Sunny', 'Body Pain', 4),
(188, 2022, 'April', 4, 'Sunny', 'Cough and Colds', 16),
(189, 2022, 'April', 4, 'Sunny', 'Asthma', 32),
(190, 2022, 'April', 4, 'Sunny', 'Fever', 39),
(191, 2022, 'April', 4, 'Sunny', 'Diarrhea', 49),
(192, 2022, 'April', 4, 'Sunny', 'Headache', 8),
(193, 2022, 'April', 4, 'Sunny', 'Toothache', 69),
(194, 2022, 'April', 4, 'Sunny', 'Flu', 15),
(195, 2022, 'April', 4, 'Sunny', 'Sore Throat', 32),
(196, 2022, 'May', 5, 'Sunny', 'Dengue', 5),
(197, 2022, 'May', 5, 'Sunny', 'Hypertension', 66),
(198, 2022, 'May', 5, 'Sunny', 'Fever', 108),
(199, 2022, 'May', 5, 'Sunny', 'Body Pain', 8),
(200, 2022, 'May', 5, 'Sunny', 'Asthma', 25),
(201, 2022, 'May', 5, 'Sunny', 'Toothache', 14),
(202, 2022, 'May', 5, 'Sunny', 'Cough and Colds', 16),
(203, 2022, 'May', 5, 'Sunny', 'Tuberculosis (TB)', 33),
(204, 2022, 'June', 6, 'Rainy', 'Cough and Colds', 145),
(205, 2022, 'June', 6, 'Rainy', 'Diarrhea', 8),
(206, 2022, 'June', 6, 'Rainy', 'Asthma', 14),
(207, 2022, 'June', 6, 'Rainy', 'Toothache', 16),
(208, 2022, 'June', 6, 'Rainy', 'Dengue', 16),
(209, 2022, 'June', 6, 'Rainy', 'Headache', 40),
(210, 2022, 'June', 6, 'Rainy', 'Flu', 33),
(211, 2022, 'July', 7, 'Rainy', 'Flu', 34),
(212, 2022, 'July', 7, 'Rainy', 'Asthma', 54),
(213, 2022, 'July', 7, 'Rainy', 'Skin Infection', 51),
(214, 2022, 'July', 7, 'Rainy', 'Toothache', 91),
(215, 2022, 'July', 7, 'Rainy', 'Cough and Colds', 43),
(216, 2022, 'July', 7, 'Rainy', 'Body Pain', 14),
(217, 2022, 'August', 8, 'Rainy', 'Dengue', 4),
(218, 2022, 'August', 8, 'Rainy', 'Tuberculosis (TB)', 65),
(219, 2022, 'August', 8, 'Rainy', 'Hypertension', 109),
(220, 2022, 'August', 8, 'Rainy', 'Body Pain', 52),
(221, 2022, 'August', 8, 'Rainy', 'Asthma', 9),
(222, 2022, 'August', 8, 'Rainy', 'Cough and Colds', 26),
(223, 2022, 'August', 8, 'Rainy', 'Diarrhea', 22),
(224, 2022, 'September', 9, 'Rainy', 'Hypertension', 4),
(225, 2022, 'September', 9, 'Rainy', 'Headache', 16),
(226, 2022, 'September', 9, 'Rainy', 'Fever', 49),
(227, 2022, 'September', 9, 'Rainy', 'Flu', 127),
(228, 2022, 'September', 9, 'Rainy', 'Diarrhea', 29),
(229, 2022, 'September', 9, 'Rainy', 'Skin Infection', 11),
(230, 2022, 'September', 9, 'Rainy', 'Dengue', 32),
(231, 2022, 'November', 11, 'Rainy', 'Flu', 44),
(232, 2022, 'November', 11, 'Rainy', 'Body Pain', 17),
(233, 2022, 'November', 11, 'Rainy', 'Dengue', 52),
(234, 2022, 'November', 11, 'Rainy', 'Headache', 52),
(235, 2022, 'November', 11, 'Rainy', 'Diarrhea', 9),
(236, 2022, 'November', 11, 'Rainy', 'Skin Infection', 26),
(237, 2022, 'November', 11, 'Rainy', 'Toothache', 33),
(238, 2022, 'November', 11, 'Rainy', 'Fever', 54),
(239, 2022, 'December', 12, 'Sunny', 'Cough and Colds', 136),
(240, 2022, 'December', 12, 'Sunny', 'Diarrhea', 13),
(241, 2022, 'December', 12, 'Sunny', 'Fever', 81),
(242, 2022, 'December', 12, 'Sunny', 'Tuberculosis (TB)', 4),
(243, 2022, 'December', 12, 'Sunny', 'Dengue', 22),
(244, 2022, 'December', 12, 'Sunny', 'Asthma', 14),
(245, 2022, 'December', 12, 'Sunny', 'Toothache', 14),
(246, 2022, 'December', 12, 'Sunny', 'Body Pain', 14),
(247, 2023, 'January', 1, 'Sunny', 'Diarrhea', 4),
(248, 2023, 'January', 1, 'Sunny', 'Flu', 48),
(249, 2023, 'January', 1, 'Sunny', 'Tuberculosis (TB)', 84),
(250, 2023, 'January', 1, 'Sunny', 'Fever', 26),
(251, 2023, 'January', 1, 'Sunny', 'Cough and Colds', 26),
(252, 2023, 'January', 1, 'Sunny', 'Asthma', 70),
(253, 2023, 'January', 1, 'Sunny', 'Skin Infection', 14),
(254, 2023, 'January', 1, 'Sunny', 'Toothache', 14),
(255, 2023, 'February', 2, 'Sunny', 'Fever', 3),
(256, 2023, 'February', 2, 'Sunny', 'Skin Infection', 29),
(257, 2023, 'February', 2, 'Sunny', 'Flu', 124),
(258, 2023, 'February', 2, 'Sunny', 'Hypertension', 23),
(259, 2023, 'February', 2, 'Sunny', 'Cough and Colds', 3),
(260, 2023, 'February', 2, 'Sunny', 'Asthma', 13),
(261, 2023, 'February', 2, 'Sunny', 'Tuberculosis (TB)', 45),
(262, 2023, 'February', 2, 'Sunny', 'Toothache', 15),
(263, 2023, 'March', 3, 'Sunny', 'Flu', 69),
(264, 2023, 'March', 3, 'Sunny', 'Fever', 73),
(265, 2023, 'March', 3, 'Sunny', 'Body Pain', 71),
(266, 2023, 'March', 3, 'Sunny', 'Toothache', 24),
(267, 2023, 'March', 3, 'Sunny', 'Cough and Colds', 32),
(268, 2023, 'April', 4, 'Sunny', 'Toothache', 20),
(269, 2023, 'April', 4, 'Sunny', 'Diarrhea', 16),
(270, 2023, 'April', 4, 'Sunny', 'Cough and Colds', 32),
(271, 2023, 'April', 4, 'Sunny', 'Fever', 137),
(272, 2023, 'April', 4, 'Sunny', 'Dengue', 8),
(273, 2023, 'April', 4, 'Sunny', 'Flu', 25),
(274, 2023, 'April', 4, 'Sunny', 'Asthma', 16),
(275, 2023, 'April', 4, 'Sunny', 'Headache', 16),
(276, 2023, 'May', 5, 'Sunny', 'Toothache', 47),
(277, 2023, 'May', 5, 'Sunny', 'Diarrhea', 17),
(278, 2023, 'May', 5, 'Sunny', 'Flu', 102),
(279, 2023, 'May', 5, 'Sunny', 'Dengue', 25),
(280, 2023, 'May', 5, 'Sunny', 'Headache', 31),
(281, 2023, 'May', 5, 'Sunny', 'Hypertension', 17),
(282, 2023, 'May', 5, 'Sunny', 'Cough and Colds', 43),
(283, 2023, 'June', 6, 'Rainy', 'Diarrhea', 5),
(284, 2023, 'June', 6, 'Rainy', 'Fever', 41),
(285, 2023, 'June', 6, 'Rainy', 'Toothache', 32),
(286, 2023, 'June', 6, 'Rainy', 'Cough and Colds', 74),
(287, 2023, 'June', 6, 'Rainy', 'Asthma', 47),
(288, 2023, 'June', 6, 'Rainy', 'Tuberculosis (TB)', 17),
(289, 2023, 'June', 6, 'Rainy', 'Skin Infection', 17),
(290, 2023, 'June', 6, 'Rainy', 'Hypertension', 40),
(291, 2023, 'July', 7, 'Rainy', 'Hypertension', 67),
(292, 2023, 'July', 7, 'Rainy', 'Diarrhea', 31),
(293, 2023, 'July', 7, 'Rainy', 'Dengue', 51),
(294, 2023, 'July', 7, 'Rainy', 'Toothache', 40),
(295, 2023, 'July', 7, 'Rainy', 'Asthma', 27),
(296, 2023, 'July', 7, 'Rainy', 'Cough and Colds', 14),
(297, 2023, 'July', 7, 'Rainy', 'Tuberculosis (TB)', 35),
(298, 2023, 'July', 7, 'Rainy', 'Fever', 22),
(299, 2023, 'August', 8, 'Rainy', 'Cough and Colds', 78),
(300, 2023, 'August', 8, 'Rainy', 'Flu', 65),
(301, 2023, 'August', 8, 'Rainy', 'Hypertension', 52),
(302, 2023, 'August', 8, 'Rainy', 'Body Pain', 40),
(303, 2023, 'August', 8, 'Rainy', 'Toothache', 9),
(304, 2023, 'August', 8, 'Rainy', 'Diarrhea', 14),
(305, 2023, 'August', 8, 'Rainy', 'Headache', 14),
(306, 2023, 'August', 8, 'Rainy', 'Skin Infection', 17),
(307, 2023, 'September', 9, 'Rainy', 'Fever', 68),
(308, 2023, 'September', 9, 'Rainy', 'Diarrhea', 48),
(309, 2023, 'September', 9, 'Rainy', 'Dengue', 50),
(310, 2023, 'September', 9, 'Rainy', 'Flu', 8),
(311, 2023, 'September', 9, 'Rainy', 'Cough and Colds', 11),
(312, 2023, 'September', 9, 'Rainy', 'Asthma', 32),
(313, 2023, 'September', 9, 'Rainy', 'Tuberculosis (TB)', 13),
(314, 2023, 'September', 9, 'Rainy', 'Hypertension', 22),
(315, 2023, 'September', 9, 'Rainy', 'Body Pain', 15),
(316, 2023, 'November', 11, 'Rainy', 'Skin Infection', 25),
(317, 2023, 'November', 11, 'Rainy', 'Flu', 17),
(318, 2023, 'November', 11, 'Rainy', 'Headache', 52),
(319, 2023, 'November', 11, 'Rainy', 'Asthma', 52),
(320, 2023, 'November', 11, 'Rainy', 'Fever', 54),
(321, 2023, 'November', 11, 'Rainy', 'Cough and Colds', 36),
(322, 2023, 'November', 11, 'Rainy', 'Hypertension', 14),
(323, 2023, 'November', 11, 'Rainy', 'Diarrhea', 40),
(324, 2023, 'December', 12, 'Sunny', 'Flu', 4),
(325, 2023, 'December', 12, 'Sunny', 'Fever', 55),
(326, 2023, 'December', 12, 'Sunny', 'Cough and Colds', 96),
(327, 2023, 'December', 12, 'Sunny', 'Diarrhea', 28),
(328, 2023, 'December', 12, 'Sunny', 'Toothache', 32),
(329, 2023, 'December', 12, 'Sunny', 'Dengue', 15),
(330, 2023, 'December', 12, 'Sunny', 'Hypertension', 54),
(331, 2023, 'December', 12, 'Sunny', 'Headache', 19),
(332, 2024, 'January', 1, 'Sunny', 'Toothache', 4),
(333, 2024, 'January', 1, 'Sunny', 'Headache', 49),
(334, 2024, 'January', 1, 'Sunny', 'Asthma', 108),
(335, 2024, 'January', 1, 'Sunny', 'Cough and Colds', 64),
(336, 2024, 'January', 1, 'Sunny', 'Dengue', 40),
(337, 2024, 'January', 1, 'Sunny', 'Flu', 4),
(338, 2024, 'January', 1, 'Sunny', 'Fever', 20),
(339, 2024, 'February', 2, 'Sunny', 'Dengue', 32),
(340, 2024, 'February', 2, 'Sunny', 'Asthma', 86),
(341, 2024, 'February', 2, 'Sunny', 'Toothache', 24),
(342, 2024, 'February', 2, 'Sunny', 'Diarrhea', 22),
(343, 2024, 'February', 2, 'Sunny', 'Cough and Colds', 17),
(344, 2024, 'February', 2, 'Sunny', 'Fever', 62),
(345, 2024, 'February', 2, 'Sunny', 'Flu', 13),
(346, 2024, 'March', 3, 'Sunny', 'Skin Infection', 5),
(347, 2024, 'March', 3, 'Sunny', 'Cough and Colds', 66),
(348, 2024, 'March', 3, 'Sunny', 'Flu', 75),
(349, 2024, 'March', 3, 'Sunny', 'Asthma', 49),
(350, 2024, 'March', 3, 'Sunny', 'Diarrhea', 14),
(351, 2024, 'March', 3, 'Sunny', 'Hypertension', 17),
(352, 2024, 'March', 3, 'Sunny', 'Fever', 17),
(353, 2024, 'March', 3, 'Sunny', 'Dengue', 33),
(354, 2024, 'April', 4, 'Sunny', 'Cough and Colds', 43),
(355, 2024, 'April', 4, 'Sunny', 'Asthma', 48),
(356, 2024, 'April', 4, 'Sunny', 'Flu', 16),
(357, 2024, 'April', 4, 'Sunny', 'Diarrhea', 49),
(358, 2024, 'April', 4, 'Sunny', 'Skin Infection', 8),
(359, 2024, 'April', 4, 'Sunny', 'Dengue', 57),
(360, 2024, 'April', 4, 'Sunny', 'Sore Throat', 15),
(361, 2024, 'April', 4, 'Sunny', 'Fever', 32),
(362, 2024, 'May', 5, 'Sunny', 'Toothache', 80),
(363, 2024, 'May', 5, 'Sunny', 'Diarrhea', 43),
(364, 2024, 'May', 5, 'Sunny', 'Hypertension', 35),
(365, 2024, 'May', 5, 'Sunny', 'Cough and Colds', 14),
(366, 2024, 'May', 5, 'Sunny', 'Flu', 33),
(367, 2024, 'May', 5, 'Sunny', 'Skin Infection', 42),
(368, 2024, 'May', 5, 'Sunny', 'Fever', 35),
(369, 2024, 'June', 6, 'Rainy', 'Cough and Colds', 56),
(370, 2024, 'June', 6, 'Rainy', 'Hypertension', 76),
(371, 2024, 'June', 6, 'Rainy', 'Flu', 55),
(372, 2024, 'June', 6, 'Rainy', 'Skin Infection', 49),
(373, 2024, 'June', 6, 'Rainy', 'Toothache', 8),
(374, 2024, 'June', 6, 'Rainy', 'Sore Throat', 16),
(375, 2024, 'June', 6, 'Rainy', 'Diarrhea', 16),
(376, 2024, 'July', 7, 'Rainy', 'Fever', 56),
(377, 2024, 'July', 7, 'Rainy', 'Skin Infection', 53),
(378, 2024, 'July', 7, 'Rainy', 'Toothache', 69),
(379, 2024, 'July', 7, 'Rainy', 'Cough and Colds', 41),
(380, 2024, 'July', 7, 'Rainy', 'Dengue', 9),
(381, 2024, 'July', 7, 'Rainy', 'Tuberculosis (TB)', 28),
(382, 2024, 'July', 7, 'Rainy', 'Diarrhea', 12),
(383, 2024, 'July', 7, 'Rainy', 'Hypertension', 23),
(384, 2024, 'August', 8, 'Rainy', 'Fever', 70),
(385, 2024, 'August', 8, 'Rainy', 'Skin Infection', 40),
(386, 2024, 'August', 8, 'Rainy', 'Flu', 52),
(387, 2024, 'August', 8, 'Rainy', 'Dengue', 40),
(388, 2024, 'August', 8, 'Rainy', 'Hypertension', 23),
(389, 2024, 'August', 8, 'Rainy', 'Headache', 47),
(390, 2024, 'August', 8, 'Rainy', 'Cough and Colds', 18),
(391, 2024, 'September', 9, 'Rainy', 'Flu', 4),
(392, 2024, 'September', 9, 'Rainy', 'Toothache', 66),
(393, 2024, 'September', 9, 'Rainy', 'Cough and Colds', 99),
(394, 2024, 'September', 9, 'Rainy', 'Diarrhea', 8),
(395, 2024, 'September', 9, 'Rainy', 'Hypertension', 14),
(396, 2024, 'September', 9, 'Rainy', 'Headache', 32),
(397, 2024, 'September', 9, 'Rainy', 'Asthma', 14),
(398, 2024, 'September', 9, 'Rainy', 'Skin Infection', 22),
(399, 2024, 'September', 9, 'Rainy', 'Fever', 16),
(400, 2024, 'November', 11, 'Rainy', 'Cough and Colds', 4),
(401, 2024, 'November', 11, 'Rainy', 'Diarrhea', 31),
(402, 2024, 'November', 11, 'Rainy', 'Skin Infection', 104),
(403, 2024, 'November', 11, 'Rainy', 'Fever', 50),
(404, 2024, 'November', 11, 'Rainy', 'Headache', 14),
(405, 2024, 'November', 11, 'Rainy', 'Tuberculosis (TB)', 36),
(406, 2024, 'November', 11, 'Rainy', 'Toothache', 32),
(407, 2024, 'November', 11, 'Rainy', 'Flu', 23),
(408, 2024, 'December', 12, 'Sunny', 'Diarrhea', 37),
(409, 2024, 'December', 12, 'Sunny', 'Skin Infection', 48),
(410, 2024, 'December', 12, 'Sunny', 'Dengue', 85),
(411, 2024, 'December', 12, 'Sunny', 'Body Pain', 13),
(412, 2024, 'December', 12, 'Sunny', 'Fever', 27),
(413, 2024, 'December', 12, 'Sunny', 'Hypertension', 4),
(414, 2024, 'December', 12, 'Sunny', 'Flu', 22),
(415, 2024, 'December', 12, 'Sunny', 'Sore Throat', 70),
(416, 2025, 'January', 1, 'Sunny', 'Skin Infection', 4),
(417, 2025, 'January', 1, 'Sunny', 'Cough and Colds', 36),
(418, 2025, 'January', 1, 'Sunny', 'Dengue', 97),
(419, 2025, 'January', 1, 'Sunny', 'Toothache', 12),
(420, 2025, 'January', 1, 'Sunny', 'Fever', 26),
(421, 2025, 'January', 1, 'Sunny', 'Diarrhea', 73),
(422, 2025, 'January', 1, 'Sunny', 'Tuberculosis (TB)', 14),
(423, 2025, 'January', 1, 'Sunny', 'Flu', 28),
(424, 2025, 'February', 2, 'Sunny', 'Diarrhea', 84),
(425, 2025, 'February', 2, 'Sunny', 'Toothache', 40),
(426, 2025, 'February', 2, 'Sunny', 'Cough and Colds', 46),
(427, 2025, 'February', 2, 'Sunny', 'Dengue', 21),
(428, 2025, 'February', 2, 'Sunny', 'Body Pain', 45),
(429, 2025, 'February', 2, 'Sunny', 'Flu', 14),
(430, 2025, 'March', 3, 'Sunny', 'Toothache', 30),
(431, 2025, 'March', 3, 'Sunny', 'Diarrhea', 17),
(432, 2025, 'March', 3, 'Sunny', 'Fever', 83),
(433, 2025, 'March', 3, 'Sunny', 'Cough and Colds', 109),
(434, 2025, 'March', 3, 'Sunny', 'Asthma', 25),
(435, 2025, 'March', 3, 'Sunny', 'Headache', 14),
(436, 2025, 'April', 4, 'Sunny', 'Fever', 29),
(437, 2025, 'April', 4, 'Sunny', 'Flu', 57),
(438, 2025, 'April', 4, 'Sunny', 'Diarrhea', 65),
(439, 2025, 'April', 4, 'Sunny', 'Cough and Colds', 48),
(440, 2025, 'April', 4, 'Sunny', 'Asthma', 8),
(441, 2025, 'April', 4, 'Sunny', 'Skin Infection', 14),
(442, 2025, 'April', 4, 'Sunny', 'Headache', 16),
(443, 2025, 'April', 4, 'Sunny', 'Dengue', 33),
(444, 2025, 'May', 5, 'Sunny', 'Skin Infection', 107),
(445, 2025, 'May', 5, 'Sunny', 'Toothache', 51),
(446, 2025, 'May', 5, 'Sunny', 'Cough and Colds', 60),
(447, 2025, 'May', 5, 'Sunny', 'Hypertension', 9),
(448, 2025, 'May', 5, 'Sunny', 'Asthma', 39),
(449, 2025, 'May', 5, 'Sunny', 'Sore Throat', 17),
(450, 2025, 'June', 6, 'Rainy', 'Toothache', 22),
(451, 2025, 'June', 6, 'Rainy', 'Skin Infection', 31),
(452, 2025, 'June', 6, 'Rainy', 'Diarrhea', 42),
(453, 2025, 'June', 6, 'Rainy', 'Cough and Colds', 16),
(454, 2025, 'June', 6, 'Rainy', 'Fever', 50),
(455, 2025, 'June', 6, 'Rainy', 'Tuberculosis (TB)', 25),
(456, 2025, 'June', 6, 'Rainy', 'Flu', 91),
(457, 2025, 'July', 7, 'Rainy', 'Fever', 115),
(458, 2025, 'July', 7, 'Rainy', 'Toothache', 32),
(459, 2025, 'July', 7, 'Rainy', 'Flu', 51),
(460, 2025, 'July', 7, 'Rainy', 'Skin Infection', 52),
(461, 2025, 'July', 7, 'Rainy', 'Cough and Colds', 27),
(462, 2025, 'July', 7, 'Rainy', 'Asthma', 14),
(463, 2025, 'August', 8, 'Rainy', 'Body Pain', 4),
(464, 2025, 'August', 8, 'Rainy', 'Hypertension', 58),
(465, 2025, 'August', 8, 'Rainy', 'Flu', 75),
(466, 2025, 'August', 8, 'Rainy', 'Asthma', 52),
(467, 2025, 'August', 8, 'Rainy', 'Dengue', 9),
(468, 2025, 'August', 8, 'Rainy', 'Fever', 26),
(469, 2025, 'August', 8, 'Rainy', 'Toothache', 36),
(470, 2025, 'August', 8, 'Rainy', 'Skin Infection', 14),
(471, 2025, 'August', 8, 'Rainy', 'Diarrhea', 18),
(472, 2025, 'September', 9, 'Rainy', 'Dengue', 4),
(473, 2025, 'September', 9, 'Rainy', 'Body Pain', 17),
(474, 2025, 'September', 9, 'Rainy', 'Skin Infection', 98),
(475, 2025, 'September', 9, 'Rainy', 'Cough and Colds', 38),
(476, 2025, 'September', 9, 'Rainy', 'Flu', 43),
(477, 2025, 'September', 9, 'Rainy', 'Diarrhea', 46),
(478, 2025, 'September', 9, 'Rainy', 'Fever', 12),
(479, 2025, 'September', 9, 'Rainy', 'Tuberculosis (TB)', 15),
(512, 2024, 'January', 1, 'Sunny', 'Essential Hypertension', 1),
(513, 2025, 'September', 9, 'Rainy', 'acute_gastroenteritis', 2),
(514, 2025, 'September', 9, 'Rainy', 'asthma_exacerbation', 1),
(515, 2025, 'September', 9, 'Rainy', 'fever_unspecified', 2),
(516, 2025, 'October', 10, 'Rainy', 'Bronchial Asthma', 1),
(517, 2025, 'October', 10, 'Rainy', 'dengue_suspected', 7),
(522, 2025, 'October', 10, 'Rainy', 'Flu', 2);

-- --------------------------------------------------------

--
-- Table structure for table `medical_history`
--

CREATE TABLE `medical_history` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `condition_value` varchar(255) NOT NULL,
  `specify` varchar(255) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `medical_history`
--

INSERT INTO `medical_history` (`id`, `user_id`, `condition_value`, `specify`, `created_at`) VALUES
(1, 6, 'Breathing Problems,Sleep Apnea', '', '2025-07-27 14:35:36');

-- --------------------------------------------------------

--
-- Table structure for table `medicines`
--

CREATE TABLE `medicines` (
  `medicine_id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `description` text DEFAULT NULL,
  `stock_quantity` int(11) DEFAULT 0,
  `status` enum('available','unavailable') DEFAULT 'available',
  `category` enum('medicine','vitamin') NOT NULL DEFAULT 'medicine'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `medicines`
--

INSERT INTO `medicines` (`medicine_id`, `name`, `description`, `stock_quantity`, `status`, `category`) VALUES
(1, 'Metformin', 'Used to control blood sugar levels for diabetes patients', 74, 'available', 'medicine'),
(2, 'Losartan', 'Used to treat high blood pressure', 60, 'available', 'medicine'),
(3, 'Salbutamol', 'Reliever for asthma symptoms and bronchospasm', 50, 'available', 'medicine'),
(4, 'Loperamide', 'Treats diarrhea', 100, 'available', 'medicine'),
(5, 'Cetirizine', 'Used for allergies and itching', 100, 'available', 'medicine'),
(6, 'Mefenamic Acid', 'Pain reliever for headache, toothache, and dysmenorrhea', 90, 'available', 'medicine'),
(7, 'Ferrous Sulfate', 'Iron supplement to treat anemia', 150, 'available', 'medicine'),
(8, 'Zinc Sulfate', 'Boosts immune function and helps prevent infections', 110, 'available', 'medicine'),
(9, 'ORS (Oral Rehydration Salts)', 'Rehydrates body and replaces lost fluids due to diarrhea', 200, 'available', 'medicine'),
(10, 'Multivitamins', 'General health supplement', 177, 'available', 'medicine'),
(11, 'Hydrocortisone Cream', 'Used for skin inflammation, eczema, or rashes', 40, 'available', 'medicine'),
(12, 'Clotrimazole', 'Antifungal cream for skin infections', 35, 'available', 'medicine'),
(13, 'Vitamin C', 'Boosts immune system', 48, 'available', 'vitamin'),
(14, 'Multivitamins', 'General supplement for daily needs', -30, 'available', 'vitamin'),
(15, 'Vitamin D3', 'Supports bone health', 40, 'available', 'vitamin'),
(16, 'Iron Supplement', 'Helps with anemia', 30, 'available', 'vitamin'),
(17, 'Vitamin B-Complex', 'Boosts energy and brain function', 25, 'available', 'vitamin'),
(18, 'Calcium', 'For bone and teeth strength', 35, 'available', 'vitamin'),
(19, 'Zinc', 'Supports metabolism and immune function', 15, 'available', 'vitamin');

-- --------------------------------------------------------

--
-- Table structure for table `medicine_requests`
--

CREATE TABLE `medicine_requests` (
  `request_id` int(11) NOT NULL,
  `patient_id` int(11) DEFAULT NULL,
  `medicine_id` int(11) DEFAULT NULL,
  `quantity` int(11) DEFAULT NULL,
  `status` enum('pending','approved','declined') DEFAULT 'pending',
  `request_date` timestamp NOT NULL DEFAULT current_timestamp(),
  `pickup_schedule` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `medicine_requests`
--

INSERT INTO `medicine_requests` (`request_id`, `patient_id`, `medicine_id`, `quantity`, `status`, `request_date`, `pickup_schedule`) VALUES
(11, 3, 1, 3, 'approved', '2025-07-21 08:50:15', '2025-07-28 16:56:00'),
(12, 3, 19, 10, '', '2025-07-21 08:50:15', NULL),
(13, 5, 1, 3, 'approved', '2025-07-22 03:42:46', '2025-07-29 11:43:00'),
(14, 5, 13, 2, '', '2025-07-22 03:42:46', NULL),
(15, 3, 13, 1, 'pending', '2025-07-30 08:15:08', NULL),
(16, 3, 15, 2, 'pending', '2025-07-30 08:16:13', NULL),
(17, 7, 14, 5, 'pending', '2025-08-01 16:10:02', NULL),
(18, 7, 13, 2, 'approved', '2025-08-01 16:10:02', '2025-08-04 15:30:00'),
(20, 13, 14, 10, 'approved', '2025-11-03 15:09:48', '2025-11-11 11:30:00'),
(21, 13, 1, 10, 'approved', '2025-11-03 15:33:35', '2025-11-04 11:00:00'),
(22, 14, 5, 20, 'approved', '2025-11-03 17:01:49', '2025-11-04 01:00:00');

-- --------------------------------------------------------

--
-- Table structure for table `meeting_tokens`
--

CREATE TABLE `meeting_tokens` (
  `id` int(11) NOT NULL,
  `meeting_id` int(11) NOT NULL,
  `role` enum('user','staff') NOT NULL,
  `token` char(64) NOT NULL,
  `display_name` varchar(80) DEFAULT NULL,
  `expires_at` datetime DEFAULT NULL,
  `max_uses` int(11) DEFAULT 0,
  `used_count` int(11) DEFAULT 0,
  `created_at` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `meeting_tokens`
--

INSERT INTO `meeting_tokens` (`id`, `meeting_id`, `role`, `token`, `display_name`, `expires_at`, `max_uses`, `used_count`, `created_at`) VALUES
(1, 4, 'user', 'acc84c9546c997a347dd11c3d4e0f1d0243997197c86b770f9d227ea954e3cea', 'Patient', '2025-10-12 16:20:44', 0, 0, '2025-10-09 16:20:44'),
(2, 4, 'staff', '0bb4406d6eebc34d991790c81aecab0035e80d1fc2d60dc1aaf5fe2aba93e057', 'Doctor', '2025-10-12 16:20:44', 0, 0, '2025-10-09 16:20:44');

-- --------------------------------------------------------

--
-- Table structure for table `messages`
--

CREATE TABLE `messages` (
  `id` int(11) NOT NULL,
  `sender_id` int(11) NOT NULL,
  `receiver_id` int(11) NOT NULL,
  `message` text NOT NULL,
  `created_at` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `messages`
--

INSERT INTO `messages` (`id`, `sender_id`, `receiver_id`, `message`, `created_at`) VALUES
(1, 9, 1, 'hi po', '2025-10-17 15:47:46'),
(2, 9, 1, 'mag tatanong lang', '2025-10-17 16:07:10'),
(3, 11, 1, 'can I ask?', '2025-10-17 16:09:28'),
(4, 9, 1, 'gumana na dapat', '2025-10-17 16:12:43');

-- --------------------------------------------------------

--
-- Table structure for table `oral_services`
--

CREATE TABLE `oral_services` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `full_name` varchar(100) DEFAULT NULL,
  `age` int(11) DEFAULT NULL,
  `contact_number` varchar(20) DEFAULT NULL,
  `oral_service` enum('tooth_extraction','dental_checkup','cleaning') NOT NULL,
  `preferred_schedule` datetime DEFAULT NULL,
  `doctor_id` int(11) DEFAULT NULL,
  `status` enum('pending','approved','rejected','expired') DEFAULT NULL,
  `consultation_status` enum('pending','in_progress','completed') DEFAULT 'pending',
  `findings` text DEFAULT NULL,
  `remarks` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `oral_services`
--

INSERT INTO `oral_services` (`id`, `user_id`, `full_name`, `age`, `contact_number`, `oral_service`, `preferred_schedule`, `doctor_id`, `status`, `consultation_status`, `findings`, `remarks`, `created_at`, `updated_at`) VALUES
(1, 4, 'Sofia Isabella  Connor', 30, '091757783421', 'cleaning', '2025-06-01 14:00:00', NULL, 'expired', 'pending', NULL, NULL, '2025-07-19 04:58:07', '2025-07-19 14:31:56'),
(2, 6, 'Joshua Rafael  Blanc', 17, '09465587562', 'cleaning', NULL, NULL, 'pending', 'pending', NULL, NULL, '2025-07-28 06:01:51', '2025-07-28 06:01:51'),
(3, 3, 'Charlotte Bella Nova', 29, '09557321458', 'dental_checkup', '2025-07-31 08:00:00', NULL, 'expired', 'pending', NULL, NULL, '2025-07-30 08:23:31', '2025-08-01 17:18:24'),
(4, 7, 'Lenaj Rolf', 30, '09544662627', 'cleaning', '2025-08-04 13:00:00', NULL, 'expired', 'pending', NULL, NULL, '2025-08-01 16:13:38', '2025-09-24 07:14:57'),
(5, 13, 'Angel Cruz', 30, '09670984729', 'cleaning', NULL, NULL, 'rejected', 'pending', NULL, NULL, '2025-11-03 16:02:03', '2025-11-03 16:02:19'),
(6, 13, 'Angel Cruz', 30, '09670984729', 'tooth_extraction', '2025-11-12 10:30:00', 2, 'approved', 'pending', NULL, NULL, '2025-11-03 16:03:01', '2025-11-03 16:05:27'),
(7, 13, 'Angel Cruz', 30, '09670984729', 'tooth_extraction', NULL, NULL, 'rejected', 'pending', NULL, NULL, '2025-11-03 16:07:22', '2025-11-03 16:09:49'),
(8, 13, 'Angel Cruz', 30, '09670984729', 'cleaning', NULL, NULL, 'pending', 'pending', NULL, NULL, '2025-11-03 16:10:00', '2025-11-03 16:10:00'),
(9, 13, 'Angel Cruz', 30, '09670984729', 'dental_checkup', '2025-11-17 09:30:00', 2, 'approved', 'pending', NULL, NULL, '2025-11-03 16:26:21', '2025-11-03 16:26:37'),
(10, 14, 'Cristina Albindo', 30, '09205057976', 'dental_checkup', '2025-11-04 15:00:00', 2, 'approved', 'pending', NULL, NULL, '2025-11-03 17:02:31', '2025-11-03 17:05:41');

-- --------------------------------------------------------

--
-- Table structure for table `password_reset`
--

CREATE TABLE `password_reset` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `token` varchar(255) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `expires_at` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `personal_medical_history`
--

CREATE TABLE `personal_medical_history` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `had_surgery` varchar(10) DEFAULT NULL,
  `surgery_type` varchar(255) DEFAULT NULL,
  `surgery_year` int(11) DEFAULT NULL,
  `recent_hospitalization` varchar(255) DEFAULT NULL,
  `allergic_meds` varchar(10) DEFAULT NULL,
  `meds_allergy_details` varchar(255) DEFAULT NULL,
  `allergic_foods` varchar(10) DEFAULT NULL,
  `foods_allergy_details` varchar(255) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `personal_medical_history`
--

INSERT INTO `personal_medical_history` (`id`, `user_id`, `had_surgery`, `surgery_type`, `surgery_year`, `recent_hospitalization`, `allergic_meds`, `meds_allergy_details`, `allergic_foods`, `foods_allergy_details`, `created_at`) VALUES
(1, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'Yes', 'Sea Foods', '2025-07-08 13:45:15'),
(2, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'Yes', 'Sea Foods', '2025-07-09 06:21:06'),
(3, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'Yes', 'Sea Foods', '2025-07-09 06:28:14'),
(4, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'Yes', 'Sea Foods', '2025-07-09 06:33:36'),
(5, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'Yes', 'Sea Foods', '2025-07-09 06:39:22'),
(6, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'Yes', 'Sea Foods', '2025-07-09 06:40:03'),
(7, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'Yes', 'Sea Foods', '2025-07-09 07:17:49'),
(8, 1, 'Yes', 'Appendectomy', 2007, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'Yes', 'Sea Foods', '2025-07-09 07:18:23'),
(9, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'Yes', 'Sea Foods', '2025-07-09 08:19:42'),
(10, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 08:20:11'),
(11, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 11:43:36'),
(12, 1, 'Yes', 'Appendectomy', 2015, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 15:33:12'),
(13, 1, 'Yes', 'Appendectomy', 2020, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 15:33:28'),
(14, 1, 'Yes', 'Appendectomy', 2020, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 15:49:23'),
(15, 1, 'Yes', 'Appendectomy', 2020, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 15:50:00'),
(16, 1, 'Yes', 'Appendectomy', 2020, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 15:51:55'),
(17, 1, 'Yes', 'Appendectomy', 2020, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 15:52:21'),
(18, 1, 'Yes', 'Appendectomy', 2020, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 15:53:51'),
(19, 1, 'Yes', 'Appendectomy', 2020, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 15:55:12'),
(20, 1, 'Yes', 'Appendectomy', 2020, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'No', '', '2025-07-09 16:05:54'),
(21, 2, 'No', '', 0, '', 'No', '', 'No', '', '2025-07-14 16:31:12'),
(22, 1, 'Yes', 'Appendectomy', 2020, 'St. Luke\'s BGC', 'Yes', 'Penicillin', 'Yes', 'Peanut', '2025-07-16 05:09:35'),
(23, 4, '', '', 0, '', '', '', 'Yes', 'Peanut', '2025-07-17 09:12:16'),
(24, 4, 'Yes', 'Caesarean', 2025, '', '', '', 'Yes', 'Peanut', '2025-07-17 11:35:43'),
(25, 4, 'Yes', 'Caesarean', 2025, '', '', '', 'Yes', 'Peanut', '2025-07-18 05:34:29'),
(26, 6, '', '', 0, '', '', '', 'No', 'Shrimp', '2025-07-27 13:37:07'),
(27, 9, 'No', '', 0, '', 'No', '', 'Yes', 'fish', '2025-09-10 09:00:54');

-- --------------------------------------------------------

--
-- Table structure for table `pregnancy_forms`
--

CREATE TABLE `pregnancy_forms` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `first_name` varchar(100) NOT NULL,
  `middle_initial` varchar(5) DEFAULT NULL,
  `last_name` varchar(100) NOT NULL,
  `suffix` varchar(10) DEFAULT NULL,
  `dob` date NOT NULL,
  `gender` enum('Male','Female') NOT NULL,
  `age` int(11) NOT NULL,
  `marital_status` varchar(50) DEFAULT NULL,
  `contact_number` varchar(20) DEFAULT NULL,
  `complete_address` varchar(255) DEFAULT NULL,
  `email_address` varchar(100) DEFAULT NULL,
  `pregnant` enum('yes','no') DEFAULT NULL,
  `miscarriage` enum('yes','no') DEFAULT NULL,
  `multiple` enum('yes','no') DEFAULT NULL,
  `weeks` int(11) DEFAULT NULL,
  `edd1` date DEFAULT NULL,
  `edd2` date DEFAULT NULL,
  `gravida` int(11) DEFAULT NULL,
  `births` int(11) DEFAULT NULL,
  `living` int(11) DEFAULT NULL,
  `symptoms` text DEFAULT NULL,
  `other_symptoms` varchar(255) DEFAULT NULL,
  `conditions` text DEFAULT NULL,
  `other_condition` varchar(255) DEFAULT NULL,
  `medications` enum('yes','no') DEFAULT NULL,
  `allergies` enum('yes','none') DEFAULT NULL,
  `allergy_details` varchar(255) DEFAULT NULL,
  `prenatal_checkup` date DEFAULT NULL,
  `preferred_schedule` datetime DEFAULT NULL,
  `status` enum('pending','approved','rejected','expired') NOT NULL DEFAULT 'pending',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `doctor_id` int(11) DEFAULT NULL,
  `consultation_status` enum('pending','in_progress','completed') DEFAULT 'pending',
  `findings` text DEFAULT NULL,
  `next_schedule` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `pregnancy_forms`
--

INSERT INTO `pregnancy_forms` (`id`, `user_id`, `first_name`, `middle_initial`, `last_name`, `suffix`, `dob`, `gender`, `age`, `marital_status`, `contact_number`, `complete_address`, `email_address`, `pregnant`, `miscarriage`, `multiple`, `weeks`, `edd1`, `edd2`, `gravida`, `births`, `living`, `symptoms`, `other_symptoms`, `conditions`, `other_condition`, `medications`, `allergies`, `allergy_details`, `prenatal_checkup`, `preferred_schedule`, `status`, `created_at`, `updated_at`, `doctor_id`, `consultation_status`, `findings`, `next_schedule`) VALUES
(1, 3, 'Charlotte Bella', 'Luna', 'Nova', '', '1996-12-21', 'Female', 28, '0', '09557321458', 'San Miguel Pasig City', 'char_bella@gmail.com', 'yes', 'no', 'no', 6, '2025-10-15', '0000-00-00', 1, 1, 1, 'None', '', 'none', '', 'no', 'none', '', '2025-06-15', '2025-07-21 10:00:00', 'expired', '2025-07-15 16:19:09', '2025-07-21 09:02:04', NULL, 'pending', NULL, NULL),
(2, 4, 'Sofia Isabella ', 'Forb', 'Connor', '', '1994-08-26', 'Male', 30, '0', '091757783421', 'Pineda Pasig City', 'sofia_bella@gmail.com', NULL, NULL, NULL, 6, '2025-09-19', '0000-00-00', 1, 1, 1, 'None', '', 'none', '', '', '', '', '0000-00-00', '2025-07-18 21:02:00', 'expired', '2025-07-19 11:32:55', '2025-07-19 15:13:51', NULL, 'pending', NULL, NULL),
(3, 3, 'Charlotte Bella', 'Luna', 'Nova', '', '1996-12-21', 'Female', 28, '0', '09557321458', 'San Miguel Pasig City', 'char_bella@gmail.com', 'yes', 'no', 'no', 40, '2025-08-04', '0000-00-00', 2, 2, 2, 'None', '', 'none', '', 'no', 'none', '', '0000-00-00', '2025-08-01 14:00:00', 'expired', '2025-07-28 10:29:01', '2025-08-01 17:09:09', NULL, 'pending', NULL, NULL),
(4, 3, 'Charlotte Bella', 'Luna', 'Nova', '', '1996-12-21', 'Female', 28, '0', '09557321458', 'San Miguel Pasig City', 'char_bella@gmail.com', 'yes', 'no', 'no', 10, '2026-04-07', '0000-00-00', 1, 1, 1, 'nausea/vomiting', '', 'none', '', 'no', 'none', '', '0000-00-00', NULL, 'pending', '2025-07-30 08:01:59', '2025-07-30 08:01:59', NULL, 'pending', NULL, NULL),
(5, 7, 'Lenaj', 'Rolf', 'Rolf', '', '1995-06-10', 'Female', 30, '0', '09544662627', 'San Miguel, Pasig City', 'lenaj_rolf@gmail.com', 'yes', 'no', 'no', 10, '2026-04-07', '0000-00-00', 3, 3, 3, 'Back pain', '', 'none', '', 'no', 'none', '', '0000-00-00', '2025-08-04 08:00:00', 'expired', '2025-08-01 16:02:06', '2025-09-18 11:36:18', NULL, 'pending', NULL, NULL),
(7, 13, 'Angel', 'A.', 'Cruz', 'N/A', '1995-09-09', 'Female', 30, '0', '09670984729', 'Pasig City', 'angel@gmail.com', 'yes', 'no', 'no', 12, '2026-05-03', '0000-00-00', 2, 1, 1, 'nausea/vomiting', '', 'hypertension', '', 'no', 'yes', '', '2025-10-31', '2025-11-05 09:00:00', 'approved', '2025-11-03 15:07:43', '2025-11-03 15:15:17', 1, 'pending', NULL, NULL),
(8, 14, 'Cristina', 'F.', 'Albindo', 'N/A', '1995-09-09', 'Female', 30, '0', '09205057976', 'Pineda Pasig City', 'cristina@gmail.com', 'yes', 'no', 'no', 12, '2026-05-04', '0000-00-00', 3, 2, 1, 'nausea/vomiting', '', 'asthma', '', 'no', 'none', '', '2025-11-03', '2025-11-04 09:00:00', 'approved', '2025-11-03 17:00:36', '2025-11-03 17:03:41', 1, 'pending', NULL, NULL);

-- --------------------------------------------------------

--
-- Table structure for table `screening_data`
--

CREATE TABLE `screening_data` (
  `id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `first_name` varchar(50) DEFAULT NULL,
  `middle_initial` varchar(5) DEFAULT NULL,
  `last_name` varchar(50) DEFAULT NULL,
  `suffix` varchar(10) DEFAULT NULL,
  `dob` date DEFAULT NULL,
  `gender` varchar(10) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `marital_status` varchar(20) DEFAULT NULL,
  `contact_number` varchar(20) DEFAULT NULL,
  `emergency_contact` varchar(50) DEFAULT NULL,
  `relationship` varchar(30) DEFAULT NULL,
  `address` text DEFAULT NULL,
  `guardian_name` varchar(50) DEFAULT NULL,
  `guardian_relationship` varchar(30) DEFAULT NULL,
  `guardian_contact` varchar(20) DEFAULT NULL,
  `medical_conditions` text DEFAULT NULL,
  `medical_others` text DEFAULT NULL,
  `had_surgery` varchar(10) DEFAULT NULL,
  `surgery_type` varchar(100) DEFAULT NULL,
  `surgery_year` varchar(10) DEFAULT NULL,
  `recent_hospitalization` varchar(10) DEFAULT NULL,
  `medications` text DEFAULT NULL,
  `allergies_meds` text DEFAULT NULL,
  `allergies_foods` text DEFAULT NULL,
  `foods_allergy_details` text DEFAULT NULL,
  `symptoms_selected` text DEFAULT NULL,
  `symptoms_others` text DEFAULT NULL,
  `temperature` varchar(10) DEFAULT NULL,
  `blood_pressure` varchar(15) DEFAULT NULL,
  `height` varchar(10) DEFAULT NULL,
  `weight` varchar(10) DEFAULT NULL,
  `bmi` varchar(10) DEFAULT NULL,
  `medical_concern` varchar(30) DEFAULT NULL,
  `returning_patient` varchar(5) DEFAULT NULL,
  `preferred_schedule` datetime DEFAULT NULL,
  `schedule_time` datetime DEFAULT NULL,
  `health_concern` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `status` enum('pending','approved','rejected') NOT NULL DEFAULT 'pending',
  `doctor_id` int(11) DEFAULT NULL,
  `consultation_status` enum('pending','in_progress','completed') DEFAULT 'pending',
  `findings` text DEFAULT NULL,
  `prescription` text DEFAULT NULL,
  `consultation_mode` enum('online','face_to_face') NOT NULL DEFAULT 'face_to_face',
  `meeting_id` int(11) DEFAULT NULL,
  `meeting_url` varchar(512) DEFAULT NULL,
  `patient_join_token` char(64) DEFAULT NULL,
  `doctor_join_token` char(64) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `screening_data`
--

INSERT INTO `screening_data` (`id`, `user_id`, `first_name`, `middle_initial`, `last_name`, `suffix`, `dob`, `gender`, `email`, `marital_status`, `contact_number`, `emergency_contact`, `relationship`, `address`, `guardian_name`, `guardian_relationship`, `guardian_contact`, `medical_conditions`, `medical_others`, `had_surgery`, `surgery_type`, `surgery_year`, `recent_hospitalization`, `medications`, `allergies_meds`, `allergies_foods`, `foods_allergy_details`, `symptoms_selected`, `symptoms_others`, `temperature`, `blood_pressure`, `height`, `weight`, `bmi`, `medical_concern`, `returning_patient`, `preferred_schedule`, `schedule_time`, `health_concern`, `created_at`, `status`, `doctor_id`, `consultation_status`, `findings`, `prescription`, `consultation_mode`, `meeting_id`, `meeting_url`, `patient_join_token`, `doctor_join_token`) VALUES
(2, 2, 'Marcus Joshua', 'Levi', 'William', 'Jr.', '2000-08-21', 'Male', 'marcjosh@gmail.com', '', '09351457855', '', '', 'San Miguel Pasig City', '', '', '', 'None', '', 'No', '', '0', '', '', '', '', NULL, 'Headache', '', '', '', '', '', '', 'Check-up', 'No', '2025-07-01 08:00:00', NULL, 'Extreme headache for almost 4 weeks', '2025-07-14 19:03:24', '', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(3, 3, 'Charlotte Bella', 'Luna', 'Nova', '', '1996-12-21', 'Female', 'char_bella@gmail.com', '', '09557321458', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', NULL, 'Fatigue', '', '', '', '', '', '', 'Check-up', 'No', '2025-09-18 19:08:00', NULL, 'I am experiencing fatigue in the past 2 months', '2025-07-15 05:50:19', 'approved', 1, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(4, 1, 'Sean Andrew', 'Owen', 'Jackson', '', '1996-12-21', 'Male', 'seanandrew@gmail.com', '', '09557436598', '', '', 'Pineda ', '', '', '', 'Bowel/Bladder, Diabetes, Anxiety Attack', '', 'Yes', 'Appendectomy', '2020', 'St. Luke\'s', 'Metformin, Aspirin', 'Penicillin', 'Peanut', NULL, 'Cough, Chills', '', '36.9', '150/100', '160', '70', '', 'Check-up', 'Yes', '2020-01-20 13:30:00', '2020-01-20 13:30:00', '1 week cough', '2025-07-16 05:39:46', 'approved', 1, 'completed', 'Chills, Cough', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(5, 4, 'Sofia Isabella ', 'Forb', 'Connor', '', '1994-08-26', 'Male', 'sofia_bella@gmail.com', 'Single', '091757783421', '', '', 'Pineda Pasig City', '', '', '', 'Array', '', 'Yes', 'Caesarean', '2025', '', '', '', 'Yes', 'Peanut', 'Diarrhea', '', '', '', '', '', '', 'Check-up', 'No', '2025-07-21 14:00:00', NULL, 'I experience it for almost 1 month.', '2025-07-18 06:49:00', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(6, 5, 'Mary Christine', 'Joy', 'Malapad', '', '2004-09-12', 'Female', 'joyjoy@gmail.com', 'Single', '09558741355', '', '', 'San Miguel Pasig City', '', '', '', 'Array', '', '', '', '', '', '', '', '', '', 'Diarrhea', '', '', '', '', '', '', 'Check-up', 'No', '2025-07-29 11:00:00', NULL, 'For almost 3 weeks', '2025-07-21 17:20:40', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(7, 5, 'Mary Christine', 'Joy', 'Malapad', '', '2004-09-12', 'Female', 'joyjoy@gmail.com', 'Single', '09558741355', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Diarrhea', '', '', '', '', '', '', 'Check-up', 'No', '2025-07-29 11:00:00', NULL, 'Experiencing for almost 3 weeks', '2025-07-22 03:40:57', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(8, 6, 'Joshua Rafael ', 'Calv', 'Blanc', 'Jr.', '2007-12-12', 'Male', 'j.rafael@gmail.com', 'Single', '09465587562', 'Ilona Blanc', 'Older Sister', 'San Miguel Pasig City', 'Rochelle Blanc', 'Mother', '09887541123', 'Breathing Problems,Sleep Apnea', '', '', '', '0', '', '', '', 'Yes', 'Shrimp', 'Fever', '', '', '', '', '', '', 'Check-up', 'No', '0000-00-00 00:00:00', NULL, 'Fever', '2025-07-28 05:13:17', 'pending', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(9, 3, 'Charlotte Bella', 'Luna', 'Nova', '', '1996-12-21', 'Female', 'char_bella@gmail.com', '', '09557321458', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Skin Rash', '', '', '', '', '', '', 'Check-up', 'No', '2025-09-18 19:08:00', NULL, 'skin rashes', '2025-07-28 06:11:45', 'approved', 1, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(10, 7, 'Lenaj', 'Rolf', 'Rolf', '', '1995-06-10', 'Female', 'lenaj_rolf@gmail.com', 'Married', '09544662627', '', '', 'San Miguel, Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Shortness of Breath', '', '', '', '156', '50', '20.55', 'Check-up', 'No', '2025-08-02 08:00:00', NULL, '', '2025-08-01 15:55:12', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(11, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Vomiting', '', '34', '128', '162', '60', '42', '', 'No', '2025-09-17 17:01:00', NULL, 'wala lang naman hehe', '2025-09-10 06:51:45', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(12, 8, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Chest Pain, Vomiting', '', '', '', '', '', '', 'Check-up', 'No', '2025-09-13 17:11:00', NULL, 'wala', '2025-09-10 07:55:21', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(13, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Diarrhea', '', '', '', '', '', '', '', 'No', '2025-09-17 17:01:00', NULL, 'hehe', '2025-09-10 07:58:05', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(14, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Sore Throat', '', '', '', '', '', '', '', 'No', '2025-09-17 17:01:00', NULL, 'hindi naman talaga', '2025-09-10 08:07:34', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(15, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Runny Nose', '', '', '', '', '', '', '', 'No', '2025-09-17 17:01:00', NULL, 'wala lang', '2025-09-10 09:02:19', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(16, 9, '', '', '', '', '0000-00-00', '', '', '', '', '', '', '', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', '', '', '', '', '', '', '', '', '', '2025-09-17 17:01:00', NULL, '', '2025-09-10 09:02:54', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(17, 9, '', '', '', '', '0000-00-00', '', '', '', '', '', '', '', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', '', '', '', '', '', '', '', '', '', '2025-09-17 17:01:00', NULL, '', '2025-09-10 09:03:18', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(18, 9, '', '', '', '', '0000-00-00', '', '', '', '', '', '', '', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', '', '', '', '', '', '', '', '', '', '2025-09-17 17:01:00', NULL, '', '2025-09-10 09:03:25', 'approved', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(19, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Chills', '', '32', '140', '172', '65', '52', '', 'No', '2025-09-23 10:30:00', '2025-09-23 10:30:00', 'hindi naman siguro try lang eh', '2025-09-16 09:00:40', 'approved', 1, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(20, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Vomiting', '', '', '', '', '', '', '', 'No', '2025-09-23 14:00:00', '2025-09-23 14:00:00', 'wala naman', '2025-09-19 12:44:37', 'approved', 1, 'completed', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(21, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Chills, Vomiting', '', '34', '128', '172', '65', '52', '', 'No', '2025-09-24 14:30:00', '2025-09-24 14:30:00', 'wala hehe', '2025-09-24 05:52:28', 'approved', 1, 'completed', 'Patient presents with chills and recurrent vomiting (non-bloody, non-bilious), onset ~8–12 hrs prior to consult. Reports low-grade fever and nausea; denies severe abdominal pain, diarrhea, hematemesis, melena, dysuria, flank pain, headache, or rash. Able to tolerate small sips of fluid. Urine output acceptable.\r\n\r\nVitals: T 38.2°C, HR 96, BP 110/70, RR 18, SpO2 98% RA.\r\nExam: Alert, no acute distress. Mucous membranes slightly dry; cap refill <2s. Abdomen soft, non-distended, non-tender; no guarding/rebound; no CVA tenderness. Neuro grossly intact; skin warm, no petechiae.\r\nImpression: Mild dehydration; otherwise stable.', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(22, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Diarrhea', '', '35', '145', '182', '67', '62', '', 'No', '2025-09-25 13:30:00', '2025-09-25 13:30:00', 'hindi ko alam yan', '2025-09-24 05:59:16', 'approved', 1, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(23, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Sore Throat', '', '35', '148', '165', '62', '63', '', 'No', '2025-09-24 16:30:00', '2025-09-24 16:30:00', 'wala naman talaga', '2025-09-24 07:39:51', 'approved', 1, 'completed', 'red, swollen tonsils, swollen lymph nodes', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(24, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Chills', '', '35', '128', '178', '72', '62', '', 'No', '2025-09-26 15:30:00', '2025-09-26 15:30:00', 'wala hehe', '2025-09-26 06:52:04', 'approved', 1, 'completed', 'Patient reports sudden onset chills beginning 1–2 days prior to consult. Associated subjective fever, body aches, mild headache. No cough, no sore throat, no diarrhea, no dysuria, no rash, no recent travel or known sick contacts reported.\r\nVitals (today): Temp 38.4 °C (febrile), BP 110/70 mmHg, HR 96 bpm, RR 18/min.\r\nGeneral: Ill-appearing but non-toxic, oriented, good skin turgor, no respiratory distress.\r\nHEENT: Non-erythematous oropharynx, no tonsillar exudates.\r\nChest/Lungs: Clear to auscultation bilaterally.\r\nCV: Regular rate/rhythm, no murmurs.\r\nAbdomen: Soft, non-tender, no guarding.\r\nSkin: No petechiae, no rashes.\r\nAssessment: Febrile illness with chills, no localizing signs on exam.', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(25, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Vomiting', '', '35', '140', '172', '65', '52', '', 'No', '2025-09-30 14:30:00', '2025-09-30 14:30:00', 'try lang daw pero wala naman talaga', '2025-09-27 05:22:53', 'approved', 1, 'completed', 'Foul-smelling vomit: suggests a distal intestinal obstruction. \r\nProjectile vomiting: can indicate increased intracranial pressure.', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(26, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Chills', '', '', '', '', '', '', '', 'No', '0000-00-00 00:00:00', NULL, 'wala naman try lang', '2025-10-08 09:25:10', 'pending', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(27, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Vomiting', '', '', '', '', '', '', '', 'No', '2025-10-09 14:00:00', '2025-10-09 14:00:00', 'wala naman talaga hehe', '2025-10-08 10:04:29', 'approved', 1, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(28, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Diarrhea', '', '', '', '', '', '', '', 'No', '2025-10-09 13:30:00', '2025-10-09 13:30:00', 'wala naman', '2025-10-08 10:28:43', 'approved', 1, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(29, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Diarrhea, Chills', '', '', '', '', '', '', '', 'No', '2025-10-09 15:30:00', '2025-10-09 15:30:00', 'wala talaga hehe', '2025-10-08 10:30:25', 'approved', 1, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(30, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Runny Nose', '', '', '', '', '', '', '', 'No', '2025-10-09 13:00:00', '2025-10-09 13:00:00', 'wala', '2025-10-08 10:34:29', 'approved', 1, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(31, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Diarrhea', '', '', '', '', '', '', '', 'No', '2025-10-10 13:00:00', '2025-10-10 13:00:00', 'wala hehe', '2025-10-08 10:50:50', 'approved', 1, 'pending', NULL, NULL, 'online', 1, NULL, NULL, NULL),
(32, 11, 'josephine', 'C', 'Celo', 'N/A', '1962-05-13', 'Female', 'josephine@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind Rizal', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Diarrhea, Chills', '', '', '', '', '', '', '', 'No', '2025-10-09 15:00:00', '2025-10-09 15:00:00', 'wala naman talaga yan hehe', '2025-10-08 11:01:32', 'approved', 1, 'pending', NULL, NULL, 'online', 2, NULL, NULL, NULL),
(33, 11, 'josephine', 'C', 'Celo', 'N/A', '1962-05-13', 'Female', 'josephine@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind Rizal', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Chills, Vomiting', '', '', '', '', '', '', '', 'No', '2025-10-09 16:00:00', '2025-10-09 16:00:00', 'wala naman ', '2025-10-09 07:41:42', 'approved', 1, 'pending', NULL, NULL, 'online', 3, NULL, NULL, NULL),
(34, 11, 'josephine', 'C', 'Celo', 'N/A', '1962-05-13', 'Female', 'josephine@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind Rizal', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Runny Nose, Chills', '', '', '', '', '', '', '', 'No', '2025-10-09 16:30:00', '2025-10-09 16:30:00', 'wala', '2025-10-09 08:20:16', 'approved', 1, 'pending', NULL, NULL, 'online', 5, NULL, NULL, NULL),
(35, 11, 'josephine', 'C', 'Celo', 'N/A', '1962-05-13', 'Female', 'josephine@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind Rizal', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Diarrhea, Vomiting', '', '', '', '', '', '', '', 'No', '2025-10-10 13:30:00', '2025-10-10 13:30:00', 'wala hehe', '2025-10-09 09:07:22', 'approved', 1, 'pending', NULL, NULL, 'online', NULL, 'https://meet.jit.si/econsulta-7212df26e0d61319', '2f7f7579cb74f670747140bf2c176a9a', '9913912696f8582432d0fab72c92e35e'),
(36, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Sore Throat, Vomiting', '', '', '', '', '', '', '', 'No', '2025-10-10 14:00:00', '2025-10-10 14:00:00', 'wala naman talaga', '2025-10-09 09:35:10', 'approved', 1, 'pending', NULL, NULL, 'online', NULL, 'https://meet.jit.si/econsulta-bfd0c1c49c488726', 'd9fb97032b218efa4ae0173a876fb752', 'ca3152a33ee12027f0e291bd3c0ecfc2'),
(37, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Sore Throat', '', '', '', '', '', '', '', 'No', '2025-10-14 13:00:00', '2025-10-14 13:00:00', 'wala naman', '2025-10-10 09:11:03', 'approved', 1, 'pending', NULL, NULL, 'online', NULL, 'https://meet.jit.si/econsulta-1cb28fca9c7026bb', '4be9b4dd3cdd10e10769c8f6c7858443', '82a6be060caba5d5adf66a7675fd6bff'),
(38, 11, 'josephine', 'C', 'Celo', 'N/A', '1962-05-13', 'Female', 'josephine@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind Rizal', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Diarrhea', '', '34', '128', '172', '65', '62', '', 'No', '2025-10-14 14:00:00', '2025-10-14 14:00:00', 'wala naman talaga yan', '2025-10-10 10:19:15', 'approved', 1, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(39, 9, 'Rodrigo', 'L.', 'Celo', 'N/A', '1957-11-05', 'Male', 'rodrigo@gmail.com', 'Married', '09662301403', '09212853000', '', 'Eastwind', '', '', '', '', '', 'No', '', '0', '', '', 'No', 'Yes', 'fish', 'Chills, Vomiting', '', '', '', '', '', '', '', 'No', '2025-10-17 16:30:00', '2025-10-17 16:30:00', 'wala naman hehe sabi lang ni joy', '2025-10-17 08:59:21', 'approved', 1, 'pending', NULL, NULL, 'online', NULL, 'https://meet.jit.si/econsulta-6aeb72271515635f', '4ac0e8a56d638e437ddbfc905248a436', '8611ed08b2e4eed7ba87f5c8790e0856'),
(40, 5, 'Mary Christine', 'Joy', 'Malapad', '', '2004-09-12', 'Female', 'joyjoy@gmail.com', 'Single', '09558741355', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Fever, Cough, Headache, Muscle Pain', '', '', '', '', '', '', 'Check-up', 'No', '2025-10-20 09:00:00', '2025-10-20 09:00:00', 'High fever', '2025-10-17 12:14:19', 'approved', 1, 'completed', 'Dengue Fever', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(41, 5, 'Mary Christine', 'Joy', 'Malapad', '', '2004-09-12', 'Female', 'joyjoy@gmail.com', 'Single', '09558741355', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Shortness of Breath', '', '37.4', '120/80', '', '', '', 'Check-up', 'No', '2025-10-20 13:00:00', '2025-10-20 13:00:00', 'Wheezing, shortness of breath', '2025-10-17 12:26:59', 'approved', 1, 'completed', 'Bronchial Asthma', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(42, 5, 'Mary Christine', 'Joy', 'Malapad', '', '2004-09-12', 'Female', 'joyjoy@gmail.com', 'Single', '09558741355', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Fever, Chills', '', '39.2', '100/70', '162', '54', '', 'Check-up', 'No', '2025-10-20 10:00:00', '2025-10-20 10:00:00', 'High fever', '2025-10-17 12:31:57', 'approved', 1, 'completed', 'Dengue Fever', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(43, 5, 'Mary Christine', 'Joy', 'Malapad', '', '2004-09-12', 'Female', 'joyjoy@gmail.com', 'Single', '09558741355', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Headache, Others', 'dizziness', '39.2', '100/70', '160', '70', '', 'Check-up', 'No', '2024-01-17 10:00:00', '2024-01-17 10:00:00', 'dizziness and Headache', '2025-10-17 12:48:51', 'approved', 1, 'completed', 'Headache, dizziness', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(44, 5, 'Mary Christine', 'Joy', 'Malapad', '', '2004-09-12', 'Female', 'joyjoy@gmail.com', 'Single', '09558741355', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Fever, Headache, Muscle Pain', '', '39.2', '100/70', '156', '54', '', 'Check-up', 'No', '2025-10-13 13:30:00', '2025-10-13 13:30:00', 'High Fever', '2025-10-18 07:10:20', 'approved', 1, 'completed', 'High Fever', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(45, 5, 'Mary Christine', 'Joy', 'Malapad', '', '2004-09-12', 'Female', 'joyjoy@gmail.com', 'Single', '09558741355', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Fever', '', '39.2', '100/70', '156', '54', '', 'Check-up', 'No', '2025-10-15 10:00:00', '2025-10-15 10:00:00', 'High Fever', '2025-10-18 07:17:06', 'approved', 1, 'completed', 'High Fever', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(46, 5, 'Mary Christine', 'Joy', 'Malapad', '', '2004-09-12', 'Female', 'joyjoy@gmail.com', 'Single', '09558741355', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Fever', '', '39.2', '100/70', '156', '54', '', 'Check-up', 'No', '2025-10-16 13:00:00', '2025-10-16 13:00:00', 'Fever', '2025-10-18 07:24:44', 'approved', 1, 'completed', 'High Fever', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(47, 5, 'Mary Christine', 'Joy', 'Malapad', '', '2004-09-12', 'Female', 'joyjoy@gmail.com', 'Single', '09558741355', '', '', 'San Miguel Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Headache, Chills', '', '39.2', '100/70', '156', '54', '', 'Check-up', 'No', '2025-10-20 15:00:00', '2025-10-20 15:00:00', 'Chills', '2025-10-18 07:26:53', 'approved', 1, 'completed', 'Chills', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(48, 1, 'Sean Andrew', 'Owen', 'Jackson', '', '1996-12-21', 'Male', 'seanandrew@gmail.com', '', '09557436598', '', '', 'Pineda ', '', '', '', '', '', 'Yes', 'Appendectomy', '2020', 'St. Luke\'s', '', 'Yes', 'Yes', 'Peanut', 'Cough, Headache, Chills', '', '', '', '', '', '', 'Check-up', 'No', '2025-10-21 09:30:00', '2025-10-21 09:30:00', 'Chills', '2025-10-18 07:29:19', 'approved', 1, 'completed', 'Fever', NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(49, 12, 'Che', 'B', 'CheB', 'N/A', '2004-09-09', 'Female', 'cheche@gmail.com', 'Single', '09670984729', '09786868691', '', 'Pasig', 'Samantha Nicole', 'Mother', '09684838701', '', '', '', '', '', '', '', '', '', '', 'Fever', '', '', '', '', '', '', 'Check-up', 'No', NULL, NULL, '1 week na akong inuubo', '2025-11-03 13:57:17', 'pending', NULL, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(50, 13, 'Angel', 'A.', 'Cruz', 'N/A', '1995-09-09', 'Female', 'angel@gmail.com', 'Married', '09670984729', '09786868691', '', 'Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Cough, Headache', '', '', '', '', '', '', 'Check-up', 'No', '2025-12-02 11:00:00', '2025-12-02 11:00:00', 'Inuubo ako isang linggo na.', '2025-11-03 15:04:50', 'approved', 1, 'in_progress', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL),
(51, 14, 'Cristina', 'F.', 'Albindo', 'N/A', '1995-09-09', 'Female', 'cristina@gmail.com', 'Married', '09205057976', '09786868691', '', 'Pineda Pasig City', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Fever, Headache', '', '', '', '', '', '', 'Check-up', 'No', '2025-11-04 10:30:00', '2025-11-04 10:30:00', 'Inuubo ako limang araw na.', '2025-11-03 16:57:49', 'approved', 1, 'pending', NULL, NULL, 'face_to_face', NULL, NULL, NULL, NULL);

--
-- Triggers `screening_data`
--
DELIMITER $$
CREATE TRIGGER `trg_screening_update_schedule` AFTER UPDATE ON `screening_data` FOR EACH ROW BEGIN
  IF NEW.schedule_time <> OLD.schedule_time THEN
    UPDATE doctor_notes
    SET schedule_time = NEW.schedule_time
    WHERE screening_id = NEW.id;
  END IF;
END
$$
DELIMITER ;

-- --------------------------------------------------------

--
-- Table structure for table `specializations`
--

CREATE TABLE `specializations` (
  `id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `stock_history`
--

CREATE TABLE `stock_history` (
  `id` int(11) NOT NULL,
  `medicine_id` int(11) NOT NULL,
  `change_type` enum('IN','OUT') NOT NULL,
  `quantity` int(11) NOT NULL,
  `remarks` varchar(255) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `tbdots`
--

CREATE TABLE `tbdots` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `first_name` varchar(255) NOT NULL,
  `middle_initial` varchar(10) DEFAULT NULL,
  `last_name` varchar(255) NOT NULL,
  `dob` date NOT NULL,
  `contact_number` varchar(20) DEFAULT NULL,
  `gender` varchar(20) NOT NULL,
  `tb_case_number` varchar(100) NOT NULL,
  `tb_type` varchar(100) NOT NULL,
  `category` varchar(10) NOT NULL,
  `treatment_start` date NOT NULL,
  `sputum_result` varchar(50) NOT NULL,
  `hiv_status` varchar(50) NOT NULL,
  `treatment_phase` varchar(100) NOT NULL,
  `dots_provider` varchar(255) DEFAULT NULL,
  `remarks` text DEFAULT NULL,
  `preferred_schedule` datetime DEFAULT NULL,
  `doctor_id` int(11) DEFAULT NULL,
  `status` enum('pending','approved','rejected','expired') DEFAULT NULL,
  `consultation_status` enum('pending','in_progress','completed') DEFAULT 'pending',
  `findings` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `tbdots`
--

INSERT INTO `tbdots` (`id`, `user_id`, `first_name`, `middle_initial`, `last_name`, `dob`, `contact_number`, `gender`, `tb_case_number`, `tb_type`, `category`, `treatment_start`, `sputum_result`, `hiv_status`, `treatment_phase`, `dots_provider`, `remarks`, `preferred_schedule`, `doctor_id`, `status`, `consultation_status`, `findings`, `created_at`, `updated_at`) VALUES
(1, 4, 'Sofia Isabella ', 'Forb', 'Connor', '1994-08-26', NULL, 'Male', 'TB-2025-001', 'Pulmonary / Extra-pulmonary', 'I', '2025-01-07', 'Positive', 'Negative', 'Intensive', 'Dr. Santos', '', '2025-09-03 08:30:00', NULL, 'expired', 'pending', NULL, '2025-07-18 19:30:43', '2025-11-03 13:47:38'),
(2, 3, 'Charlotte Bella', 'Luna', 'Nova', '1996-12-21', NULL, 'Female', 'TB-2025-001', 'Pulmonary / Extra-pulmonary', 'I', '2025-01-07', 'Positive', 'Negative', 'Intensive', 'Dr. Santos', '', NULL, NULL, 'pending', 'pending', NULL, '2025-07-21 08:58:53', '2025-07-21 08:58:53'),
(3, 3, 'Charlotte Bella', 'Luna', 'Nova', '1996-12-21', NULL, 'Female', 'TB-2025-001', 'Pulmonary / Extra-pulmonary', 'I', '2025-07-07', 'Positive', 'Negative', 'Intensive', '', '', '2025-07-30 11:45:00', NULL, 'expired', 'pending', NULL, '2025-07-28 15:28:05', '2025-08-01 17:17:46'),
(4, 3, 'Charlotte Bella', 'Luna', 'Nova', '1996-12-21', NULL, 'Female', 'TB-2025-001', 'Pulmonary', 'I', '2025-07-07', 'Positive', 'Negative', 'Intensive', '', '', NULL, NULL, 'pending', 'pending', NULL, '2025-07-30 08:56:01', '2025-07-30 08:56:01'),
(5, 7, 'Lenaj', 'Rolf', 'Rolf', '1995-06-10', NULL, 'Female', 'TB-2025-001', 'Pulmonary / Extra-pulmonary', 'I', '2025-07-07', 'Positive', 'Negative', 'Intensive', '', '', '2025-08-15 13:00:00', NULL, 'expired', 'pending', NULL, '2025-08-01 16:12:58', '2025-11-03 13:47:38'),
(6, 13, 'Angel', 'A.', 'Cruz', '1995-09-09', '09670984729', 'Female', 'TB 0839', 'Pulmonary', 'II', '2025-10-28', 'Negative', 'Negative', 'Intensive Phase', '', '', NULL, NULL, 'rejected', 'pending', NULL, '2025-11-03 15:52:39', '2025-11-03 15:52:59'),
(7, 13, 'Angel', 'A.', 'Cruz', '1995-09-09', '09670984729', 'Female', 'TB 0839', 'Pulmonary', 'II', '2025-10-22', 'Negative', 'Negative', 'Intensive Phase', '', '', '2025-11-13 10:30:00', 1, 'approved', 'pending', NULL, '2025-11-03 15:55:34', '2025-11-03 15:55:51'),
(8, 14, 'Cristina', 'F.', 'Albindo', '1995-09-09', '09205057976', 'Female', 'TB 0839', 'Pulmonary', 'II', '2025-11-03', 'Negative', 'Negative', 'Intensive Phase', '', '', '2025-11-04 14:00:00', 1, 'approved', 'pending', NULL, '2025-11-03 17:02:19', '2025-11-03 17:05:18');

-- --------------------------------------------------------

--
-- Table structure for table `temp_machine_learning`
--

CREATE TABLE `temp_machine_learning` (
  `year` int(11) DEFAULT NULL,
  `month` varchar(50) DEFAULT NULL,
  `month_num` int(11) DEFAULT NULL,
  `disease` varchar(100) DEFAULT NULL,
  `cases` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `email` varchar(50) NOT NULL,
  `password` varchar(255) NOT NULL,
  `first_name` varchar(50) NOT NULL,
  `last_name` varchar(50) NOT NULL,
  `middle_initial` varchar(5) DEFAULT NULL,
  `suffix` varchar(10) DEFAULT NULL,
  `address` text NOT NULL,
  `contact_number` varchar(20) NOT NULL,
  `years_of_residency` int(11) NOT NULL,
  `age` int(11) NOT NULL,
  `dob` date DEFAULT NULL,
  `gender` varchar(10) DEFAULT NULL,
  `marital_status` enum('Single','Married','Divorced','Widowed','Engaged') DEFAULT NULL,
  `emergency_contact` varchar(50) DEFAULT NULL,
  `proof_of_identity` varchar(255) DEFAULT NULL,
  `status` enum('pending','approved','rejected') DEFAULT 'pending',
  `role` enum('regular','senior') NOT NULL DEFAULT 'regular',
  `is_senior_citizen` enum('yes','no') DEFAULT 'no'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `email`, `password`, `first_name`, `last_name`, `middle_initial`, `suffix`, `address`, `contact_number`, `years_of_residency`, `age`, `dob`, `gender`, `marital_status`, `emergency_contact`, `proof_of_identity`, `status`, `role`, `is_senior_citizen`) VALUES
(1, 'seanandrew@gmail.com', 'Seanandrew@2025', 'Sean Andrew', 'Jackson', 'Owen', '', 'San Miguel, Pasig City', '09557436598', 20, 20, '1996-12-21', 'Male', NULL, NULL, 'uploads/postal_id.png', 'approved', 'regular', 'no'),
(2, 'marcjosh@gmail.com', 'Marcjosh@2025', 'Marcus Joshua', 'William', 'Levi', 'Jr.', 'San Miguel Pasig City', '09351457855', 25, 25, '2000-08-21', 'Male', NULL, NULL, 'uploads/postal_id.png', 'approved', 'regular', 'no'),
(3, 'char_bella@gmail.com', 'Charlotte@2025', 'Charlotte Bella', 'Nova', 'Luna', '', 'San Miguel Pasig City', '09557321458', 29, 29, '1996-12-21', 'Female', NULL, NULL, 'uploads/postal_id.png', 'approved', 'regular', 'no'),
(4, 'sofia_bella@gmail.com', 'Sofiabella@2025', 'Sofia Isabella ', 'Connor', 'Forb', '', 'Pineda Pasig City', '091757783421', 30, 30, '1994-08-26', 'Male', 'Single', NULL, 'uploads/postal_id.png', 'approved', 'regular', 'no'),
(5, 'joyjoy@gmail.com', 'Joyjoy_2025', 'Mary Christine', 'Malapad', 'Joy', '', 'San Miguel Pasig City', '09558741355', 20, 20, '2004-09-12', 'Female', 'Single', NULL, 'uploads/postal_id.png', 'approved', 'regular', 'no'),
(6, 'j.rafael@gmail.com', 'Joshuarafael@2025', 'Joshua Rafael ', 'Blanc', 'Calv', 'Jr.', 'San Miguel Pasig City', '09465587562', 17, 17, '2007-12-12', 'Male', 'Single', NULL, 'uploads/postal_id.png', 'approved', 'regular', 'no'),
(7, 'lenaj_rolf@gmail.com', 'Lenaj@2025', 'Lenaj', 'Rolf', 'Rolf', '', 'San Miguel, Pasig City', '09544662627', 30, 30, '1995-06-10', 'Female', 'Married', NULL, 'uploads/postal_id.png', 'approved', 'regular', 'no'),
(8, 'celojosephrodney@gmail.com', 'Joseph23!', 'Joseph Rodney', 'Celo', 'C.', 'N/A', '165 banaag st Pineda Pasig City', '09212853000', 10, 22, '2002-12-26', 'Male', 'Single', '09662301403', 'uploads/5fe3acc3-e06c-4ed2-ac69-6121d9292ac0.jfif', 'approved', 'regular', 'no'),
(9, 'rodrigo@gmail.com', 'Rodrigo23!', 'Rodrigo', 'Celo', 'L.', 'N/A', 'Eastwind', '09662301403', 10, 68, '1957-11-05', 'Male', 'Married', '09212853000', 'uploads/c5e94841-013e-454a-bedd-aa8d4b2523ab.jfif', 'approved', 'regular', 'yes'),
(10, 'celojohnmichael@gmail.com', 'Johncelo23!', 'John Michael', 'Celo', 'C.', 'N/A', 'Rizal', '09662301403', 15, 35, '1989-04-14', 'Male', 'Single', '09212853000', 'uploads/5fe3acc3-e06c-4ed2-ac69-6121d9292ac0.jfif', 'approved', 'regular', 'no'),
(11, 'josephine@gmail.com', 'Josephine23!', 'josephine', 'Celo', 'C', 'N/A', 'Eastwind Rizal', '09662301403', 10, 64, '1962-05-13', 'Female', 'Married', '09212853000', 'uploads/c5e94841-013e-454a-bedd-aa8d4b2523ab.jfif', 'approved', 'regular', 'yes'),
(13, 'angel@gmail.com', 'Angel@2025', 'Angel', 'Cruz', 'A.', 'N/A', 'Pasig City', '09670984729', 30, 30, '1995-09-09', 'Female', 'Married', '09786868691', 'uploads/ValidID.png', 'approved', 'regular', 'no'),
(14, 'cristina@gmail.com', 'Cristina@2025', 'Cristina', 'Albindo', 'F.', 'N/A', 'Pineda Pasig City', '09205057976', 30, 30, '1995-09-09', 'Female', 'Married', '09786868691', 'uploads/ValidID.png', 'approved', 'regular', 'no');

-- --------------------------------------------------------

--
-- Table structure for table `user_medications`
--

CREATE TABLE `user_medications` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `med_name` varchar(255) DEFAULT NULL,
  `med_dose` varchar(255) DEFAULT NULL,
  `med_frequency` varchar(255) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `user_medications`
--

INSERT INTO `user_medications` (`id`, `user_id`, `med_name`, `med_dose`, `med_frequency`, `created_at`) VALUES
(1, 1, 'Paracetamol', '500mg', 'Twice a day', '2025-07-08 13:45:20'),
(2, 1, 'Vitamin C', '1000mg', 'Once a day', '2025-07-08 13:45:20'),
(3, 1, 'Ibuprofen', '200mg', 'As needed', '2025-07-08 13:45:20'),
(4, 1, '', '', '', '2025-07-08 14:19:39'),
(5, 1, 'Metformin', '300', 'Twice a day', '2025-07-09 15:53:51'),
(6, 1, 'Metformin', '300', 'Twice a day', '2025-07-09 15:55:12'),
(7, 1, 'Metformin', '300', 'Twice a day', '2025-07-09 16:05:54'),
(8, 1, 'Metformin', '300', 'Twice a day', '2025-07-16 05:09:35'),
(9, 1, 'Aspirin', '200', 'Once a day', '2025-07-16 05:09:35'),
(10, 4, 'Losartan', '50mg', 'Once daily (evening)', '2025-07-18 05:34:29'),
(11, 4, 'Metformin', '500mg', 'Twice a day', '2025-07-18 05:34:29'),
(12, 9, 'Paracetamol', '300', 'sometimes', '2025-09-10 09:00:54'),
(13, 9, 'Vitamins', '100', 'sometimes', '2025-09-10 09:00:54');

-- --------------------------------------------------------

--
-- Table structure for table `video_meetings`
--

CREATE TABLE `video_meetings` (
  `id` int(11) NOT NULL,
  `provider` enum('jitsi','zoom','gmeet') NOT NULL DEFAULT 'jitsi',
  `room_slug` varchar(120) NOT NULL,
  `url` varchar(255) NOT NULL,
  `scheduled_start` datetime NOT NULL,
  `scheduled_end` datetime DEFAULT NULL,
  `status` enum('scheduled','started','ended','cancelled') NOT NULL DEFAULT 'scheduled',
  `screening_id` int(11) NOT NULL,
  `created_by` int(11) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `video_meetings`
--

INSERT INTO `video_meetings` (`id`, `provider`, `room_slug`, `url`, `scheduled_start`, `scheduled_end`, `status`, `screening_id`, `created_by`, `created_at`) VALUES
(1, 'jitsi', 'CONSULT-T3T7HO-380B6012881D', 'https://meet.jit.si/CONSULT-T3T7HO-380B6012881D', '2025-10-10 13:00:00', '0000-00-00 00:00:00', 'scheduled', 31, 0, '2025-10-08 10:51:24'),
(2, 'jitsi', 'CONSULT-T3T7Z8-87CC44369DA7', 'https://meet.jit.si/CONSULT-T3T7Z8-87CC44369DA7', '2025-10-09 15:00:00', '0000-00-00 00:00:00', 'scheduled', 32, 0, '2025-10-08 11:01:56'),
(3, 'jitsi', 'CONSULT-T3UTE5-400422664CF3', 'https://meet.jit.si/CONSULT-T3UTE5-400422664CF3', '2025-10-09 16:00:00', '0000-00-00 00:00:00', 'scheduled', 33, 0, '2025-10-09 07:42:05'),
(4, 'jitsi', '1b1a8f6538ba6831734b', '', '2025-10-09 16:30:00', '2025-10-09 16:30:00', 'scheduled', 34, 0, '2025-10-09 08:20:44'),
(5, 'jitsi', 'CONSULT-T3UV6K-76B7666B81AD', 'https://meet.jit.si/CONSULT-T3UV6K-76B7666B81AD', '2025-10-09 16:30:00', '0000-00-00 00:00:00', 'scheduled', 34, 0, '2025-10-09 08:20:44');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `admin`
--
ALTER TABLE `admin`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `announcements`
--
ALTER TABLE `announcements`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `chat_messages`
--
ALTER TABLE `chat_messages`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_sender` (`sender_id`,`sender_type`),
  ADD KEY `idx_receiver` (`receiver_id`,`receiver_type`);

--
-- Indexes for table `doctors`
--
ALTER TABLE `doctors`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indexes for table `doctor_notes`
--
ALTER TABLE `doctor_notes`
  ADD PRIMARY KEY (`id`),
  ADD KEY `screening_id` (`screening_id`),
  ADD KEY `doctor_id` (`doctor_id`);

--
-- Indexes for table `doctor_schedule`
--
ALTER TABLE `doctor_schedule`
  ADD PRIMARY KEY (`id`),
  ADD KEY `doctor_id` (`doctor_id`);

--
-- Indexes for table `family_planning`
--
ALTER TABLE `family_planning`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_fp_user` (`user_id`);

--
-- Indexes for table `feedback`
--
ALTER TABLE `feedback`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `infants_consultation`
--
ALTER TABLE `infants_consultation`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_user_id` (`user_id`),
  ADD KEY `fk_infant_doctor` (`doctor_id`);

--
-- Indexes for table `login_history`
--
ALTER TABLE `login_history`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_type` (`user_type`),
  ADD KEY `login_time` (`login_time`);

--
-- Indexes for table `machine_learning`
--
ALTER TABLE `machine_learning`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `unique_disease_month` (`year`,`month_num`,`disease`);

--
-- Indexes for table `medical_history`
--
ALTER TABLE `medical_history`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `medicines`
--
ALTER TABLE `medicines`
  ADD PRIMARY KEY (`medicine_id`);

--
-- Indexes for table `medicine_requests`
--
ALTER TABLE `medicine_requests`
  ADD PRIMARY KEY (`request_id`),
  ADD KEY `patient_id` (`patient_id`),
  ADD KEY `medicine_id` (`medicine_id`);

--
-- Indexes for table `meeting_tokens`
--
ALTER TABLE `meeting_tokens`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `token` (`token`),
  ADD KEY `meeting_id` (`meeting_id`);

--
-- Indexes for table `messages`
--
ALTER TABLE `messages`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `oral_services`
--
ALTER TABLE `oral_services`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `fk_oralservices_doctor` (`doctor_id`);

--
-- Indexes for table `password_reset`
--
ALTER TABLE `password_reset`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `personal_medical_history`
--
ALTER TABLE `personal_medical_history`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `pregnancy_forms`
--
ALTER TABLE `pregnancy_forms`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `fk_preg_doctor` (`doctor_id`);

--
-- Indexes for table `screening_data`
--
ALTER TABLE `screening_data`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_doctor` (`doctor_id`);

--
-- Indexes for table `specializations`
--
ALTER TABLE `specializations`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `stock_history`
--
ALTER TABLE `stock_history`
  ADD PRIMARY KEY (`id`),
  ADD KEY `medicine_id` (`medicine_id`);

--
-- Indexes for table `tbdots`
--
ALTER TABLE `tbdots`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_user` (`user_id`),
  ADD KEY `fk_tbdots_doctor` (`doctor_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `user_id` (`email`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indexes for table `user_medications`
--
ALTER TABLE `user_medications`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `video_meetings`
--
ALTER TABLE `video_meetings`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `room_slug` (`room_slug`),
  ADD KEY `screening_id` (`screening_id`),
  ADD KEY `scheduled_start` (`scheduled_start`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `admin`
--
ALTER TABLE `admin`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `announcements`
--
ALTER TABLE `announcements`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `chat_messages`
--
ALTER TABLE `chat_messages`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT for table `doctors`
--
ALTER TABLE `doctors`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `doctor_notes`
--
ALTER TABLE `doctor_notes`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=17;

--
-- AUTO_INCREMENT for table `doctor_schedule`
--
ALTER TABLE `doctor_schedule`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=13;

--
-- AUTO_INCREMENT for table `family_planning`
--
ALTER TABLE `family_planning`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;

--
-- AUTO_INCREMENT for table `feedback`
--
ALTER TABLE `feedback`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `infants_consultation`
--
ALTER TABLE `infants_consultation`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `login_history`
--
ALTER TABLE `login_history`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=33;

--
-- AUTO_INCREMENT for table `machine_learning`
--
ALTER TABLE `machine_learning`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=524;

--
-- AUTO_INCREMENT for table `medical_history`
--
ALTER TABLE `medical_history`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `medicines`
--
ALTER TABLE `medicines`
  MODIFY `medicine_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=20;

--
-- AUTO_INCREMENT for table `medicine_requests`
--
ALTER TABLE `medicine_requests`
  MODIFY `request_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=23;

--
-- AUTO_INCREMENT for table `meeting_tokens`
--
ALTER TABLE `meeting_tokens`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `messages`
--
ALTER TABLE `messages`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT for table `oral_services`
--
ALTER TABLE `oral_services`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT for table `password_reset`
--
ALTER TABLE `password_reset`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `personal_medical_history`
--
ALTER TABLE `personal_medical_history`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=28;

--
-- AUTO_INCREMENT for table `pregnancy_forms`
--
ALTER TABLE `pregnancy_forms`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT for table `screening_data`
--
ALTER TABLE `screening_data`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=52;

--
-- AUTO_INCREMENT for table `specializations`
--
ALTER TABLE `specializations`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `stock_history`
--
ALTER TABLE `stock_history`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `tbdots`
--
ALTER TABLE `tbdots`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=15;

--
-- AUTO_INCREMENT for table `user_medications`
--
ALTER TABLE `user_medications`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=14;

--
-- AUTO_INCREMENT for table `video_meetings`
--
ALTER TABLE `video_meetings`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `doctor_notes`
--
ALTER TABLE `doctor_notes`
  ADD CONSTRAINT `doctor_notes_ibfk_1` FOREIGN KEY (`screening_id`) REFERENCES `screening_data` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `doctor_notes_ibfk_2` FOREIGN KEY (`doctor_id`) REFERENCES `doctors` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `doctor_schedule`
--
ALTER TABLE `doctor_schedule`
  ADD CONSTRAINT `doctor_schedule_ibfk_1` FOREIGN KEY (`doctor_id`) REFERENCES `doctors` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `family_planning`
--
ALTER TABLE `family_planning`
  ADD CONSTRAINT `fk_fp_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `infants_consultation`
--
ALTER TABLE `infants_consultation`
  ADD CONSTRAINT `fk_infant_doctor` FOREIGN KEY (`doctor_id`) REFERENCES `doctors` (`id`) ON DELETE SET NULL,
  ADD CONSTRAINT `fk_user_id` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `medicine_requests`
--
ALTER TABLE `medicine_requests`
  ADD CONSTRAINT `medicine_requests_ibfk_1` FOREIGN KEY (`patient_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `medicine_requests_ibfk_2` FOREIGN KEY (`medicine_id`) REFERENCES `medicines` (`medicine_id`) ON DELETE CASCADE;

--
-- Constraints for table `meeting_tokens`
--
ALTER TABLE `meeting_tokens`
  ADD CONSTRAINT `meeting_tokens_ibfk_1` FOREIGN KEY (`meeting_id`) REFERENCES `video_meetings` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `oral_services`
--
ALTER TABLE `oral_services`
  ADD CONSTRAINT `fk_oralservices_doctor` FOREIGN KEY (`doctor_id`) REFERENCES `doctors` (`id`) ON DELETE SET NULL,
  ADD CONSTRAINT `oral_services_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

--
-- Constraints for table `password_reset`
--
ALTER TABLE `password_reset`
  ADD CONSTRAINT `password_reset_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `personal_medical_history`
--
ALTER TABLE `personal_medical_history`
  ADD CONSTRAINT `personal_medical_history_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `pregnancy_forms`
--
ALTER TABLE `pregnancy_forms`
  ADD CONSTRAINT `fk_preg_doctor` FOREIGN KEY (`doctor_id`) REFERENCES `doctors` (`id`),
  ADD CONSTRAINT `pregnancy_forms_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `screening_data`
--
ALTER TABLE `screening_data`
  ADD CONSTRAINT `fk_doctor` FOREIGN KEY (`doctor_id`) REFERENCES `doctors` (`id`);

--
-- Constraints for table `stock_history`
--
ALTER TABLE `stock_history`
  ADD CONSTRAINT `stock_history_ibfk_1` FOREIGN KEY (`medicine_id`) REFERENCES `medicines` (`medicine_id`);

--
-- Constraints for table `tbdots`
--
ALTER TABLE `tbdots`
  ADD CONSTRAINT `fk_tbdots_doctor` FOREIGN KEY (`doctor_id`) REFERENCES `doctors` (`id`) ON DELETE SET NULL,
  ADD CONSTRAINT `fk_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

--
-- Constraints for table `user_medications`
--
ALTER TABLE `user_medications`
  ADD CONSTRAINT `user_medications_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
