"use client";
import React from 'react';
import { motion } from "framer-motion";
import { siteConfig } from "@/config/site";
import { title, subtitle } from "@/components/primitives";

export default function Home() {
  const fadeIn = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  };
  
  const staggerChildren = {
    animate: {
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  return (
    <section className="flex flex-col items-center justify-center gap-4 py-8 md:py-10 min-h-[80vh]">
      {/* Medical-themed gradient background */}
      <div className="absolute inset-0 -z-10 bg-gradient-to-br from-teal-50 via-white to-blue-50 overflow-hidden">
        <motion.div 
          className="absolute top-0 left-0 w-1/2 h-1/2 bg-gradient-to-br from-teal-200/30 to-transparent rounded-full blur-3xl"
          animate={{ 
            x: [0, 10, 0], 
            y: [0, 15, 0],
          }}
          transition={{ 
            repeat: Infinity, 
            duration: 8,
            ease: "easeInOut" 
          }}
        />
        <motion.div 
          className="absolute bottom-0 right-0 w-1/2 h-1/2 bg-gradient-to-br from-blue-200/30 to-transparent rounded-full blur-3xl"
          animate={{ 
            x: [0, -10, 0], 
            y: [0, -15, 0],
          }}
          transition={{ 
            repeat: Infinity, 
            duration: 10,
            ease: "easeInOut" 
          }}
        />
      </div>

      <motion.div 
        className="inline-block max-w-xl text-center justify-center"
        initial="initial"
        animate="animate"
        variants={staggerChildren}
      >
        <motion.div variants={fadeIn}>
          <span className={title()}>Introducing&nbsp;</span>
          <span className={title({ color: "violet" })}>minoHealth CRM&nbsp;</span>
        </motion.div>
        
        <motion.div 
          variants={{
            initial: { opacity: 0, y: 20 },
            animate: { opacity: 1, y: 0, transition: { delay: 0.4, duration: 0.6 } }
          }}
        >
          <span className={title()}>
            Automate The Boring Work with minoHealth CRM
          </span>
        </motion.div>

        <motion.div 
          className={subtitle({ class: "mt-8" })}
          variants={{
            initial: { opacity: 0, scale: 0.95 },
            animate: { opacity: 1, scale: 1, transition: { delay: 0.8, duration: 0.5 } }
          }}
        >
          Streamline patient management, automate appointment scheduling, and enhance 
          clinical workflows with AI-powered healthcare solutions.
        </motion.div>

        <motion.div 
          className="flex gap-4 mt-8 justify-center"
          variants={{
            initial: { opacity: 0, y: 20 },
            animate: { opacity: 1, y: 0, transition: { delay: 1.2, duration: 0.5 } }
          }}
        >
            <motion.button 
            className="px-6 py-3 bg-gradient-to-r from-blue-500 to-teal-400 text-white rounded-full font-semibold hover:shadow-lg"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => window.location.href = '/appointment-manager'}
            >
            Get Started
            </motion.button>
          {/* <motion.button 
            className="px-6 py-3 bg-white border border-gray-400 text-slate-700 rounded-full font-semibold hover:shadow-lg"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
          >
            Learn More
          </motion.button> */}
        </motion.div>
      </motion.div>

      {/* Floating medical icons */}
      <motion.div 
        className="absolute -z-5 opacity-10"
        animate={{
          y: [0, 10, 0],
        }}
        transition={{
          repeat: Infinity,
          duration: 5,
          ease: "easeInOut"
        }}
      >
        <svg width="600" height="600" viewBox="0 0 600 600" className="stroke-blue-500 stroke-1 fill-none">
          <path d="M300,100 L300,500 M100,300 L500,300" strokeWidth="4" />
          <circle cx="300" cy="300" r="150" />
        </svg>
      </motion.div>
    </section>
  );
}