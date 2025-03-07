export type SiteConfig = typeof siteConfig;

export const siteConfig = {
  name: "MinoHealth CRM",
  description: "Comprehensive Healthcare Management System",
  navItems: [
    {
      label: "Dashboard",
      href: "/",
    },
    {
      label: "Appointment Manager",
      href: "/appointment-manager",
    },
    {
      label: "Data Extractor",
      href: "/data-extractor",
    },
    {
      label: "Reports",
      href: "/reports",
    },
    {
      label: "Appointment Scheduler",
      href: "/appointment-scheduler",
    },
  ],
  links: {
    github: "#",
    twitter: "#",
    docs: "#",
    discord: "#",
    sponsor: "#",
  },
};
