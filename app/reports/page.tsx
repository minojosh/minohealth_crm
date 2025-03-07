"use client";

import { Card, CardBody } from "@heroui/card";
import { Divider } from "@heroui/divider";
import { Button } from "@heroui/button";
import { Select, SelectItem } from "@heroui/select";

export default function Reports() {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Reports & Differential Diagnosis</h1>
        <div className="flex gap-2">
          <Select
            label="Report Type"
            placeholder="Select a report type"
            className="w-[200px]"
          >
            <SelectItem key="diagnosis" value="diagnosis">Diagnosis Reports</SelectItem>
            <SelectItem key="patient" value="patient">Patient Reports</SelectItem>
            <SelectItem key="appointments" value="appointments">Appointment Reports</SelectItem>
          </Select>
          <Button color="primary">Generate Report</Button>
        </div>
      </div>
      <Divider />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardBody>
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-semibold">Diagnosis Distribution</h2>
              <div className="h-[300px] flex items-center justify-center border rounded-lg">
                <p className="text-default-500">Chart will be displayed here</p>
              </div>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardBody>
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-semibold">Treatment Outcomes</h2>
              <div className="h-[300px] flex items-center justify-center border rounded-lg">
                <p className="text-default-500">Chart will be displayed here</p>
              </div>
            </div>
          </CardBody>
        </Card>
      </div>
      <Card>
        <CardBody>
          <div className="flex flex-col gap-4">
            <h2 className="text-xl font-semibold">Recent Differential Diagnoses</h2>
            <div className="space-y-4">
              <div className="p-4 border rounded-lg">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="font-medium">Case #1234</h3>
                    <p className="text-sm text-default-500">Patient: John Doe</p>
                  </div>
                  <Button size="sm" variant="light">View Details</Button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                  <div className="p-3 bg-default-100 rounded">
                    <h4 className="text-sm font-medium mb-1">Primary Diagnosis</h4>
                    <p className="text-sm text-default-500">Pending analysis</p>
                  </div>
                  <div className="p-3 bg-default-100 rounded">
                    <h4 className="text-sm font-medium mb-1">Differential Diagnoses</h4>
                    <p className="text-sm text-default-500">Pending analysis</p>
                  </div>
                  <div className="p-3 bg-default-100 rounded">
                    <h4 className="text-sm font-medium mb-1">Confidence Score</h4>
                    <p className="text-sm text-default-500">N/A</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}